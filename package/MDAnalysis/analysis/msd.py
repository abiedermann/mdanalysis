# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- http://www.mdanalysis.org
# Copyright (c) 2006-2016 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
"""
MSD --- :mod:`MDAnalysis.analysis.msd`
===============================================================================

A tool for computing mean squared displacement and diffusion coefficients

"""
from six.moves import range, zip

import inspect
import logging
import numpy as np
import six
import sqlite3
import pandas as pd
from scipy.optimize import curve_fit
from scipy.sparse import lil_matrix, vstack

from MDAnalysis.core.universe import Universe
from .base import AnalysisBase
from MDAnalysis import coordinates
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.log import ProgressMeter

logger = logging.getLogger("MDAnalysis.analysis.MSD")
# declaring static conversion from A^2/ps to cm^2/s
_conversion = float(10**-4)/(2) # note: divide by dim


class MSD(AnalysisBase):
    """
    Steps:
        - Log info in a relational database / python array
        - During post processing, calculate MSD across database data
        - Return csv file with msd vs time data
    """


    #==========================================================================
    # Constructor
    #==========================================================================
    def __init__(self, universe, select_list, start,
                 stop, step=1, begin_fit=None, end_fit=None,
                 timestep_size=1, n_bootstraps=1000, dim=3, quiet=True,
                 database_name="msd.db", process_trajectory=True,
                 min_event_length=0, sim_sample_limit=None, adjust=None):
        """
        Parameters
        ----------
        universe : mda.Universe
            MDAnalysis Universe object
        select_list : list
            list of selection strings
        start : int
            start frame of analysis
        stop : int
            stop frame of analysis
        step : int, optional (default=1)
            number of frames to skip between each analysed frame
        begin_fit : list
            list of fit interval beginnings, corresponding to select_list order
        end_fit : list
            list of fit interval endings, corresponding to select_list order
        timestep_size : float
            number of ps per frame (used for MSD estimation)
        n_bootstraps : int
            number of bootstraps used for estimating standard deviation of
            the diffusion coefficient
        dim : int
            number of dimensions of diffusion
        quiet : bool, optional
            Turn off verbosity
        database_name : str, optional
            name of permanent database file
        process_trajectory : bool, optional
            determines whether to process trajectory or read from an already
            generated database specified by database_name
        min_event_length : int, optional
            minimum number of sampled frames necessary for an event to be included
            in diffusion calculation (helps to significantly reduce computational
            expense caused by molecules quickly diffusing across a boundary)
        sim_sample_limit : int, optional
            specifies the maximum number of diffusion events to follow at any time,
            allowing one to effectively subsample from a large number of molecules
        """
        self._quiet = quiet
        self.universe = universe
        if process_trajectory:
            self._setup_frames(universe.trajectory, start, stop, step)
        else:
            self.start = start
            self.stop = stop
            self.step = step
        self.select_list =  select_list
        self._process_trajectory = process_trajectory
        self.timestep_size = timestep_size
        self.begin_fit = begin_fit
        self.end_fit = end_fit
        self.n_bootstraps = n_bootstraps
        self._min_event_length = min_event_length
        self._sim_sample_limit = sim_sample_limit
        self.dim = dim
        self.adjust = adjust
        # initializing dictionaries for event log creation
        self._current_events = [dict() for i in range(len(select_list))]
        
        # selection x time
        self.n_steps = (self.stop-self.start)/self.step + 1
        self.MSD = np.zeros([len(self.select_list),self.n_steps],dtype=float)
        self.total_samples = np.zeros([len(self.select_list),self.n_steps],dtype=float)
        self.MSDs = [lil_matrix([],dtype=float) for i in range(len(self.select_list))]
        self.MSDs_annotate = [[] for i in range(len(self.select_list))]
        self.n_samples = [lil_matrix([],dtype=int) for i in range(len(self.select_list))]

        self.D = [0.0 for i in range(len(self.select_list))]
        self.err_D = [0.0 for i in range(len(self.select_list))]

        # initialize database file
        self.database_name = database_name
        self._conn = sqlite3.connect(self.database_name)
        self._c = self._conn.cursor()
        self.event_log_table_name = "Event_Log"
        self.trajectory_table_name = "Trajectory"

    #==========================================================================
    # Handle atom group selections
    #==========================================================================
    def _select(self):
        """Generates list of atom groups that satisfy select_list strings

        Returns
        -------
        selections : list
            list of atom groups
        """
        selections = []
        for sel in self.select_list:
            selections.append(self.universe.select_atoms(sel))
            if selections[-1].n_atoms == 0:
                selections[-1] = None
        return selections


    #==========================================================================
    # Log data in database
    #==========================================================================
    def _single_frame(self):
        """Calculate data from a single frame of trajectory"""
        selections = self._select()
        for i, sel in enumerate(selections):
            if sel is not None:
                # Update Event Log Table
                atoms_in_sel = set(sel.atoms.ids)
                atoms_in_current_events = set(self._current_events[i].keys())

                # Add atoms which are in sel but not in current events
                # to current events
                new_events = atoms_in_sel - atoms_in_current_events
                for new_atom in new_events:
                    if self._sim_sample_limit is None:
                        self._current_events[i][new_atom] = self._frame_index
                    elif len(atoms_in_current_events) <= self._sim_sample_limit:
                        self._current_events[i][new_atom] = self._frame_index

                # Log and remove atoms which are in current events but not in
                # sel
                event_terminations = atoms_in_current_events - atoms_in_sel
                for event in event_terminations:
                    # avoiding logging trivial diffusion events
                    if (float(self._frame_index-self._current_events[i][event])/self.step >
                        self._min_event_length):
                        datastr = """INSERT INTO {tn} (atomnr, start, stop,
                            classification) VALUES ({an}, {start}, {stop}, 
                            {classification})""".format(
                            tn=self.event_log_table_name, an=event,
                            start=self._current_events[i][event],
                            stop=self._frame_index-self.step, classification=i)
                        self._c.execute(datastr)
                    self._current_events[i].pop(event)
        
            # Log all ongoing events at the end of the trajectory
            if(self._frame_index == self.stop-self.step):
                event_terminations = set(self._current_events[i].keys())
                for event in event_terminations:
                    # avoid logging trivial diffusion events
                    if (float(self._frame_index-self._current_events[i][event])/self.step >
                        self._min_event_length):
                        #if(self._frame_index != self._current_events[i][event]):
                        datastr = """INSERT INTO {tn} (atomnr, start, stop,
                            classification) VALUES ({an}, {start}, {stop}, 
                            {classification})""".format(
                            tn=self.event_log_table_name, an=event,
                            start=self._current_events[i][event],
                            stop=self._frame_index, classification=i)
                        self._c.execute(datastr)
                    self._current_events[i].pop(event)

        # Update Trajectory Table (avoiding duplication)
        all_atoms = set.union(*[set(sel.atoms) for sel in selections])
        for atom in all_atoms:
            self._c.execute("""INSERT INTO {tn} (atomnr, time, x,
                 y, z) VALUES ({an}, {t}, {x}, {y}, {z})""".format(
                tn=self.trajectory_table_name,
                an=atom.id,t=self._ts.frame, x=atom.position[0],
                y=atom.position[1], z=atom.position[2]))
        return


    #==========================================================================
    # Setup Database
    #==========================================================================
    def _prepare(self):
        """Sets up database tables before the analysis loop begins."""
        # classification is the index of the corresponding selection in
        # select_list
        self._c.execute("""CREATE TABLE {table_name} (atomnr INT, start INT,
                           stop INT, classification TINYINT)""".format(
                        table_name=self.event_log_table_name))
        self._c.execute("""CREATE TABLE {table_name} (atomnr INT, time INT,
                           x REAL, y REAL, z REAL)""".format(
                        table_name=self.trajectory_table_name))
        return


    #==========================================================================
    # Using Fast Fourier transform code from:
    # http://stackoverflow.com/questions/34222272/
    # computing-mean-square-displacement-using-python-and-fft
    @staticmethod                                                              
    def autocorrFFT(x):                    
        N=len(x)                                      
        F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding   
        PSD = F * F.conjugate()                                      
        res = np.fft.ifft(PSD)                                                
        res= (res[:N]).real   #now we have the autocorrelation in convention B 
        n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)                  
        return res/n #this is the autocorrelation in convention A              

    @staticmethod                                               
    def msd_fft(r):                                                     
        N=len(r)                                                               
        D=np.square(r).sum(axis=1)                                             
        D=np.append(D,0)                                                       
        S2=sum([MSD.autocorrFFT(r[:, i]) for i in range(r.shape[1])])        
        Q=2*D.sum()                                                             
        S1=np.zeros(N)                                                         
        for m in range(N):                                                 
            Q=Q-D[m-1]-D[N-m]                                             
            S1[m]=Q/(N-m)                                                     
        return S1-2*S2
    #==========================================================================


    @staticmethod
    def lin_func(x,D,b):
        """Functional form of fit"""
        return D*x+b

    @staticmethod
    def estimate_diffusion(msd,b,e,timestep_size,step,dim):
        """Returns estimate of diffusion coefficient for each selection
        
        Parameters
        ----------
        msd : list
            estimate of msd at corresponding list indexes
        b : int
            first frame in the fit interval (inclusive)
        e : int
            ending frame of the fit interval (exclusive)

        Returns
        -------
        D : float
            estimate of diffusion coefficient (cm^2/s)
        """
        xdata = [timestep_size*step*i for i in range(b,e)]
        popt, pcov = curve_fit(MSD.lin_func,xdata,
                               msd[b:e])
        return popt[0]*_conversion/dim


    def estimate_err_diffusion(self,msds,num_samples,b,e,n_events):
        """Bootstraps MSDs to estimate uncertainty in diffusion coefficient

        Parameters
        ----------
        msds : 2d numpy array, np.float32
            array of msd estimates for individual events
        num_samples : 2d numpy array, np.int32
            number of samples associated with corresponding samples in msds
        b : int
            first frame in the fit interval (inclusive)
        e : int
            ending frame of the fit interval (exclusive)
        
        Returns
        -------
        std_D : float
            Returns bootstrapped standard deviation in diffusion coefficient
            estimate
        """
        est_D = [] # list of diffusion coefficient estimates
        # generating bootstrapped MSDs, then storing diffusion estimates
        for interation in range(self.n_bootstraps):
            # pick randomly from msd with replacement
            indices = np.random.randint(0,n_events,n_events)
            eMSDs = vstack([msds.getrow(indices[n]) for n
                            in range(n_events)])
            en_samples = vstack([num_samples.getrow(indices[n]) for n
                            in range(n_events)])
            # calculate msd from bootstrapped msds
            eMSD = np.array(np.nan_to_num(np.divide((en_samples.multiply(eMSDs)).sum(axis=0),
                                     en_samples.sum(axis=0)))).reshape(self.n_steps)
            # add estimate to list of estimates
            est_D.append(self.estimate_diffusion(eMSD,b,e,self.timestep_size,self.step,self.dim))

        return np.std(np.array(est_D),axis=0,ddof=1)


    #==========================================================================
    # Calculate MSD and diffusion coefficient
    #==========================================================================
    def _conclude(self):
        """Calculate MSD"""
        # For each selection
        for i, sel_item in enumerate(self.select_list):
            # loop though all events with this classification
            data = self._c.execute("""SELECT * FROM {tn} WHERE 
                                       classification={cl}""".format(
                                    tn=self.event_log_table_name,cl=i)).fetchall()
            n_events = len(data)
            print(i)
            print(n_events)
            self.MSDs_annotate[i] = [tuple() for k in range(n_events)]
            # event x time
            self.MSDs[i] = lil_matrix((n_events, self.n_steps), dtype=float)
            # event x time
            self.n_samples[i] = lil_matrix((n_events,self.n_steps), dtype=int)
            for j, event in enumerate(data):
                self.MSDs_annotate[i][j] = event
                # pull data from database into a time X [x,y,z] numpy array
                x = pd.read_sql_query("""SELECT x, y, z FROM {tn} WHERE 
                                   atomnr={an} AND time >= {start} AND
                                   time <= {stop}""".format(
                                   tn=self.trajectory_table_name, an=event[0],
                                   start=event[1], stop=event[2]),
                                   self._conn).as_matrix()
                if self.adjust is not None:
                    if self.adjust[i] is not None:
                        x = x - self.adjust[i][:len(x)]
                msd_data = self.msd_fft(x)
                self.MSDs[i][j] = np.pad(msd_data, (0, self.n_steps
                                         - len(msd_data)), 'constant')

                samples = np.arange(len(msd_data),0,-1,dtype=int)
                self.n_samples[i][j] = np.pad(samples, (0, self.n_steps
                                      - len(samples)), 'constant')

            self.MSDs[i] = self.MSDs[i].tocsr()
            self.n_samples[i] = self.n_samples[i].tocsr()

            self.MSD[i] = np.nan_to_num(np.divide(
                            (self.n_samples[i].multiply(self.MSDs[i])).sum(axis=0),
                                     self.n_samples[i].sum(axis=0)))
            self.total_samples[i] = self.n_samples[i].sum(axis=0)

            # if data for diffusion calculation is specified, calculated diffusion
            # coefficient and uncertainty
            if((self.begin_fit is not None) & (self.end_fit is not None)):
                self.D[i] = self.estimate_diffusion(self.MSD[i],self.begin_fit[i],
                                                    self.end_fit[i],self.timestep_size,
                                                    self.step,self.dim)
                self.err_D[i] = \
                    self.estimate_err_diffusion(self.MSDs[i], self.n_samples[i],
                                                self.begin_fit[i], self.end_fit[i], n_events)
        return

    def run(self):
        """Perform the calculation"""
        logger.info("Starting preparation")
        if self._process_trajectory:
            self._prepare() # note: prepare initializes previous frame data
            logger.info("Reading trajectory data")
            for i, ts in enumerate(
                    self._trajectory[self.start:self.stop:self.step]):
                self._frame_index = i
                self._ts = ts
                logger.info("--> Doing frame {} of {}".format(i+1, self.n_frames))
                self._single_frame()
                # commit database writes every 100 frames
                if i % 100 == 0:
                    self._conn.commit()
                self._pm.echo(self._frame_index)
            logger.info("Indexing Database")
            # Create database index for faster queries
            self._c.execute("CREATE INDEX atom_entry ON {tn} (atomnr, time)".format(
                            tn=self.trajectory_table_name))
            del(self._trajectory)
            del(self.universe)
        self._conn.commit()
        logger.info("Calculating MSD... (this may take several minutes)")
        self._conclude()
        return self
