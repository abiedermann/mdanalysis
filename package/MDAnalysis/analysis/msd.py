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
Analysis building blocks --- :mod:`MDAnalysis.analysis.base`
============================================================

A collection of useful building blocks for creating Analysis
classes.

"""
from six.moves import range, zip

import inspect
import logging
import numpy as np
import six
import sqlite3
import pandas as pd
from scipy.optimize import curve_fit

from MDAnalysis.core.universe import Universe
from .base import AnalysisBase
from MDAnalysis import coordinates
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.log import ProgressMeter

logger = logging.getLogger("MDAnalysis.analysis.MSD")


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
    def __init__(self, universe, select_list, start=None,
                 stop=None, step=None, begin_fit=0, end_fit=None,
                 timestep_size=None, n_bootstraps=1000, dim=3, quiet=True,
                 database_name="msd.db", process_trajectory=True):
        """
        Parameters
        ----------
        universe : mda.Universe
            MDAnalysis Universe object
        select_list : list
            list of selection strings
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        quiet : bool, optional
            Turn off verbosity
        """
        self._quiet = quiet
        self.universe = universe
        self._setup_frames(universe.trajectory, start, stop, step)
        self.select_list =  select_list
        self._process_trajectory = process_trajectory
        self.timestep_size = timestep_size
        self.begin_fit = begin_fit
        self.end_fit = end_fit
        self.n_bootstraps = n_bootstraps

        self._conversion = float(10**-4)/(2*dim)

        # initializing dictionaries for event log creation
        self._current_events = [dict() for i in range(len(select_list))]
        
        # selection x time
        self.n_steps = self.stop-self.start/self.step
        self.MSD = np.zeros([len(self.select_list),self.n_steps],dtype=float)
        self.MSDs = [np.array([]) for i in range(len(self.select_list))]
        self.n_samples = [np.array([]) for i in range(len(self.select_list))]

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
                    self._current_events[i][new_atom] = self._frame_index

                # Log and remove atoms which are in current events but not in
                # sel
                event_terminations = atoms_in_current_events - atoms_in_sel
                for event in event_terminations:
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
        # set up database tables
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
    # Calculate MSD
    #==========================================================================

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
                                                                  
    def msd_fft(self, r):                                                     
        N=len(r)                                                               
        D=np.square(r).sum(axis=1)                                             
        D=np.append(D,0)                                                       
        S2=sum([self.autocorrFFT(r[:, i]) for i in range(r.shape[1])])        
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

    def estimate_diffusion(self,msd):
        """Returns estimate of diffusion coefficient for each selection"""
        xdata = [self.timestep_size*self.step*i for i in range(self.begin_fit,
                                                               self.end_fit)]
        for i, sel_item in enumerate(self.select_list):
            popt, pcov = curve_fit(self.lin_func,xdata,
                                   msd[self.begin_fit:self.end_fit])
        return popt[0]*self._conversion

    def estimate_err_diffusion(self,msd,num_samples):
        """Bootstraps MSDs to estimate uncertainty in diffusion coefficient"""
        est_D = []
        n_events = len(msd)
        eMSDs = np.zeros([n_events,self.n_steps],dtype=np.float32)
        en_samples = np.zeros([n_events,self.n_steps],dtype=np.int32)
    
        # generating bootstrapped MSDs, then storing diffusion estimates
        for b in range(self.n_bootstraps):
            indices = np.random.randint(0,n_events,n_events)
            for n in range(n_events):
                eMSDs[n] = msd[indices[n]]
                en_samples[n] = num_samples[indices[n]]
            eMSD = np.divide((en_samples*eMSDs).sum(axis=0),
                                     en_samples.sum(axis=0))
            est_D.append(self.estimate_diffusion(eMSD))
        return np.std(np.array(est_D),axis=0,ddof=1)


    def _conclude(self):
        """Calculate MSD"""
        # For each selection
        for i, sel_item in enumerate(self.select_list):
            # loop though all events with this classification
            data = self._c.execute("""SELECT * FROM {tn} WHERE 
                                       classification={cl}""".format(
                                    tn=self.event_log_table_name,cl=i)).fetchall()
            n_events = len(data)
            # event x time
            self.MSDs[i] = np.zeros([n_events,self.n_steps],
                            dtype=float)
            # event x time
            self.n_samples[i] = np.zeros([n_events,self.n_steps])
            for j, event in enumerate(data):
                # pull data from database into a time X [x,y,z] numpy array
                x = pd.read_sql_query("""SELECT x, y, z FROM {tn} WHERE 
                                   atomnr={an} AND time >= {start} AND
                                   time <= {stop} ORDER BY time""".format(
                                   tn=self.trajectory_table_name, an=event[0],
                                   start=event[1], stop=event[2]),
                                   self._conn).as_matrix()
                #print(x)
                msd_data = self.msd_fft(x)
                #print("MSD DATA:")
                #print msd_data
                #print('\n')
                self.MSDs[i][j] = np.pad(msd_data, (0, self.n_steps
                                         - len(msd_data)), 'constant')
                samples = np.arange(len(msd_data),0,-1,dtype=int)
                self.n_samples[i][j] = np.pad(samples, (0, self.n_steps
                                      - len(samples)), 'constant')
            # calculating MSD using weighted average
            #print self.n_samples[i]
            #print self.n_samples[i]*self.MSDs[i]
            #print self.n_samples[i].sum(axis=0)
            # Don't print divide by zeros errors (you will throw this data
            # out later
            self.MSD[i] = np.divide((self.n_samples[i]*self.MSDs[i]).sum(axis=0),
                                     self.n_samples[i].sum(axis=0))
            
            self.D[i] = self.estimate_diffusion(self.MSD[i])
            self.err_D[i] = self.estimate_err_diffusion(self.MSDs[i],
                                                        self.n_samples[i])
            # Debugging:
            #my_axis = [2*k for k in range(len(self.MSD[i]))]
            #plt.plot(my_axis,self.MSD[i])
            #plt.show()
        return

    def run(self):
        """Perform the calculation"""
        logger.info("Starting preparation")
        if self._process_trajectory:
            self._prepare() # note: prepare initializes previous frame dat
            for i, ts in enumerate(
                    self._trajectory[self.start:self.stop:self.step]):
                self._frame_index = i
                self._ts = ts
                # logger.info("--> Doing frame {} of {}".format(i+1, self.n_frames))
                self._single_frame()
                self._pm.echo(self._frame_index)
            logger.info("Finishing up")
        self._conn.commit()
        self._conclude()
        return self

