# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- http://www.MDAnalysis.org
# Copyright (c) 2006-2015 Naveen Michaud-Agrawal, Elizabeth J. Denning, Oliver Beckstein
# and contributors (see AUTHORS for the full list)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

from __future__ import absolute_import, print_function, division

import sys
import os
import getopt
import warnings
from collections import deque

import numpy as np
import pickle
# from six import range

import MDAnalysis
from MDAnalysis.analysis.distances import distance_vector


class MSD(object):
    """Object for calculating Mean Square Displacement (MSD)

    This object can be used to accurately calculate the MSD of dynamic
    (or static) selections (for example: name CL and z > 50, etc.).
    The variables len_msd and dt_restart can be used to mitigate
    computational and memory costs. This allows calculations to scale
    well with simulation time and size--though you must have
    enough RAM to load the trajectory during calculation.

    The algorithm uses all available data in calculation, as the
    calculation incorporates all atoms which have remained in the
    dynamic selection over the interval, tau, from the restart.

    Parameters
    ----------
    universe : object
        Universe object containing trajectory data
    select_list : list
        List of selection strings
    t0 : int
        Time when analysis begins (in frames)
    tf : int
        Time when analysis ends (in frames)
    dt_restart : int
        Spacing of restarts (in frames)
    out_name: str
        Name of pickle output file: (out_name).p
    len_msd : int
        Specifies the maximum tau to be calculated (frames)
        (saves memory in array allocation)
    write_output : bool
        Specifies whether to write pickle file
    max_data : bool
        If true, include tau values after tf in calculation (this is
        useful for avoiding data loss in command line parallelization
        of analysis)

    Returns
    -------
    (out_name) : pickle file
        pickle file in format [msd, n_samples] where msd and n_samples
        have len(select_list) lists containing results, indexed
        according to the value of tau (in frames). (i.e. in order)
    """

    def __init__(self, universe, select_list, t0, tf, dt_restart,
                 out_name='msd', len_msd=250, write_output=True,
                 max_data=False):
        self.universe = universe
        self.select_list = select_list
        if type(select_list) is str:
            self.select_list = select_list.split(',')
        print(self.select_list)
        self.t0 = int(t0)
        self.tf = int(tf)
        self.dt_restart = int(dt_restart)
        self.out_name = out_name
        # handling short simulation cases
        n_frames = int(tf) - int(t0) + 1
        if int(len_msd) > int(n_frames):
            len_msd = int(n_frames)
        self.len_msd = int(len_msd)
        self.write_output = write_output
        self.max_data = max_data

    def _select(self,frame):
        """Generates list of atom groups that satisfy select_list
           strings

        Parameters
        ----------
        frame : int
            number of the current frame

        Returns
        -------
        selections : list
            list of atom groups
        """
        selections = []
        for i in range(len(self.select_list)):
            try:
                selections.append(self.universe.select_atoms(self.select_list[i]))
                assert selections[i].n_atoms > 0
            except:
                try:
                    selections.pop(i)
                except:
                    pass
                selections.append(None)
        
        return selections

    def _init_pos_deque(self):
        """Initializes lists necessary for MSD calculations at the
           first restart.

        Deques with maxlen=len_msd are used to conserve memory.

        Returns:
        --------
        pos : list
            contains position information, organized:
            selection X frame X atom X (x, y, z)
        set_list : list
            contains set of atoms which satisfy each selection at each
            frame, organized:
            selection X frame X set of atom ids
        dict_list : list
            contains dictionaries which map atom ids to corresponding
            pos index, organized:
            selection X frame X dict
        dim : list
            contains box dimensions at each frame, organized:
            frame X box dimensions
        """

        n_sel = len(self.select_list)
        
        # dictionaries with key: (id) and value: (array index)
        # selection X frame X dictionary
        dict_list = [[dict() for j in range(self.len_msd)]
                     for i in range(n_sel)]
        # atom ids which satisfy the selection at each frame
        set_list = [[set() for j in range(self.len_msd)]
                    for i in range(n_sel)]

        # pre-allocate position array
        pos = [[np.array([]) for j in range(self.len_msd)]
               for i in range(n_sel)]
        # pre-allocate array containing box dimensions
        dim = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
               for i in range(self.len_msd)]

        print("Populating deque...")
        for ts in self.universe.trajectory[self.t0:self.t0
                                           + self.len_msd]:
            this_frame = ts.frame
            selections = self._select(this_frame)

            dim[this_frame-self.t0] = ts.dimensions
            for i in range(n_sel):
                if selections[i] is not None: # if there is data
                    pos[i][this_frame-self.t0] = selections[i].positions
                    
                    temp_list = selections[i].atoms.ids
                    # store set of atom ids which satisfy selection
                    set_list[i][this_frame-self.t0] = set(temp_list)
                    # link atom id to array index
                    for j in range(len(temp_list)):
                        dict_list[i][this_frame-self.t0][temp_list[j]] = j
                    
            if this_frame % (self.len_msd/10) == 0:
                print("Frame: "+str(this_frame))
        # converting lists to deques
        dim = deque(dim, maxlen=self.len_msd)
        for i in range(n_sel):
            dict_list[i] = deque(dict_list[i], maxlen=self.len_msd)
            set_list[i] = deque(set_list[i], maxlen=self.len_msd)
            pos[i] = deque(pos[i], maxlen=self.len_msd)
        print("Complete!")

        return pos, set_list, dict_list, dim

    def _update_deques(self, pos, set_list, dict_list, dim, this_restart):
        """Updates lists necessary for MSD calculations

        Returns:
        --------
        pos : list
            contains position information, organized:
            selection X frame X atom X (x, y, z)
        set_list : list
            contains set of atoms which satisfy each selection at each
            frame, organized:
            selection X frame X set of atom ids
        dict_list : list
            contains dictionaries which map atom ids to corresponding
            pos index, organized:
            selection X frame X dict
        dim : list
            contains box dimensions at each frame, organized:
            frame X box dimensions
        """
        # stop updating when there are no more frames to analyze
        top_frame = this_restart + self.len_msd
        calc_cutoff = self.tf
        if self.max_data:
            calc_cutoff = len(self.universe.trajectory) - 1

        if top_frame+self.dt_restart > calc_cutoff:
            # feed in zeros
            dim.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i in range(len(self.select_list)):
                pos[i].append(np.array([]))
                set_list[i].append(set())
                dict_list[i].append(dict())
            return
        
        for ts in self.universe.trajectory[top_frame:top_frame
                                           + self.dt_restart]:
            selections = self._select(ts.frame)
            dim.append(ts.dimensions)

            for i in range(len(selections)):
                if selections[i] is not None: # if there is data
                    pos[i].append(selections[i].positions)

                    temp_list = selections[i].atoms.ids
                    # store set of atom ids which satisfy selection
                    set_list[i].append(set(temp_list))
                    # link atom id to array index
                    temp_dict = dict()
                    for j in range(len(temp_list)):
                        temp_dict[temp_list[j]] = j
                    dict_list[i].append(temp_dict)
                else:
                    pos[i].append(np.array([]))
                    set_list[i].append(set())
                    dict_list[i].append(dict())

    def _process_pos_data(self, pos, set_list, dict_list, dim):
        """Runs MSD calculation and returns data

        Returns:
        --------
        msd : np.array(dtype=float32)
            Contains msd data, organized:
            selection X tau
        n_samples : np.array(dtype=int)
            Contains the number of samples corresponding to each msd
            entry, organized:
            selection X tau
        """
        n_sel = len(pos)

        n_samples = np.zeros((n_sel, self.len_msd), dtype='int')
        msd = np.zeros((n_sel, self.len_msd), dtype='float32')
        
        calc_cutoff = self.tf
        if self.max_data:
            calc_cutoff = len(self.universe.trajectory) - 1

        # for each restart point
        for j in range(self.t0, self.tf+1, self.dt_restart):
            for i in range(n_sel):  # for each selection
                atoms_in_sel = set_list[i][0]  # storing initial set at restart
                for ts in range(j, calc_cutoff+1):  # for each frame after restart
                    # avoid computing irrelevantly long taus
                    if ts-j == self.len_msd:
                        break

                    # updating restart set to exclude any atoms which have
                    # left the selection since the restart point
                    atoms_in_sel = atoms_in_sel.intersection(set_list[i][ts-j])
                    
                    # find mutual atoms at times 0 and ts-j
                    shared0 = [dict_list[i][0][k] for k in atoms_in_sel]
                    shared = [dict_list[i][ts-j][k] for k in atoms_in_sel]
                    
                    # move to next restart if there's nothing to evaluate
                    if len(shared) == 0:
                        break # skip to next restart
                    
                    msd[i][ts-j] = (msd[i][ts-j]*n_samples[i][ts-j] + np.power(
                                    distance_vector(pos[i][0][shared0],
                                                    pos[i][ts-j][shared],
                                                    dim[ts-j]), 2).mean(axis=0)
                                    ) / (n_samples[i][ts-j]+1)
                    n_samples[i][ts-j] += 1
            
            if j % (100) == 0:
                print("Frame: "+str(j))
                if self.write_output:
                    pickle.dump([msd, n_samples], open(self.out_name, 'wb'))

            # Update deques
            self._update_deques(pos, set_list, dict_list, dim, j)

        return msd, n_samples

    def run(self):
        """Analyze trajectory and output pickle object"""

        # check that trajectory length is correct
        assert len(self.universe.trajectory) >= (self.tf-self.t0+1), \
            ("Sample interval exceeds trajectory length. This may result"
             "from choice of t0/tf or insufficient RAM when loading the"
             "trajectory.")

        pos, set_list, dict_list, dim = self._init_pos_deque()

        msd, n_samples = self._process_pos_data(pos, set_list, dict_list, dim)

        if self.write_output:
            pickle.dump([msd, n_samples], open(self.out_name, 'wb'))
        # for testing purposes
        else:
            return [msd, n_samples]


'''
def safe_outdir(outdir):
    """
    Creates outdir and all necessary intermediate directories if
    they don't exist
    """
    dirs = outdir.split('/')
    for i in range(len(dirs)):
    dir_str = '/'
    for j in range(i+1):
        dir_str += dirs[j]+'/'
    if not os.path.exists(dir_str):pwd
        os.system('mkdir '+dir_str)
    return
'''

if __name__ == "__main__":
    """
    Options:
    --------
    -f name of trajectory file
    -s name of tpr file
    --sel string containing selections separated by commas
    -b t0 (frames)
    -e tf (frames)
    --dt number of frames between restarts
    -o output file name (optional)
    --len maximum number of tau frames to calculate (optional)
    --out write output boolean (optional, default=True)
    --max_data include data from after tf if it exists
      this allows for command-line level parallelization without
      data loss. (optional, default=False)
    """
    os.system('')
    # handling input arguments
    this_opts, args = getopt.getopt(sys.argv[1:], "f:s:b:e:o:",
                               ['dt=', 'sel=', 'len=', 'out=', 'max_data'])
    opts = dict()
    for opt, arg in this_opts:
        opts[opt] = arg

    # handling optional parameters
    if "-o" not in opts:
        opts["-o"] = "msd.p"
    if "--len" not in opts:
        opts["--len"] = 250
    if "--max_data" in opts:
        opts["--max_data"] = True
    else:
        opts["--max_data"] = False
    if "--out" in opts:
        if 'False' in opts['--out'] or 'false' in opts['--out']:
            opts['--out'] = False
    if "--out" not in opts:
        opts['--out'] = True

    u = MDAnalysis.Universe(opts['-s'], opts['-f'])

    # run calculation
    # safe_outdir(outdir)
    this_MSD = MSD(u, opts['--sel'], opts['-b'], opts['-e'], opts['--dt'],
                   out_name=opts['-o'], len_msd=opts['--len'],
                   write_output=opts['--out'],max_data=opts["--max_data"])
    this_MSD.run()
