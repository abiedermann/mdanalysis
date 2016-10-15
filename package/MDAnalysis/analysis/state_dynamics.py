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

"""
state_dynamics --- :mod:'MDAnalysis.analysis.state_dynamics'
==============================================================

:Author: Andrew Biedermann
:Year: 2016
:Copyright: GNU Public License v3

This module implements an extension of the stable-states residence time
calculation proposed by Laage and Hynes. It logs and characterizes pairing
events, outputting data which can be used to estimate the mean residence time
(MRT) of a selected atom group with respect to a reference atom group. The
calculation is sensitive to the number of reference atoms involved in each
pairing event, allowing one to analyze the influence of higher order pairing
interactions on the transport of atoms in the selected atom group. An example
of how to calculate the MRT from output data is included in comments within
the state_dynamics.py module.

For convenience, command line options are included. See the main method of
the state_dynamics.py module for python script implementation.

state_dynamics command line usage
---------------------------------

This generates data which can be used to calculate the MRT of Cl with respect
to Na:

    >>> ./state_dynamics.py -f mysim.trr -s mysim.tpr --sel 'name CL'
        --ref 'name NA' --cut1=3.0 --cut2=4.0 --max_order=1 -b 0 -e 50000
        -o mrt_data.p

By specifying --max_order=1, pairing events will be categorized as events
where the Cl atom is paired to only 1 Na atom and events where the Cl atom
is paired to >1 Na atom. If you specify --max_order=3, data will be split
into events with Cl paired to 1, 2, or 3 Na atoms, with a 4th category which
holds all events where Cl is paired to >3 Na atoms. If you don't care about
the orders of interaction, --max_order=0 will calculate the MRT as defined by
Laage and Hynes.

Options:
--------
-f name of trajectory file
-s name of tpr file
--sel selection for selection atoms
--ref selection for reference atoms
--cut1 inner cutoff radius (nm)
--cut2 outer cutoff radius (nm)
--max_order maximum bond order to calculate
-b t0 (frames)
-e tf (frames)
-o output file name (optional)
--out write_output boolean (default=True)

References
----------

Damien Laage and James T Hynes. On the residence time for water in a
solute hydration shell: application to aqueous halide solutions. The Journal
of Physical Chemistry B, 112(26):7697–7701, 2008.

Scott H Northrup and James T Hynes. The stable states picture of chemical
reactions. i. formulation for rate constants and initial condition effects. The
Journal of Chemical Physics, 73(6):2700–2714, 1980.
"""


import sys
import getopt
import copy

import numpy as np
import pickle
from collections import deque

import MDAnalysis
from MDAnalysis.analysis.distances import binary_contact_matrix


class BinaryPair(object):
    """Generates data for residence time and state distribution
       calculations

    Class used to generate and analyze binary pair dynamics, using the
    metrics of residence time (estimated via the stable state method)
    and state distribution (reported in fractional form), for selection
    with respect to reference. For example: the residence time of Na
    (selection) with respect to a functional group (reference).

    Parameters:
    -----------
      universe : universe object
        universe object containing trajectory data
      selection : str
        string containing selection for selection
      reference : str
        string containing selection for reference
      cut1 : float
        first cutoff (Angstroms)
      cut2 : float
        second cutoff (Angstroms)
      max_order : int
        maximum order of interactions to track (note: there is always
        an extra catch all group)
      t0 : int
        first frame to analyze
      tf : int
        last frame to analyze
      out_name : str
        name of output pickle file
      write_output : bool
        Specifies whether to write pickle file

    Returns:
    --------
    (out_file).p : pickle file
        pickle file in format [pair, end_pair, dominated,
        aborted, handoff_counter, self.hop_counter]
    """

    def _gen_arrays(self):
        """Initializes arrays containing pairing intervals

        Returns:
        --------
          pair
            np.array containing all pairing events
            np.array(order) X dict(atom id : deque(pairing intervals (tuple)))
          end_pair
            np.array containing all unterminated pairing events
            np.array(order) X dict(atom id : deque(pairing intervals (tuple)))
          dominated
            np.array containing all frames dominated by higher order
            behavior
            np.array(order) X dict(atom id : deque(pairing intervals (tuple)))
        """
        # Note: +1 is for the higher-order catch-all
        pair = np.array([dict() for i in range(self.max_order + 1)])
        end_pair = np.array([dict() for i in range(self.max_order + 1)])
        dominated = np.array([dict() for i in range(self.max_order + 1)])
        aborted = np.array([dict() for i in range(self.max_order + 1)])

        return pair, end_pair, dominated, aborted

    def __init__(self, universe, selection, reference, cut1, cut2, max_order,
                 t0, tf, out_name='state_dynamics', write_output=True):
        self.universe = universe
        self.selection = selection
        self.reference = reference
        self.cut1 = float(cut1)
        self.cut2 = float(cut2)
        self.max_order = int(max_order)
        self.t0 = int(t0)
        self.tf = int(tf)
        self.out_name = out_name
        self.sel_frames = 0
        self.write_output = bool(write_output)

        """
        Variable definitions:
        ------------------------------
            p_dict : dict(sets)
                Contains all atoms participating in pairing events
                during the current frame. Ordered:
                {sel atom id : set(ref atom ids in paired state)}*
            p_dict_order: dict
                Maps sel atom id to the order of the pairing event.
                Ordered:
                {sel atom id : pairing event order (int)}
            p_dict_time: dict(dicts)
                Contains time when each ref atom id joined the current
                pairing event. Ordered:
                dict({ref atom id : time atom joined pairing event})
            handoff_counter : int
                Counts the number of times the len(p_dict_time[i])
                > p_dict_order[i] at the end of a pairing event,
                indicating that the sel atom paired with more than
                the number of atoms accounted for by p_dict_order
                over the course of the pairing event.**
            hop_counter : int
                Counts instances when consecutive atom id sets are
                disjoint, indicating that although the ion is still
                paired, it hopped (i.e. escaped) to pair with another
                ref atom. Hops are treated as terminating events.

            pair : np.array(dict)
                np.array containing all pairing events
                np.array(order) X dict(atom id :
                                       deque(pairing intervals (tuple)))
            end_pair
                np.array containing all unterminated pairing events
                np.array(order) X dict(atom id :
                                       deque(pairing intervals (tuple)))
            dominated
                np.array containing all frames dominated by higher
                order behavior or disrupted by membrane boundary. These
                intervals should not be used in residence time calcs.
                np.array(order) X dict(atom id :
                                       deque(pairing intervals (tuple)))

        * Note: Atom ids must be used to identify pairing events, since
                atom index in each selection's atom list will change
                from frame to frame.
        ** Note: handoff_counter does not account for possible handoff
                 events occurring through salt bridging mechanisms.
                 However, if one assumes an equal probability of each
                 ref atom retaining the sel atom, and one accounts for
                 the residence time of the sel atom in such states,
                 the relative significance of the handoff transport
                 mechanism can still be estimated.
        """

        self.p_dict = dict()
        self.p_dict_order = dict()
        self.p_dict_time = dict()
        self.handoff_counter = 0
        self.hop_counter = 0
        self.pair, self.end_pair, self.dominated, self.aborted = \
            self._gen_arrays()

    @staticmethod
    def _safe_add(this_dict, key, value, data_structure):
        """Safely adds value to this_dict[key], initializing if
        necessary.

        Handles adding dynamic data structures to dictionary objects.

        Parameters:
        -----------
            this_dict : dict
                dictionary containing data
            key : key
                key in dictionary
            value : value
                value to be added to dictionary
            data_structure : set, deque, or dict
                data structure nested within this_dict
        """
        if key not in this_dict:
            this_dict[key] = data_structure
        if type(data_structure).__name__ == 'set':
            this_dict[key].add(value)
        elif type(data_structure).__name__ == 'deque':
            this_dict[key].append(value)
        elif type(data_structure).__name__ == 'dict':
            this_dict[key] = value
        return

    @staticmethod
    def _nested_safe_add(this_dict, key, value, value_value):
        """Safely adds value_value to this_dict[key][value],
        initializing nested dict if necessary.

        Parameters:
        -----------
            this_dict : dict(dicts)
                Contains dicts of ints (intended for p_dict_time)
            key : key
                key in dictionary
            value : value
                value in this_dict[key]
            value_value : int
                value to be added to this_dict[key][value]
        """
        if key not in this_dict:
            this_dict[key] = dict()
        this_dict[key][value] = value_value

        return

    def _gen_dict(self, sel, ref, last_p_dict, this_frame):
        """Determines pairing events for the current frame and logs the
        start of new pairing events.

        This function calculates the binary contact vector (a function
        which returns which atoms are within the inner radius or within
        the buffer region between the inner and outer cutoffs). Info
        from the binary contact vector is to generate p_dict for the
        current frame, according to the stable-states definition. New
        pairing events are logged in p_dict_time.

        Parameters:
        -----------
            sel : AtomGroup
                atom group corresponding to selected atoms
            ref : AtomGroup
                atom group corresponding to reference atoms
            last_p_dict : dict(sets)
                p_dict from the previous frame
            this_frame : int
                Number of the current frame

        Returns:
        --------
            p_dict : dict(set)
                Contains all atoms participating in pairing events
                during the current frame. Ordered:
                {sel atom id : set(ref atom ids in paired state)}
        """
        sel_ids = tuple(sel.ids)
        ref_ids = tuple(ref.ids)

        p_dict = dict()
        # get distances
        dist1, dist2 = binary_contact_matrix(
            sel.positions, ref.positions, self.cut1, self.cut2,
            box=self.universe.trajectory.ts.dimensions)

        # add each ref within the first cutoff, including time for new
        # additions
        for i in range(len(dist1[0])):
            key = sel_ids[dist1[0][i]]
            value = ref_ids[dist1[1][i]]

            self._safe_add(p_dict, key, value, set())

            # if key:value already exists in p_dict_time, do nothing
            if key in self.p_dict_time:
                if value in self.p_dict_time[key]:
                    continue
            # Otherwise it's new, so add it.
            self._nested_safe_add(self.p_dict_time, key, value,
                                  this_frame)

        # if previous dictionary has key:value pair, add buffer zone
        # atoms to p_dict
        for i in range(len(dist2[0])):
            key = sel_ids[dist2[0][i]]
            value = ref_ids[dist2[1][i]]
            # if key:value in last_p_dict, add the pair to p_dict
            if key in last_p_dict:
                if value in last_p_dict[key]:
                    self._safe_add(p_dict, key, value, set())

        return p_dict

    def _log_termination(self, store_array, this_order, time_dict, key,
                         this_frame):
        """logs termination events, recording the tuple: (list of ref
        atoms involved in pairing event, pairing time interval)

        Method catches higher order events not of interest to analysis
        and groups them into the catch-all dict. It then gets the ref
        atoms from the time_dict (either p_dict_time or
        last_p_dict_time) and stores the tuple (ref atoms in pairing
        event, pairing event interval) in the dictionary corresponding
        to the pairing event order.

        Parameters:
        -----------
            store_array : np.array(dict(deques))
                array storing pairing event data
            this_order : int
                order of the pairing event
            time_dict : dict(dicts)
                Contains reference atoms and the time they joined the
                current pairing event
            key : key
                Atom id of sel atom involved in pairing event
            this_frame : int
                Current frame, used as the ending frame in the pairing
                interval
        """
        # handling catch-all
        if this_order > len(store_array):
            this_order = len(store_array)

        # get ref atom ids involved in pairing event
        values = time_dict[key].keys()
        
        self._safe_add(
            store_array[this_order - 1], key,
            (values, (time_dict[key][values[0]], this_frame)), deque())
        
        return

    def _handle_sels(self, sel_ids, ref_ids, this_frame):
        """If changes in selected atoms will cause pairing event
        termination, log termination in aborted.

        Parameters:
        -----------
            sel_ids: tuple(ints)
                selections from selection str for current frame
            ref_ids: tuple(ints)
                selections from reference str for current frame
            this_frame : int
                Current frame, used as the ending frame in the pairing
                interval
        """
        # if sel atom in pairing event moves outside of selection
        for sel in self.p_dict.keys():
            if sel not in sel_ids:
                self._log_termination(self.aborted, self.p_dict_order.pop(sel),
                                      self.p_dict_time, sel, this_frame)
                self.p_dict_time.pop(sel)
                self.p_dict.pop(sel)
                continue
            # if ref atom in pairing event moves outside of selection
            # causing pairing event to end
            if len(self.p_dict[sel]) == 1:
                # if ref atom not in current reference group
                if list(self.p_dict[sel])[0] not in ref_ids:
                    self._log_termination(
                        self.aborted, self.p_dict_order.pop(sel),
                        self.p_dict_time, sel, this_frame)
                    self.p_dict_time.pop(sel)
                    self.p_dict.pop(sel)
        return

    def _categorize_terminations(self, last_p_dict, this_frame):
        """Logs termination events in (pair) according to pairing event
         order, checks for handoffs, and removes terminated event data
         from p_dict_order and p_dict_time.

         Parameters:
         -----------
            last_p_dict : dict(sets)
                p_dict from the previous frame
            this_frame : int
                Current frame number
        """
        # If last_p_dict has a key that is not in p_dict, record the end
        # of this binding event
        for key in last_p_dict.keys():
            if key not in self.p_dict:
                # pop order from p_dict_order
                this_order = self.p_dict_order.pop(key)

                # check for handoffs and counter events
                if len(self.p_dict_time[key].keys()) > this_order:
                    self.handoff_counter += 1

                self._log_termination(
                    self.pair, this_order, self.p_dict_time, key, this_frame)

                self.p_dict_time.pop(key)

        return

    def _update_order(self, last_p_dict, last_p_dict_time, this_frame):
        """Handles hopping events, assign order to new pairing events,
        and handles domination of lower order pairing events.

        If p_dict and last_p_dict entries for the same key are
        disjoint, the key atom has hopped from one ref to another.
        Therefore, count hopping event, log termination in (pair),
        remove old data from p_dict_time, and update p_dict order.
        Move to the next atom entry if a hop occurs.

        Log new pairing event orders.

        If a pairing event is dominated by higher order behavior, log
        the event in dominated, update p_dict_time for each atom to
        the current frame (i.e. the start of the new pairing event),
        then update the order of the pairing event.

        Parameters:
        -----------
            last_p_dict : dict(sets)
                p_dict from the previous frame
            last_p_dict_time : dict(dicts)
                p_dict_time from previous frame
            this_frame : int
                Current frame number
        """
        for key in self.p_dict.keys():

            # identify and handle hopping events
            if key in last_p_dict:
                if self.p_dict[key].isdisjoint(last_p_dict[key]):

                    is_juggle = False

                    # checks for hops between bridged atoms
                    for value in self.p_dict[key]:
                        if value in last_p_dict_time[key].keys():
                            is_juggle = True

                    if is_juggle:
                        continue

                    self.hop_counter += 1  # a hop has occurred

                    self._log_termination(
                        self.pair, self.p_dict_order.pop(key),
                        last_p_dict_time, key, this_frame)

                    # Remove old keys from p_dict_time
                    # (necessary because this transition is not flagged
                    #  by _categorize_terminations)
                    for value in last_p_dict_time[key].keys():
                        self.p_dict_time[key].pop(value)
                    
                    self.p_dict_order[key] = len(self.p_dict[key])

                    # avoid looking for domination if a hopping event
                    # occurs
                    continue

            # if order is greater than p_dict_order, store dominated
            # interval and update p_dict_order and p_dict_time
            len_key = len(self.p_dict[key])
            # add new keys to p_dict order
            if key not in self.p_dict_order:
                self.p_dict_order[key] = len_key

            if len_key > self.p_dict_order[key]:
                # Note: p_dict_order[key] currently refers to order at
                # previous frame
                self._log_termination(self.dominated, self.p_dict_order[key],
                                      last_p_dict_time, key, this_frame)

                # updating entries in p_dict_time to this_frame
                for value in self.p_dict_time[key].keys():
                    self.p_dict_time[key][value] = this_frame

                # update order
                self.p_dict_order[key] = len_key
        return

    def run(self):
        # check that trajectory length is correct
        assert len(self.universe.trajectory) >= (self.tf-self.t0+1), \
            ("Sample interval exceeds trajectory length. This may result"
             "from choice of t0/tf or insufficient RAM when loading the"
             "trajectory.")

        for ts in self.universe.trajectory[self.t0:self.tf + 1]:
            this_frame = ts.frame

            sel = self.universe.select_atoms(self.selection)
            ref = self.universe.select_atoms(self.reference)

            self.sel_frames += len(sel)

            # handle pairing events which move out of selections
            if this_frame > self.t0:
                self._handle_sels(tuple(sel.ids), tuple(ref.ids), this_frame)

            # storing data from previous frame
            last_p_dict = self.p_dict
            # recursively copy data
            last_p_dict_time = copy.deepcopy(self.p_dict_time)

            # generate dict of paired atoms at this frame
            self.p_dict = self._gen_dict(sel, ref, last_p_dict,
                                         this_frame)

            # categorize terminated events
            if this_frame > self.t0:
                self._categorize_terminations(last_p_dict, this_frame)

            # update order and identify hops
            self._update_order(last_p_dict, last_p_dict_time, this_frame)

            # record unfinished pairing events at the end of the
            # simulation
            if this_frame == self.tf:
                for key in self.p_dict.keys():
                    self._log_termination(
                        self.end_pair, self.p_dict_order[key],
                        self.p_dict_time, key, this_frame)

            # log progress for user
            if this_frame % 500 == 0:
                print('Frame: ' + str(this_frame))
                if self.write_output:
                    pickle.dump([self.pair, self.end_pair, self.dominated,
                                 self.aborted, self.handoff_counter,
                                 self.hop_counter, self.sel_frames],
                                open(self.out_name, 'wb'))

        if self.write_output:
            pickle.dump([self.pair, self.end_pair, self.dominated,
                         self.aborted, self.handoff_counter, self.hop_counter,
                         self.sel_frames],
                        open(self.out_name, 'wb'))
        else:
            return [self.pair, self.end_pair, self.dominated,
                    self.aborted, self.handoff_counter, self.hop_counter,
                    self.sel_frames]

        print("Complete!")

"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def restimeFunc(t,tau):
    return np.exp(t/tau*(-1))

def fit(xdata,ydata,interval):
    '''
    Returns fitted ydata, MRT, err_MRT
    '''
    popt, pcov = curve_fit(restimeFunc,xdata[:interval],ydata[:interval])
    fit_data = restimeFunc(xdata[:interval],*popt)
    return fit_data, popt[0], np.sqrt(pcov)[0][0]

def calcResTime(order,data,numFrames,interval,ts):
    '''Plots MRT data and returns MRT and uncertainty in MRT

    Parameters
    ----------
    order : int
        order of interaction - 1 (since storage index is zero)
        (if you want first order pairing, order = 0, etc.)
    data : [pair, end_pair, dominated, aborted, handoff_counter, hop_counter,
            sel_frames]
        data loaded from the pickle file
    numFrames : int
        number of frames analyzed (for example: -b 0 -e 1000 --> 1001 frames)
    interval : int
        fits data from 0 to (interval) to calculate the MRT
    ts : int
        time step used in simulation
    '''

    pCounter = np.zeros(numFrames,dtype="int")
    end_pCounter = np.zeros(numFrames,dtype="int")

    # counting pairing events
    for key in data[0][order].keys():
        for i in range(len(data[0][order][key])):
            b = data[0][order][key][i][1][0]
            e = data[0][order][key][i][1][1]
            pCounter[e-b] += 1

    # counting unterminated pairing events
    for key in data[1][order].keys():
        for i in range(len(data[1][order][key])):
            b = data[1][order][key][i][1][0]
            e = data[1][order][key][i][1][1]
            end_pCounter[e-b] += 1

    tot_pairs = float(pCounter.sum() + end_pCounter.sum())

    # calculating 1-(probability that ion is unbound)
    p_paired = np.zeros(numFrames)

    for i in range(numFrames):
        p_paired[i] = 1 - float(pCounter[:i+1].sum())/tot_pairs

    xdata = [i*ts for i in range(numFrames)]

    ydata, MRT, err_MRT = fit(xdata,p_paired,interval)

    plt.figure()
    plt.scatter(xdata[:interval],pCounter[:interval])
    plt.scatter(xdata[:interval],end_pCounter[:interval])
    plt.xlabel('Pairing Interval Length (ps)',size=20)
    plt.ylabel('Frequency',size=20)

    plt.figure()
    plt.plot(xdata[:interval],p_paired[:interval],label='Data')
    plt.plot(xdata[:interval],ydata[:interval],label='Fit')
    plt.xlabel('Pairing Interval Length (ps)',size=20)
    plt.ylabel(r'P$_p$(t)',size=20)
    plt.axes().set_yscale('log')
    plt.legend(loc='best')

    plt.show()

    return MRT, err_MRT

if __name__ == "__main__":

    p_or_b = 'bind'
    species = "Na"
    orders = [0,1,2,3]

    numFrames = 1001
    fit_interval = 50
    ts = 2 # ps

    # list with dimensions: order X [data, err_data]
    results = [[0,0] for i in range(len(orders))]

    data = pickle.load(open('your_file.p','rb'))

    for order in orders:
        plt.figure()

        results[order][0], results[order] = \
                calcResTime(order,data,numFrames,fit_interval,ts)

    print(results)
"""

if __name__ == "__main__":
    """
    Options:
    --------
    -f name of trajectory file
    -s name of tpr file
    --sel selection for selection atoms
    --ref selection for reference atoms
    --cut1 inner cutoff radius (nm)
    --cut2 outer cutoff radius (nm)
    --max_order maximum bond order to calculate
    -b t0 (frames)
    -e tf (frames)
    -o output file name (optional)
    --out write_output boolean (default=True)
    """

    # handling input arguments
    this_opts, args = getopt.getopt(
        sys.argv[1:], "f:s:b:e:o:",
        ['cut1=', 'cut2=', 'max_order=', 'ref=', 'sel=', 'out='])

    opts = dict()
    for opt, arg in this_opts:
        opts[opt] = arg

    # handling optional parameters
    if "-o" not in opts:
        opts["-o"] = "state_dynamics.p"
    if "--out" in opts:
        if 'False' in opts['--out'] or 'false' in opts['--out']:
            opts['--out'] = False
    if "--out" not in opts:
        opts["--out"] = True

    print('Loading Universe...')
    u = MDAnalysis.Universe(opts['-s'], opts['-f'])
    print('Complete!')

    this_BinaryPair = BinaryPair(
        u, opts['--sel'], opts['--ref'], opts['--cut1'], opts['--cut2'],
        opts['--max_order'], opts['-b'], opts['-e'], opts['-o'], opts['--out'])
    this_BinaryPair.run()
