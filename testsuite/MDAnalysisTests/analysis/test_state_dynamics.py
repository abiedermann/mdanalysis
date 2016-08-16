# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- http://www.MDAnalysis.org
# Copyright (c) 2006-2015 Naveen Michaud-Agrawal, Elizabeth J. Denning, Oliver
# Beckstein and contributors (see AUTHORS for the full list)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
from __future__ import absolute_import, print_function

import MDAnalysis
from MDAnalysis.analysis.state_dynamics import BinaryPair
import pickle

from numpy.testing import assert_almost_equal, assert_allclose
import numpy as np

from MDAnalysisTests.datafiles import BPSH_XTC, BPSH_TPR, SD_DATA

class TestBinaryPair(object):
    def setUp(self):
        self.u = MDAnalysis.Universe(BPSH_TPR,BPSH_XTC)
        self.this_BP = BinaryPair(self.u,'name CL','name NA',0.3,0.4,2,0,
                25,write_output=False)

    def tearDown(self):
        del self.u
        del self.this_BP

    def test_binary_pair(self):
        answer = pickle.load(open(SD_DATA,'rb'))
        bp_out = self.this_BP.run()
        #assert_allclose(answer, bp_out, 10**-4, 10**-4)
        for i in range(0,4):
            for j in range(len(answer[i])):
                for key in answer[i][j].keys():
                    for k in range(len(answer[i][j][key])):
                        assert_almost_equal(answer[i][j][key][k][0],
                                            bp_out[i][j][key][k][0],
                                            decimal=4)
                        assert_almost_equal(answer[i][j][key][k][1],
                                            bp_out[i][j][key][k][1],
                                            decimal=4)
        for i in range(4,6):
            assert_almost_equal(answer[i],bp_out[i],decimal=4)
