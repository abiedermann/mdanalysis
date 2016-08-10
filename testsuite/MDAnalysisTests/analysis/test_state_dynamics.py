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
from MDAnalysis.analysis import BinaryPair
import pickle

from numpy.testing import assert_allclose
import numpy as np

from MDAnalysisTests.datafiles import BPSH_XTC, BPSH_TPR, SD_DATA

class TestBinaryPair(object):
    def __init__(self):
        self.u = MDAnalysis.Universe(BPSH_TPR,BPSH_XTC)
        self.this_BP = BinaryPair(self.u,['name CL'],0,25,5,write_output=False)
    def test_binary_pair(self):
        answer = pickle.load(open(SD_DATA,'rb'))
        assert_allclose(self.this_BP.run(),answer,10**-4,10**-4)
