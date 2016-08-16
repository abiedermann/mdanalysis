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
from MDAnalysis.analysis.MSD import MSD

from numpy.testing import assert_almost_equal, assert_allclose
import numpy as np

from MDAnalysisTests.datafiles import BPSH_XTC, BPSH_TPR

class TestMSD(object):
    def setUp(self):
        self.u = MDAnalysis.Universe(BPSH_TPR,BPSH_XTC)
        self.this_MSD = MSD(self.u,['name CL'],0,25,5,write_output=False)

    def tearDown(self):
        del self.u
        del self.this_MSD

    def test_MSD(self):
        answer = [np.array([[  0.        ,   3.47127771,   6.42875481,   9.29823494,
                             11.99853706,  15.02497864,  19.54847527,  22.4382    ,
                             25.13007355,  27.46950531,  30.27593613,  36.07775116,
                             38.90217209,  42.03682327,  45.21638489,  48.07343674,
                             55.66702652,  57.79199982,  60.18118286,  62.08581161,
                             63.77077866,  71.2060318 ,  73.94713593,  77.23184204,
                             78.38237762,  79.60108185]], dtype='float32'),
                 np.array([[6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3,
                     2, 2, 2, 2, 2, 1, 1, 1, 1, 1]])]
        msd_out = self.this_MSD.run()
	    assert_allclose(answer,msd_out,10**-4,10**-4)
