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
from __future__ import print_function
import numpy as np
import MDAnalysis
import MDAnalysis.analysis.pca as pca
from numpy.testing import (assert_almost_equal, assert_equal,
                           assert_array_almost_equal,raises)


from MDAnalysisTests.datafiles import PDB, XTC


class TestPCA(object):
    def setUp(self):
        self.u = MDAnalysis.Universe(PDB, XTC)
        self.pca = pca.PCA(self.u.atoms, select='backbone and name CA',
                demean=True)
        self.pca.run()
        self.n_atoms = self.u.select_atoms('backbone and name CA').n_atoms

    def test_cum_var(self):
        assert_almost_equal(self.pca.cumulated_variance[-1], 1)
        # makes no sense to test values here, no physical meaning

    def test_pcs(self):
        assert_equal(self.pca.p_components.shape, (self.n_atoms*3, self.n_atoms*3))

    def test_different_steps(self):
        self.pca.run()
        cum_var = self.pca.cumulated_variance
        pcs = self.pca.p_components

    def test_transform(self):
        self.ag = self.u.select_atoms('backbone and name CA')
        pca_space = self.pca.transform(self.ag, n_components=1)
        assert_equal(pca_space.shape,
                     (self.u.trajectory.n_frames, 1))
