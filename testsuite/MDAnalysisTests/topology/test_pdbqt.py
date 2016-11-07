# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- http://www.MDAnalysis.org
# Copyright (c) 2006-2015 Naveen Michaud-Agrawal, Elizabeth J. Denning, Oliver Beckstein
# and contributors (see AUTHORS for the full list)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
from numpy.testing import (
    assert_,
)

import MDAnalysis as mda

from MDAnalysisTests.topology.base import ParserBase
from MDAnalysisTests.datafiles import (
    PDBQT_input,  # pdbqt_inputpdbqt.pdbqt
)


class TestPDBQT(ParserBase):
    parser = mda.topology.PDBQTParser.PDBQTParser
    filename = PDBQT_input
    expected_attrs = ['ids', 'names', 'charges', 'types', 'altLocs',
                      'resids', 'resnames',
                      'segids']
    guessed_attrs = ['masses']
    expected_n_atoms = 1805
    expected_n_residues = 199  # resids go 2-102 then 2-99
    expected_n_segments = 2  # res2-102 are A, 2-99 are B
