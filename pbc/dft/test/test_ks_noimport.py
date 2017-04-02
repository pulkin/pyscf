#!/usr/bin/env python
#
# Author: Artem Pulkin <gpulkin@gmail.com>
#

import unittest, json
import numpy

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

def assert_bands_close(one,two,interactive = False):
    """
    Compares band structures.
    """
    try:
        numpy.testing.assert_allclose(one, two)
    except AssertionError as e:
        if interactive:
            from matplotlib import pyplot
            for x in numpy.array(one).swapaxes(0,1):
                pyplot.plot(numpy.arange(len(x)),x,marker = "x",color = "b")
            for x in numpy.array(two).swapaxes(0,1):
                pyplot.plot(numpy.arange(len(x)),x,marker = "x",color = "r")
            pyplot.legend()
            print "Maximal deviation:", numpy.abs(one-two).max()
            pyplot.show()
        raise e

class Monolayer_hBN(unittest.TestCase):
    """
    Monolayer hexagonal boron nitride LDA model compared against various
    datasets.
    """
    
    k_points_cartesian_bohr = [
        [0.661017637338074, -0.3816387107717475, 0.0],
        [0.330508818669037, -0.19081935538587375, 0.0],
        [0.0, 0.0, 0.0],
        [0.44067842491408316, 0.0, 0.0],
        [0.8813568498281663, 0.0, 0.0],
        [0.7711872435170185, -0.19081935538587372, 0.0],
        [0.661017637338074, -0.3816387107717475, 0.0]
    ]

    atomic_coordinates_cartesian_angstrom = [
        [1.2574999999999876, 0.7260179636666594, 5.0],
        [2.5150000000000126, 1.4520359273333405, 5.0]
    ]

    unit_cell_angstrom = [
        [2.515, 0.0, 0.0],
        [1.2575, 2.178053891, 0.0],
        [0.0, 0.0, 10.0]
    ]

    def __prepare__(self, kpts = [1,1,1], xc = "lda", **kwargs):
        params = dict(
            unit = 'Angstrom',
            atom = list(zip(['B','N'],self.atomic_coordinates_cartesian_angstrom)),
            a = self.unit_cell_angstrom,
            basis = 'gth-szv',
            pseudo = 'gth-lda',
            gs = [16,16,75],
            verbose = 0,
        )
        params.update(kwargs)
        self.cell = pbcgto.M(**params)
        self.model = pbcdft.KRKS(self.cell, self.cell.make_kpts(kpts))
        self.model.xc = xc
    
    def test_bands_gamma(self):
        self.__prepare__()
        self.model.kernel()
        e,w = self.model.get_bands(self.k_points_cartesian_bohr)
        with open("test_ks_noimport_pyscf1.json",'r') as f:
            data = json.load(f)["bands"]
        assert_bands_close(e,data,interactive = False)

    def test_bands_3(self):
        self.__prepare__(kpts = [3,3,1])
        self.model.kernel()
        e,w = self.model.get_bands(self.k_points_cartesian_bohr)
        with open("test_ks_noimport_pyscf2.json",'r') as f:
            data = json.load(f)["bands"]
        assert_bands_close(e,data,interactive = False)
    
    def __UNDER_CONTRUCTION__test_bands_gamma_cp2k(self):
        """
        Input file:
        .. code-block:: none
            &GLOBAL
              PROJECT _hBN
              RUN_TYPE ENERGY
              PRINT_LEVEL MEDIUM
            &END GLOBAL
            &FORCE_EVAL
              METHOD Quickstep
              &SUBSYS
                &KIND B
                  &BASIS
                    1
                      1 0 1 4 1 1
                        2.8854084023  0.1420731829 -0.0759815770
                        0.8566849689 -0.0083257749 -0.2508281584
                        0.2712991753 -0.6707104603 -0.4610296144
                        0.0826101984 -0.4241277148 -0.4419922734
                  &END BASIS
                  &POTENTIAL
                    2    1
                      0.41899145    2    -5.85946171     0.90375643
                    2
                      0.37132046    1     6.29728018
                      0.36456308    0
                  &END POTENTIAL
                &END KIND
                &KIND N
                  &BASIS
                    1
                      1 0 1 4 1 1
                        6.1526903413  0.1506300537 -0.0950603476
                        1.8236332280 -0.0360100734 -0.2918864295
                        0.5676628870 -0.6942023212 -0.4739050050
                        0.1628222852 -0.3878929987 -0.3893418670
                  &END BASIS
                  &POTENTIAL
                    2    3
                       0.28379051    2   -12.41522559     1.86809592
                    2
                       0.25540500    1    13.63026257
                       0.24549453    0
                  &END POTENTIAL
                &END KIND
                &CELL
                  A 2.515000000 0.000000000  0.000000000
                  B 1.257500000 2.178053891  0.000000000
                  C 0.000000000 0.000000000 10.000000000
                &END CELL
                &COORD
                  B 1.257500000 0.7260179636666594 5.0
                  N 2.515000000 1.4520359273333405 5.0
                &END COORD
              &END SUBSYS
              &DFT
                &KPOINTS
                  FULL_GRID
                  SCHEME MONKHORST-PACK 1 1 1
                  &BAND_STRUCTURE
                    ADDED_MOS 4
                    &KPOINT_SET
                      UNITS CART_BOHR
                      NPOINTS 2
                      SPECIAL_POINT 0.5 0.0 0.0
                      SPECIAL_POINT 0 0 0
                      SPECIAL_POINT 0.6666666667 0.3333333333 0.0
                      SPECIAL_POINT 0.5 0.0 0.0
                    &END KPOINT_SET
                    FILE_NAME bands
                  &END BAND_STRUCTURE
                &END KPOINTS
                &QS
                  EXTRAPOLATION USE_PREV_WF
                &END QS
                &MGRID
                  CUTOFF 150
                  REL_CUTOFF 60
                  NGRIDS 4
                &END MGRID
                &XC
                  &XC_FUNCTIONAL PBE
                  &END XC_FUNCTIONAL
                &END XC
                &SCF
                  EPS_SCF 1.0E-10
                  MAX_SCF 60
                &END SCF
              &END DFT
            &END FORCE_EVAL
        """
        self.__prepare__(
            pseudo = 'gth-pbe',
            gs = [20,20,75],
            xc = 'pbe',
            verbose = 4,
        )
        self.model.kernel()
        e, w = self.model.get_bands(self.k_points_cartesian_bohr)
        with open("test_ks_noimport_cp2k.json",'r') as f:
            data = json.load(f)["bands"]
        # Shift
        avg = numpy.mean(e-data)
        e -= avg
        assert_bands_close(e,data,interactive = True)
        
"""
Future test: `OpenMX <http://www.openmx-square.org/>`_ v 3.8.
Prototype input file:

.. code-block:: none
    System.CurrentDirectory ./
    System.Name _hBN
    data.path /export/scratch/openmx_tests/DFT_DATA13
    level.of.stdout 1
    level.of.fileout 1
    
    Species.Number 2
    <Definition.of.Atomic.Species
     B   B7.0-s1p1    B_CA13
     N   N7.0-s1p1    N_CA13
    Definition.of.Atomic.Species>
    
    Atoms.UnitVectors.Unit Ang
    
    <Atoms.UnitVectors
    2.515 0.0 0.0
    1.2575 2.178053891 0.0
    0.0 0.0 10.0
    Atoms.UnitVectors>
    
    Atoms.Number 2
    
    Atoms.SpeciesAndCoordinates.Unit   Frac
    <Atoms.SpeciesAndCoordinates
       1    B    0.33333333333333    0.33333333333333    0.5     1.5     1.5
       2    N    0.66666666666667    0.66666666666667    0.5     2.5     2.5
    Atoms.SpeciesAndCoordinates>
    
    scf.XcType                  LDA
    scf.SpinPolarization        off
    scf.EigenvalueSolver        band
    scf.Kgrid                   3 3 1
    scf.Mixing.Type             rmm-diis
    
    Band.dispersion on
    Band.Nkpath 3
    
    <Band.kpath
      3  0.5 0.0 0.0  0.0 0.0 0.0  M G
      3  0.0 0.0 0.0  0.6666666667 0.3333333333 0.0  G K
      3   0.6666666666 0.3333333333 0.0  0.5 0.0 0.0  K M
    Band.kpath>
"""

if __name__ == '__main__':
    print("Standalone Tests for pbc.dft.krks")
    unittest.main()
