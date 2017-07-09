#!/usr/bin/env python

import unittest
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf, dft

mol = gto.Mole()
mol.verbose = 0
mol.output = '/dev/null'
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol.basis = {"H": '6-31g',
             "O": '6-31g',}
mol.build()


class KnowValues(unittest.TestCase):
    def test_project_mo_nr2nr(self):
        nao = mol.nao_nr()
        c = numpy.random.random((nao,nao))
        c1 = scf.addons.project_mo_nr2nr(mol, c, mol)
        self.assertTrue(numpy.allclose(c, c1))

        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_nr2nr(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 83.342096002254607, 11)

        mol2.cart = True
        mo2 = scf.addons.project_mo_nr2nr(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 83.436359425591888, 11)

    def test_project_mo_r2r(self):
        nao = mol.nao_2c()
        c = numpy.random.random((nao*2,nao*2))
        c = c + numpy.sin(c)*1j
        c1 = scf.addons.project_mo_r2r(mol, c, mol)
        self.assertTrue(numpy.allclose(c, c1))

        numpy.random.seed(15)
        n2c = mol.nao_2c()
        n4c = n2c * 2
        mo1 = numpy.random.random((n4c,n4c)) + numpy.random.random((n4c,n4c))*1j
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_r2r(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 2159.3715489514038, 11)

    def test_project_mo_nr2r(self):
        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_nr2r(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 172.66468850263556, 11)

    def test_frac_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf = scf.addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -107.13465364012296, 9)

    def test_dynamic_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            6      0.   0  -0.7
            6      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf = scf.addons.dynamic_occ(mf)
        self.assertAlmostEqual(mf.scf(), -74.214503776693817, 9)

    def test_follow_state(self):
        mf = scf.RHF(mol)
        mf.scf()
        mo0 = mf.mo_coeff[:,[0,1,2,3,5]]
        mf = scf.addons.follow_state(mf, mo0)
        self.assertAlmostEqual(mf.scf(), -75.178145727548511, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[:6], [2,2,2,2,0,2]))

    def test_symm_allow_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf = scf.addons.symm_allow_occ(mf)
        self.assertAlmostEqual(mf.scf(), -106.49900188208861, 9)

    def test_float_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            C      0.   0   0'''
        mol.basis = 'cc-pvdz'
        mol.build()
        mf = scf.UHF(mol)
        mf = scf.addons.float_occ(mf)
        self.assertAlmostEqual(mf.scf(), -37.590712883365917, 9)

    def test_mom_occ(self):
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf.scf()
        mo0 = mf.mo_coeff
        occ = mf.mo_occ
        occ[0][4] = 0.
        occ[0][5] = 1.
        mf = scf.addons.mom_occ(mf, mo0, occ)
        dm = mf.make_rdm1(mo0, occ)
        self.assertAlmostEqual(mf.scf(dm), -76.0606858747, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[0][:6], [1,1,1,1,0,1]))

        mf = dft.ROKS(mol)
        mf.xc = 'b3lyp'
        mf.scf()
        mo0 = mf.mo_coeff
        occ = mf.mo_occ
        setocc = numpy.zeros((2, occ.size))
        setocc[:, occ==2] = 1
        setocc[0][4] = 0
        setocc[0][5] = 1
        newocc = setocc[0][:] + setocc[1][:]
        mf = scf.addons.mom_occ(mf, mo0, setocc)
        dm = mf.make_rdm1(mo0, newocc)
        self.assertAlmostEqual(mf.scf(dm), -76.0692546639, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[:6], [2,2,2,2,1,1]))

if __name__ == "__main__":
    print("Full Tests for addons")
    unittest.main()

