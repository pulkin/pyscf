#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fci

b = 1.4
mol = gto.Mole()
mol.build(
verbose = 0,
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
symmetry = 1
)
mfr = scf.RHF(mol)
mfr.scf()
mcr = mcscf.CASSCF(mfr, 4, 4)
mcr.conv_tol_grad = 1e-6
mcr.mc1step()[0]

mfu = scf.UHF(mol)
mfu.scf()
mcu = mcscf.UCASSCF(mfu, 4, 4)
mcu.conv_tol_grad = 1e-6
mcu.mc1step()[0]


class KnowValues(unittest.TestCase):
    def test_spin_square(self):
        ss = mcscf.addons.spin_square(mcr)[0]
        self.assertAlmostEqual(ss, 0, 7)

    def test_ucasscf_spin_square(self):
        ss = mcscf.addons.spin_square(mcu)[0]
        self.assertAlmostEqual(ss, 0, 7)

    def test_rcas_natorb(self):
        mo1, ci1, mocc1 = mcscf.addons.cas_natorb(mcr)
        self.assertAlmostEqual(numpy.linalg.norm(mo1)  , 9.9260608594977491, 6)
        self.assertAlmostEqual(numpy.linalg.norm(mocc1), 5.1687145190800079, 6)

#TODO:    def test_ucas_natorb(self):
#TODO:        mo2, ci2, mocc2 = mcscf.addons.cas_natorb(mcu)
#TODO:        self.assertAlmostEqual(numpy.linalg.norm(mo2)  , 11.4470460817871*numpy.sqrt(2), 7)
#TODO:        self.assertAlmostEqual(numpy.linalg.norm(mocc2), 2.59144951056707/numpy.sqrt(2), 7)

    def test_get_fock(self):
        f1 = mcscf.addons.get_fock(mcr)
        self.assertTrue(numpy.allclose(f1, f1.T))
        self.assertAlmostEqual(numpy.linalg.norm(f1), 25.482177562349467, 7)
#TODO:        f1 = mcscf.addons.get_fock(mcu)
#TODO:        self.assertTrue(numpy.allclose(f1[0], f1[0].T))
#TODO:        self.assertTrue(numpy.allclose(f1[1], f1[1].T))
#TODO:        self.assertAlmostEqual(numpy.linalg.norm(f1), 23.597476504476919*numpy.sqrt(2), 6)

    def test_canonicalize1(self):
        numpy.random.seed(1)
        f1 = numpy.random.random(mcr.mo_coeff.shape)
        u1 = numpy.linalg.svd(f1)[0]
        mo, ci, mo_e = mcr.canonicalize(numpy.dot(mcr.mo_coeff, u1))
        e1 = numpy.einsum('ji,jk,ki', mo, f1, mo)
        self.assertAlmostEqual(e1, 44.2658681077, 7)
        mo, ci, mo_e = mcr.canonicalize(numpy.dot(mcr.mo_coeff, u1), eris=mcr.ao2mo(mcr.mo_coeff))
        e1 = numpy.einsum('ji,jk,ki', mo, f1, mo)
        self.assertAlmostEqual(e1, 44.2658681077, 7)

    def test_canonicalize(self):
        mo, ci, mo_e = mcr.canonicalize()
        self.assertAlmostEqual(numpy.linalg.norm(mo), 9.9260608594977242, 7)
        mo, ci, mo_e = mcr.canonicalize(eris=mcr.ao2mo(mcr.mo_coeff))
        self.assertAlmostEqual(numpy.linalg.norm(mo), 9.9260608594977242, 7)

    def test_make_rdm12(self):
        dmr = mcscf.addons.make_rdm1(mcr)
        dm1, dm2 = mcscf.addons.make_rdm12(mcr)
        self.assertTrue(numpy.allclose(dmr, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 3.8205551262007567, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 14.987267883423314, 5)

    def test_make_rdm1s(self):
        dm1 = mcscf.addons.make_rdm1s(mcr)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.7015404376335805, 5)
        dm1 = mcscf.addons.make_rdm1s(mcu)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.7015404376335805, 5)

    def test_sort_mo(self):
        mo1 = numpy.arange(mfr.mo_energy.size).reshape(1,-1)
        ref = [[0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
        mo2 = mcscf.addons.sort_mo(mcr, mo1, [5,6,7,9])
        self.assertTrue(numpy.allclose(mo2, ref))
        mo2 = mcscf.addons.sort_mo(mcu, (mo1,mo1), [5,6,7,9])
        self.assertTrue(numpy.allclose(mo2, (ref,ref)))
        mo2 = mcscf.addons.sort_mo(mcu, (mo1,mo1), [[5,6,7,9],[5,6,8,9]])
        ref1 = [[0, 1, 2, 3, 6, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
        self.assertTrue(numpy.allclose(mo2, (ref,ref1)))

    def test_sort_mo_by_irrep(self):
        mc1 = mcscf.CASSCF(mfr, 8, 4)
        mo0 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, {'E1ux':2, 'E1uy':2, 'E1gx':2, 'E1gy':2})
        mo1 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, {2:2, 3:2, 6:2, 7:2}, {2:0, 3:0, 6:0, 7:0})
        mo2 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, (0,0,2,2,0,0,2,2))
        mo3 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, {'E1ux':2, 'E1uy':2, 2:2, 3:2})
        self.assertTrue(numpy.allclose(mo0, mo1))
        self.assertTrue(numpy.allclose(mo0, mo2))
        self.assertTrue(numpy.allclose(mo0, mo3))

    def test_sort_mo_by_irrep1(self):
        mol = gto.M(atom='N 0 0 -.45; N 0 0 .45', basis='ccpvdz',
                    symmetry=True, verbose=0)
        mf = scf.RHF(mol).run()
        mc1 = mcscf.CASSCF(mf, 6, 6)
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'A1g': 1, 'A1u': 1, 'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1},
                {'A1g': 2, 'A1u': 2})
        self.assertEqual(list(caslst), [4,5,7,8,9,10])
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1},
                {'A1g': 2, 'A1u': 2})
        self.assertEqual(list(caslst), [4,5,7,8,9,10])
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1},
                {'A1u': 2})
        self.assertEqual(list(caslst), [4,5,7,8,9,10])
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'A1g': 1, 'A1u': 1}, {'E1uy': 1, 'E1ux': 1})
        self.assertEqual(list(caslst), [3,6,8,9,12,13])

        self.assertRaises(ValueError, mcscf.addons.caslst_by_irrep, mc1, mf.mo_coeff,
                          {'A1g': 1, 'A1u': 1}, {'E1uy': 3, 'E1ux': 3})
        self.assertRaises(ValueError, mcscf.addons.caslst_by_irrep, mc1, mf.mo_coeff,
                          {'A1g': 3, 'A1u': 4}, {'E1uy': 1, 'E1ux': 1})
        self.assertRaises(ValueError, mcscf.addons.caslst_by_irrep, mc1, mf.mo_coeff,
                          {'E2ux': 2, 'E2uy': 2}, {'E1uy': 1, 'E1ux': 1})

    def test_state_average(self):
        mc = mcscf.CASSCF(mfr, 4, 4)
        mc.fcisolver = fci.solver(mol, singlet=False)
        mc.state_average_((.64,.36))
        e = mc.kernel()[0]
        self.assertAlmostEqual(e, -108.83342083775061, 7)

    def test_state_specific(self):
        mc = mcscf.CASSCF(mfr, 4, 4)
        mc.fcisolver = fci.solver(mol, singlet=False)
        mc.state_specific_(state=1)
        e = mc.kernel()[0]
        self.assertAlmostEqual(e, -108.70065770892457, 7)

    def test_project_init_guess(self):
        b = 1.5
        mol1 = gto.M(
        verbose = 0,
        atom = [
            ['O',(  0.000000,  0.000000, -b/2)],
            ['O',(  0.000000,  0.000000,  b/2)], ],
        basis = 'ccpvtz',)
        mf1 = scf.RHF(mol1).run()
        mc1 = mcscf.CASSCF(mf1, 4, 4)
        mo1 = mcscf.project_init_guess(mc1, mfr.mo_coeff, prev_mol=mol)
        s1 = reduce(numpy.dot, (mo1.T, mf1.get_ovlp(), mo1))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
        self.assertAlmostEqual(numpy.linalg.norm(s1), 7.7459666924148349, 9)

        self.assertRaises(AssertionError, mcscf.project_init_guess, mc1, mfr.mo_coeff)


if __name__ == "__main__":
    print("Full Tests for mcscf.addons")
    unittest.main()

