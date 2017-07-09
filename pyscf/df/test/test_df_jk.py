#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.df import df_jk

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)
symol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
    symmetry = 1,
)


class KnowValues(unittest.TestCase):
    def test_rhf(self):
        mf = scf.density_fit(scf.RHF(mol))
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 9)

    def test_uhf(self):
        mf = scf.density_fit(scf.UHF(mol))
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 9)

    def test_uhf_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        mf = scf.density_fit(scf.UHF(pmol))
        self.assertAlmostEqual(mf.scf(), -76.026760700636046, 9)

    def test_rohf(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.density_fit(scf.ROHF(pmol))
        self.assertAlmostEqual(mf.scf(), -75.626515724371814, 9)

    def test_dhf(self):
        pmol = mol.copy()
        pmol.build(False, False)
        mf = scf.density_fit(scf.DHF(pmol))
        mf.conv_tol_grad = 1e-5
        self.assertAlmostEqual(mf.scf(), -76.080738677021458, 8)

    def test_rhf_symm(self):
        mf = scf.density_fit(scf.RHF(symol))
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 9)

    def test_uhf_symm(self):
        mf = scf.density_fit(scf.UHF(symol))
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 9)

    def test_rohf_symm(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.density_fit(scf.ROHF(pmol))
        self.assertAlmostEqual(mf.scf(), -75.626515724371814, 9)

    def test_rhf_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,nao,nao))
        mf = scf.density_fit(scf.RHF(mol))
        vhf1 = mf.get_veff(mol, dm, hermi=0)
        naux = mf._cderi.shape[0]
        cderi = numpy.empty((naux,nao,nao))
        for i in range(naux):
            cderi[i] = lib.unpack_tril(mf._cderi[i])
        vj0 = []
        vk0 = []
        for dmi in dm:
            v1 = numpy.einsum('kij,ij->k', cderi, dmi)
            vj0.append(numpy.einsum('kij,k->ij', cderi, v1))
            v1 = numpy.einsum('pij,jk->pki', cderi, dmi.T)
            vk0.append(numpy.einsum('pki,pkj->ij', cderi, v1))
        vj1, vk1 = df_jk.get_jk(mf.with_df, dm, 0)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(numpy.array(vk0), vk1))
        vhf0 = vj1 - vk1 * .5
        self.assertTrue(numpy.allclose(vhf0, vhf1))

    def test_uhf_veff(self):
        mf = scf.density_fit(scf.UHF(mol))
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,4,nao,nao))
        vhf = mf.get_veff(mol, dm, hermi=0)
        self.assertAlmostEqual(numpy.linalg.norm(vhf), 413.82341595365853, 9)

    def test_assign_cderi(self):
        nao = mol.nao_nr()
        w, u = scipy.linalg.eigh(mol.intor('int2e_sph', aosym='s4'))
        idx = w > 1e-9

        mf = scf.density_fit(scf.UHF(mol))
        mf._cderi = (u[:,idx] * numpy.sqrt(w[idx])).T.copy()
        self.assertAlmostEqual(mf.kernel(), -76.026765673110447, 9)


if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()

