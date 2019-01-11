from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RHF, KRHF
from pyscf.pbc.tdscf import KTDHF
from pyscf.pbc.tdscf import krhf_slow_supercell as ktd, rhf_slow as td
from pyscf.pbc.tools.pbc import super_cell

from test_common import retrieve_m, adjust_mf_phase, ov_order, assert_vectors_close

import unittest
from numpy import testing
import numpy


class DiamondTestGamma(unittest.TestCase):
    """Compare this (supercell_slow) @Gamma vs reference."""
    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        # cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        cls.model_krhf = model_krhf = KRHF(cell).density_fit()
        model_krhf.kernel()

        cls.td_model_krhf = td_model_krhf = KTDHF(model_krhf)
        td_model_krhf.nroots = 5
        td_model_krhf.kernel()

        cls.ref_m_krhf = retrieve_m(td_model_krhf)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_krhf
        del cls.model_krhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (ktd.PhysERI, ktd.PhysERI4, ktd.PhysERI8):
            e = eri(self.model_krhf)
            m = ktd.build_matrix(e)
            try:
                testing.assert_allclose(self.ref_m_krhf, m, atol=1e-14)
                vals, vecs = ktd.eig(m, nroots=self.td_model_krhf.nroots)
                testing.assert_allclose(vals, self.td_model_krhf.e, atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = ktd.kernel(self.model_krhf, driver='eig', nroots=self.td_model_krhf.nroots)
        testing.assert_allclose(vals, self.td_model_krhf.e, atol=1e-5)
        nocc = nvirt = 4
        testing.assert_equal(vecs.shape, (self.td_model_krhf.nroots, 2, 1, 1, nocc, nvirt))
        try:
            assert_vectors_close(vecs.squeeze(), numpy.array(self.td_model_krhf.xy).squeeze(), atol=1e-12)
        except Exception:
            # TODO: this exception is triggered in case of fft density fitting
            print("This is a known bug: vectors from davidson are wrong")
            raise

    def test_class(self):
        """Tests container behavior."""
        model = ktd.TDRHF(self.model_krhf)
        model.nroots = self.td_model_krhf.nroots
        model.kernel()
        testing.assert_allclose(model.e, self.td_model_krhf.e, atol=1e-5)
        try:
            assert_vectors_close(model.xy.squeeze(), numpy.array(self.td_model_krhf.xy).squeeze(), atol=1e-12)
        except Exception:
            # TODO: this exception is triggered in case of fft density fitting
            print("This is a known bug: vectors from davidson are wrong")
            raise


class DiamondTestShiftedGamma(unittest.TestCase):
    """Compare this (supercell_slow) @non-Gamma 1kp vs rhf_slow."""
    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        # cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        k = cell.get_abs_kpts((.1, .2, .3))

        # The Gamma-point reference
        cls.model_rhf = model_rhf = RHF(cell, k).density_fit()
        model_rhf.conv_tol = 1e-14
        model_rhf.kernel()

        cls.ref_e, cls.ref_v, eri = td.kernel(model_rhf, return_eri=True)
        cls.ref_m = td.build_matrix(eri)

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

        adjust_mf_phase(model_rhf, model_krhf)

        testing.assert_allclose(model_rhf.mo_energy, model_krhf.mo_energy[0])
        testing.assert_allclose(model_rhf.mo_coeff, model_krhf.mo_coeff[0])

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krhf
        del cls.model_rhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (ktd.PhysERI, ktd.PhysERI4):
            e = eri(self.model_krhf)
            m = ktd.build_matrix(e)

            try:
                testing.assert_allclose(self.ref_m, m, atol=1e-10)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = ktd.kernel(self.model_krhf, driver='eig')
        testing.assert_allclose(vals, self.ref_e, atol=1e-12)
        nocc = nvirt = 4
        testing.assert_equal(vecs.shape, (len(vals), 2, 1, 1, nocc, nvirt))
        assert_vectors_close(vecs.squeeze(), self.ref_v, atol=1e-9)


class DiamondTestSupercell2(unittest.TestCase):
    """Compare this (supercell_slow) @2kp vs rhf_slow (2x1x1 supercell)."""
    k = 2
    k_c = (0, 0, 0)

    @classmethod
    def setUpClass(cls):
        cls.cell = cell = Cell()
        # Lift some degeneracies
        cell.atom = '''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.67   1.68   1.69
        '''
        cell.basis = {'C': [[0, (0.8, 1.0)],
                            [1, (1.0, 1.0)]]}
        # cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.build()

        k = cell.make_kpts([cls.k, 1, 1], scaled_center=cls.k_c)

        # The Gamma-point reference
        cls.model_rhf = model_rhf = RHF(super_cell(cell, [cls.k, 1, 1]), kpt=k[0]).density_fit()
        model_rhf.conv_tol = 1e-14
        model_rhf.kernel()

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        model_krhf.conv_tol = 1e-14
        model_krhf.kernel()

        adjust_mf_phase(model_rhf, model_krhf)

        ke = numpy.concatenate(model_krhf.mo_energy)
        ke.sort()

        # Make sure mo energies are the same
        testing.assert_allclose(model_rhf.mo_energy, ke)

        # Make sure no degeneracies are present
        testing.assert_array_less(1e-4, ke[1:] - ke[:-1])

        cls.ov_order = ov_order(model_krhf)

        cls.ref_e, cls.ref_v, eri = td.kernel(model_rhf, return_eri=True)
        cls.ref_m = td.build_matrix(eri)

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.model_krhf
        del cls.model_rhf
        del cls.cell

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries."""
        for eri in (ktd.PhysERI, ktd.PhysERI4, ktd.PhysERI8):
            e = eri(self.model_krhf)
            m = ktd.build_matrix(e)

            try:
                testing.assert_allclose(self.ref_m, m[numpy.ix_(self.ov_order, self.ov_order)], atol=1e-5)
            except Exception:
                print("When testing {} the following exception occurred:".format(eri))
                raise

    def test_eig_kernel(self):
        """Tests default eig kernel behavior."""
        vals, vecs = ktd.kernel(self.model_krhf, driver='eig')
        testing.assert_allclose(vals, self.ref_e, atol=1e-7)
        nocc = nvirt = 4
        testing.assert_equal(vecs.shape, (len(vals), 2, self.k, self.k, nocc, nvirt))
        vecs = vecs.reshape(len(vecs), -1)[:, self.ov_order]
        assert_vectors_close(vecs.reshape(len(vals), -1), self.ref_v.reshape(len(self.ref_v), -1), atol=1e-6)


class DiamondTestSupercell3(DiamondTestSupercell2):
    """Compare this (supercell_slow) @3kp vs rhf_slow (3x1x1 supercell)."""
    k = 3
    k_c = (.1, 0, 0)
