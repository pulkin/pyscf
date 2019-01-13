from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import KRHF
from pyscf.pbc.tdscf import krhf_slow as ktd, krhf_slow_supercell as std, krhf_slow_gamma as gtd
from pyscf.tdscf.rhf_slow import eig

import unittest
from numpy import testing
import numpy

from test_common import assert_vectors_close


def k2k(*indexes):
    result = []
    offset = 0
    for i in indexes:
        result.append(offset + (i + numpy.arange(len(i)) * len(i)))
        offset += len(i) * len(i)
    return numpy.concatenate(result)


class DiamondTest(unittest.TestCase):
    """Compare this (krhf_slow) @2kp@Gamma vs `krhf_slow_gamma` and `krhf_slow_supercell`."""
    k = 2
    k_c = (0, 0, 0)
    test8 = True

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

        # K-points
        cls.model_krhf = model_krhf = KRHF(cell, k).density_fit()
        model_krhf.kernel()

        # Gamma
        cls.td_model_rhf_gamma = gtd.TDRHF(model_krhf)
        cls.td_model_rhf_gamma.kernel()
        cls.ref_m_gamma = cls.td_model_rhf_gamma.eri.tdhf_matrix()

        # Supercell
        cls.td_model_rhf_supercell = std.TDRHF(model_krhf)
        cls.td_model_rhf_supercell.kernel()
        cls.ref_m_supercell = cls.td_model_rhf_supercell.eri.tdhf_matrix()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rhf_supercell
        del cls.td_model_rhf_gamma
        del cls.model_krhf
        del cls.cell

    def test_eri_gamma(self):
        """Tests all ERI implementations: with and without symmetries (gamma-point only)."""
        for eri in (ktd.PhysERI, ktd.PhysERI4, ktd.PhysERI8):
            if eri is not ktd.PhysERI8 or self.test8:
                try:

                    e = eri(self.model_krhf)
                    m = e.tdhf_matrix(0)

                    testing.assert_allclose(self.ref_m_gamma, m, atol=1e-12)
                except Exception:
                    print("When testing {} the following exception occurred:".format(eri))
                    raise

    def test_class_gamma(self):
        """Tests container behavior (gamma-point only)."""
        model = ktd.TDRHF(self.model_krhf)
        model.kernel(k=0)
        testing.assert_allclose(model.e[0], self.td_model_rhf_gamma.e, atol=1e-12)
        nocc = nvirt = 4
        testing.assert_equal(model.xy[0].shape, (len(model.e[0]), 2, self.k, nocc, nvirt))
        assert_vectors_close(model.xy[0], self.td_model_rhf_gamma.xy, atol=1e-9)

    def test_eri(self):
        """Tests all ERI implementations: with and without symmetries (supercell)."""
        o = v = 4
        for eri in (ktd.PhysERI, ktd.PhysERI4, ktd.PhysERI8):
            if eri is not ktd.PhysERI8 or self.test8:
                try:
                    e = eri(self.model_krhf)

                    s = (2 * self.k * self.k, 2 * self.k * self.k, o*v, o*v)
                    m = numpy.zeros(s, dtype=complex)

                    for k in range(self.k):
                        # Prepare indexes
                        r1, r2, c1, c2 = ktd.get_block_k_ix(e, k)

                        r = k2k(r1, r2)
                        c = k2k(c1, c2)

                        # Build matrix
                        _m = e.tdhf_matrix(k)

                        # Assign the submatrix
                        m[numpy.ix_(r, c)] = _m.reshape((2*self.k, o*v, 2*self.k, o*v)).transpose(0, 2, 1, 3)

                    m = m.transpose(0, 2, 1, 3).reshape(self.ref_m_supercell.shape)
                    testing.assert_allclose(self.ref_m_supercell, m, atol=1e-12)
                except Exception:
                    print("When testing {} the following exception occurred:".format(eri))
                    raise

    def test_eig_kernel(self):
        """Tests container behavior (supercell)."""
        model = ktd.TDRHF(self.model_krhf)
        model.kernel()
        # vals = []
        # vecs = []
        o = v = 4
        # eri = ktd.PhysERI4(self.model_krhf)
        # for k in range(self.k):
        #     va, ve = ktd.kernel(self.model_krhf, k, driver='eig')
        #     vals.append(va)
        #     vecs.append(ve)

        # Concatenate everything
        ks = numpy.array(sum(([i] * len(model.e[i]) for i in range(self.k)), []))
        vals = numpy.concatenate(tuple(model.e[i] for i in range(self.k))).real
        vecs = numpy.concatenate(tuple(model.xy[i] for i in range(self.k)), axis=0)

        # Obtain order
        order = numpy.argsort(vals)

        # Sort
        vals = vals[order]
        vecs = vecs[order]
        ks = ks[order]

        # Verify
        testing.assert_allclose(vals, self.td_model_rhf_supercell.e, atol=1e-7)
        for k in range(self.k):
            # Prepare indexes
            r1, r2, c1, c2 = ktd.get_block_k_ix(model.eri, k)
            r = k2k(r1, r2)
            c = k2k(c1, c2)

            # Select roots
            selection = ks == k
            vecs_ref = self.td_model_rhf_supercell.xy[selection]
            vecs_test = vecs[selection]

            vecs_test_padded = numpy.zeros((len(vecs_test), 2 * self.k * self.k, o, v), dtype=vecs_test.dtype)
            vecs_test_padded[:, c] = vecs_test.reshape((len(vecs_test), 2 * self.k, o, v))
            vecs_test_padded = vecs_test_padded.reshape(vecs_ref.shape)

            testing.assert_equal(vecs_test.shape, (self.k * o * v, 2, self.k, o, v))
            testing.assert_equal(vecs_test_padded.shape, (self.k * o * v, 2, self.k, self.k, o, v))
            assert_vectors_close(vecs_test_padded, vecs_ref, atol=1e-7)


class DiamondTest3(DiamondTest):
    """Compare this (krhf_slow) @3kp@Gamma vs vs `krhf_slow_supercell`."""
    k = 3
    k_c = (0.1, 0, 0)
    test8 = False
