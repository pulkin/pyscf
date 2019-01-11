from pyscf.gto import Mole
from pyscf.scf import RHF, RKS
from pyscf.gw import GW
from pyscf.gw.gw_slow import GW as GW_slow
from pyscf.tdscf import TDRHF
from pyscf.tdscf.rhf_slow import TDRHF as TDRHF_slow

from numpy import testing
import unittest

from test_common import assert_vectors_close


class H20Test(unittest.TestCase):
    """Compare gw and gw_slow."""
    @classmethod
    def setUpClass(cls):
        cls.mol = mol = Mole()
        mol.atom = [
            [8, (0., 0., 0.)],
            [1, (0., -0.757, 0.587)],
            [1, (0., 0.757, 0.587)]]

        mol.basis = 'cc-pvdz'
        mol.verbose = 5
        mol.build()

        cls.model_rhf = model_rhf = RHF(mol)
        model_rhf.kernel()

        cls.model_rks = model_rks = RKS(mol)
        model_rks.xc = 'hf'
        model_rks.kernel()

        testing.assert_allclose(model_rhf.mo_energy, model_rks.mo_energy)
        assert_vectors_close(model_rhf.mo_coeff.T, model_rks.mo_coeff.T)

        model_rks.mo_coeff = model_rhf.mo_coeff

        cls.td_model_rks = td_model_rks = TDRHF(model_rks)
        td_model_rks.nroots = 4
        td_model_rks.kernel()

        cls.td_model_rhf_slow = td_model_rhf_slow = TDRHF_slow(model_rhf)
        td_model_rhf_slow.nroots = td_model_rks.nroots
        td_model_rhf_slow.kernel()

    @classmethod
    def tearDownClass(cls):
        # These are here to remove temporary files
        del cls.td_model_rhf_slow
        del cls.td_model_rks
        del cls.model_rhf
        del cls.model_rks
        del cls.mol

    def test_gw(self):
        """Tests container behavior."""
        gw = GW(self.model_rks, self.td_model_rks)
        gw.kernel()

        gw_slow = GW_slow(self.td_model_rhf_slow)
        gw_slow.kernel()

        testing.assert_allclose(gw_slow.mo_energy, gw.mo_energy, atol=1e-6)
