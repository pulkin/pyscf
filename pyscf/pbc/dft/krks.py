#!/usr/bin/env python
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.rks.py : Non-relativistic Restricted Kohn-Sham for periodic
                           systems at a single k-point
'''

import time
import numpy as np
from pyscf.pbc.scf import khf
from pyscf.lib import logger
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc.dft import rks


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional

    .. note::
        This is a replica of pyscf.dft.rks.get_veff with kpts added.
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.  The ._exc and ._ecoul attributes
            will be updated after return.  Attributes ._dm_last, ._vj_last and
            ._vk_last might be changed if direct SCF method is applied.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        Veff : (nkpts, nao, nao) or (*, nkpts, nao, nao) ndarray
        Veff = J + Vxc.
    '''
    needs_ao_update = False
    
    if cell is None:
        cell = ks.cell
    elif not cell == ks.cell:
        ks.cell = cell
        needs_ao_update = True
    
    if kpts is None:
        kpts = ks.kpts
    elif not np.array_equal(kpts, ks.kpts):
        ks.kpts = np.array(kpts)
        needs_ao_update = True
        
    if dm is None: dm = ks.make_rdm1()
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        small_rho_cutoff = 0
    
    ao_derivatives_required = {
        "LDA":0,
        "GGA":1,
        "MGGA":2,
    }[ks._numint._xc_type(ks.xc)]
    
    # If needs an update
    if ks._ao is None or needs_ao_update:
        ks._update_ao(ao_derivatives_required)
        
    # If number of derivatives is not sufficiently large: update as well
    shape = ks._ao[0].shape
    # Retrieve primary dimension
    if len(shape) == 2:
        prim = 1
    else:
        prim = shape[0]
    
    n = (ao_derivatives_required+1)*(ao_derivatives_required+2)*(ao_derivatives_required+3)//6
    if n > prim:
        ks._update_ao(ao_derivatives_required)

    if hermi == 2:  # because rho = 0
        n, ks._exc, vx = 0, 0, 0
    else:
        n, ks._exc, vx = ks._numint.nr_rks(cell, ks.grids, ks.xc, dm, 1,
                                           kpts, kpts_band, precomputed_ao = ks._ao)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    # ndim = 3 : dm.shape = (nkpts, nao, nao)
    ground_state = (isinstance(dm, np.ndarray) and dm.ndim == 3 and
                    kpts_band is None)
    nkpts = len(kpts)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=cell.spin)
    if abs(hyb) < 1e-10:
        vhf = vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
    else:
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vhf = vj - vk * (hyb * .5)

        if ground_state:
            ks._exc -= (1./nkpts) * np.einsum('Kij,Kji', dm, vk).real * .5 * hyb*.5

    if ground_state:
        ks._ecoul = (1./nkpts) * np.einsum('Kij,Kji', dm, vj).real * .5

    if (small_rho_cutoff > 1e-20 and ground_state and
        abs(n-cell.nelectron) < 0.01*n):
        # Filter grids the first time setup grids
        _ao = ks._ao if len(ks._ao[0].shape) == 2 else list(i[0] for i in ks._ao)
        idx = ks._numint.large_rho_indices(cell, dm, ks.grids,
                                           small_rho_cutoff, kpts,
                                           precomputed_ao = _ao)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - np.count_nonzero(idx))
        ks.grids.coords  = np.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = np.asarray(ks.grids.weights[idx], order='C')
        ks.grids.non0tab = ks.grids.make_mask(cell, ks.grids.coords)
    return vhf + vx


class KRKS(khf.KRHF):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3))):
        khf.KRHF.__init__(self, cell, kpts)
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.UniformGrids(cell)
        self.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
        self._ecoul = 0
        self._exc = 0
        # Note Do not refer to .with_df._numint because gs/coords may be different
        self._numint = numint._KNumInt(kpts)
        self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])
        self._ao = None
        
    def _update_ao(self,deriv):
        """
        Updates atomic orbitals.
        """
        self._ao = self._numint.eval_ao(
            self.cell,
            self.grids.coords,
            self.kpts,
            deriv = deriv,
        )

    def dump_flags(self):
        khf.KRHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        nkpts = len(h1e_kpts)
        e1 = 1./nkpts * np.einsum('kij,kji', h1e_kpts, dm_kpts).real

        tot_e = e1 + self._ecoul + self._exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, self._ecoul, self._exc)
        return tot_e, self._ecoul + self._exc

    density_fit = rks._patch_df_beckegrids(khf.KRHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(khf.KRHF.mix_density_fit)

