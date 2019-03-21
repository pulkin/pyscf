#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Pseudo-spectral methods (COSX, PS, SN-K)
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.sgx import sgx_jk
from pyscf.df import df_jk
from pyscf import __config__

def sgx_fit(mf, auxbasis=None, with_df=None):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding SGX or density fitting integrals.

    Args:
        mf : an SCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.

    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = sgx_fit(scf.RHF(mol))
    >>> mf.scf()
    -100.00978770917165

    >>> mol.symmetry = 1
    >>> mol.build(0, 0)
    >>> mf = sgx_fit(scf.UHF(mol))
    >>> mf.scf()
    -100.00978770951018
    '''
    from pyscf import scf
    from pyscf import df
    from pyscf.soscf import newton_ah
    assert(isinstance(mf, scf.hf.SCF))

    if isinstance(mf, _SGXHF):
        if mf.with_df is None:
            mf = mf.__class__(mf)
        elif mf.with_df.auxbasis != auxbasis:
            if (isinstance(mf, newton_ah._CIAH_SOSCF) and
                isinstance(mf._scf, _SGXHF)):
                mf.with_df = copy.copy(mf.with_df)
                mf.with_df.auxbasis = auxbasis
            else:
                raise RuntimeError('DFHF has been initialized. '
                                   'It cannot be initialized twice.')
        return mf

    if with_df is None:
        with_df = SGX(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    mf_class = mf.__class__
    class SGXHF(mf_class, _SGXHF):
        def __init__(self, mf):
            self.__dict__.update(mf.__dict__)
            self._eri = None
            self.auxbasis = auxbasis
            self.with_df = with_df

            # Grids/Integral quality varies during SCF. VHF cannot be
            # constructed incrementally.
            self.direct_scf = False

            self._last_dm = 0
            self._in_scf = False
            self._keys = self._keys.union(['auxbasis', 'with_df'])

        def build(self, mol=None, **kwargs):
            if self.direct_scf:
                self.with_df.build(self.with_df.grids_level_f)
            else:
                self.with_df.build(self.with_df.grids_level_i)
            return mf_class.build(self, mol, **kwargs)

        def pre_kernel(self, envs):
            self._in_scf = True

        def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
            if dm is None: dm = self.make_rdm1()
            with_df = self.with_df
            if not with_df:
                return mf_class.get_jk(self, mol, dm, hermi, with_j, with_k)

            if self._in_scf and not self.direct_scf:
                if numpy.linalg.norm(dm - self._last_dm) < with_df.grids_switch_thrd:
                    logger.debug(self, 'Switching SGX grids')
                    with_df.build(with_df.grids_level_f)
                    self._in_scf = False
                    self._last_dm = 0
                else:
                    self._last_dm = numpy.asarray(dm)

            return with_df.get_jk(dm, hermi, with_j, with_k, self.direct_scf_tol)

        def post_kernel(self, envs):
            self._in_scf = False
            self._last_dm = 0

        def nuc_grad_method(self):
            raise NotImplementedError

    return SGXHF(mf)

# A tag to label the derived SCF class
class _SGXHF:
    pass


class SGX(lib.StreamObject):
    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.grids_thrd = 1e-10
        self.grids_level_i = 0  # initial grids level
        self.grids_level_f = 1  # final grids level
        self.grids_switch_thrd = 0.03
        self.dfj = False  # compute J matrix using DF
        self._auxbasis = auxbasis

        self.grids = None
        self.blockdim = 600
        self.auxmol = None
        self._vjopt = None
        self._last_dm = 0
        self._keys = set(self.__dict__.keys())

    @property
    def auxbasis(self):
        return self._auxbasis
    @auxbasis.setter
    def auxbasis(self, x):
        if self._auxbasis != x:
            self._auxbasis = x
            self.auxmol = None

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('******** %s ********', self.__class__)
        log.info('max_memory = %s', self.max_memory)
        log.info('grids_level_i = %s', self.grids_level_i)
        log.info('grids_level_f = %s', self.grids_level_f)
        log.info('grids_thrd = %s', self.grids_thrd)
        log.info('grids_switch_thrd = %s', self.grids_switch_thrd)
        log.info('df_j = %s', self.df_j)
        log.info('auxbasis = %s', self.auxbasis)
        return self

    # To mimic DF object, so that SGX can be used as in DF-SCF method by setting
    # mf.with_df = SGX(mol)
    @property
    def _cderi(self):
        return self.grids

    def build(self, level=None):
        if level is None:
            level = self.grids_level_f
        self.grids = sgx_jk.get_gridss(self.mol, level, self.grids_thrd)
        return self

    def kernel(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def reset(self, mol):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        self.mol = mol
        self.grids = None
        self.auxmol = None
        self._cderi = None
        self._vjopt = None
        self._last_dm = 0
        return self

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)):
        if with_j and self.dfj:
            vj = df_jk.get_j(self, dm, hermi, direct_scf_tol)
            if with_k:
                vk = sgx_jk.get_jk(self, dm, hermi, False, with_k, direct_scf_tol)[1]
            else:
                vk = None
        else:
            vj, vk = sgx_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        return vj, vk


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.build(
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )
    method = sgx_fit(scf.RHF(mol), 'weigend')
    energy = method.scf()
    print(energy - -76.02673747045691)

    method.with_df.dfj = True
    energy = method.scf()
    print(energy - -76.02686422219752)
