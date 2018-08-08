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
# Authors: James D. McClain
#          Mario Motta
#          Yang Gao
#          Qiming Sun <osirpt.sun@gmail.com>
#          Jason Yu
#

import itertools
import time
from functools import reduce
import numpy as np
import h5py

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.cc import uccsd
from pyscf.cc import eom_uccsd
from pyscf.cc import eom_rccsd
from pyscf.pbc.cc import kccsd
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import member, gamma_point
from pyscf import __config__
from pyscf.pbc.cc import kintermediates as imd

einsum = lib.einsum

def kernel(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):
    '''Calculate excitation energy via eigenvalue solver

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested per k-point
        koopmans : bool
            Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
        left : bool
            If True, calculates left eigenvectors rather than right eigenvectors.
        eris : `object(uccsd._ChemistsERIs)`
            Holds uccsd electron repulsion integrals in chemist notation.
        imds : `object(_IMDS)`
            Holds eom intermediates in chemist notation.
        partition : bool or str
            Use a matrix-partitioning for the doubles-doubles block.
            Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
            or 'full' (full diagonal elements).
        kptlist : list
            List of k-point indices for which eigenvalues are requested.
        dtype : type
            Type for eigenvectors.
    '''
    cput0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    size = eom.vector_size()
    nroots = min(nroots,size)
    nkpts = eom.nkpts

    if kptlist is None:
        kptlist = range(nkpts)

    if dtype is None:
        dtype = imds.t1.dtype

    evals = np.zeros((len(kptlist),nroots), np.float)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)
    convs = np.zeros((len(kptlist),nroots), dtype)

    for k, kshift in enumerate(kptlist):
        matvec, diag = eom.gen_matvec(kshift, imds, left=left, **kwargs)

        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            user_guess = False
            guess = eom.get_init_guess(nroots, koopmans, diag)

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = lib.davidson_nosym1
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            conv_k, evals_k, evecs_k = eig(matvec, guess, precond, pick=pickeig,
                                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                                           max_space=eom.max_space, nroots=nroots, verbose=eom.verbose)
        else:
            conv_k, evals_k, evecs_k = eig(matvec, guess, precond,
                                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                                           max_space=eom.max_space, nroots=nroots, verbose=eom.verbose)

        evals_k = evals_k.real
        evals[k] = evals_k
        evecs[k] = evecs_k
        convs[k] = conv_k

        if nroots == 1:
            evals_k, evecs_k = [evals_k], [evecs_k]
        for n, en, vn in zip(range(nroots), evals_k, evecs_k):
            r1, r2 = eom.vector_to_amplitudes(vn)
            qp_weight = np.linalg.norm(r1)**2
            logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g',
                        n, en, qp_weight)
    log.timer('EOM-CCSD', *cput0)
    return convs, evals, evecs

########################################
# EOM-IP-CCSD
########################################

def vector_to_amplitudes_ip(vector, nkpts, nmo, nocc):
    nvir = nmo - nocc

    # TODO: some redundancy; can use symmetry of operator to reduce size
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nkpts,nkpts,nocc,nocc,nvir)
    return [r1,r2]

def amplitudes_to_vector_ip(r1,r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def ipccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{a }, i.e. 'ia' indices are coupled.
    This differs from the restricted case that uses s_{ij}^{ b}.'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ip(vector, nkpts, nmo, nocc)
    print np.linalg.norm(r1), np.linalg.norm(r2)

    Hr1 = -np.einsum('mi,m->i', imds.Foo[kshift], r1)
    for km in range(nkpts):
        Hr1 += np.einsum('me,mie->i', imds.Fov[km], r2[km, kshift])
        for kn in range(nkpts):
            Hr1 += - 0.5 * np.einsum('nmie,mne->i', imds.Wooov[kn, km, kshift],
                                     r2[km, kn])

    Hr2 = np.zeros_like(r2)
    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        Hr2[ki, kj] += lib.einsum('ae,ije->ija', imds.Fvv[ka], r2[ki, kj])

        Hr2[ki, kj] -= lib.einsum('mi,mja->ija', imds.Foo[ki], r2[ki, kj])
        Hr2[ki, kj] += lib.einsum('mj,mia->ija', imds.Foo[kj], r2[kj, ki])

        Hr2[ki, kj] -= np.einsum('maji,m->ija', imds.Wovoo[kshift, ka, kj], r1)
        for km in range(nkpts):
            kn = kconserv[ki, km, kj]
            Hr2[ki, kj] += 0.5 * lib.einsum('mnij,mna->ija',
                                            imds.Woooo[km, kn, ki], r2[km, kn])

    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        for km in range(nkpts):
            ke = kconserv[km, kshift, kj]
            Hr2[ki, kj] += lib.einsum('maei,mje->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, kj])

            ke = kconserv[km, kshift, ki]
            Hr2[ki, kj] -= lib.einsum('maej,mie->ija', imds.Wovvo[km, ka, ke],
                                      r2[km, ki])

    tmp = np.zeros(nvir, dtype=Hr2.dtype)
    for km, kn in itertools.product(range(nkpts), repeat=2):
        tmp = lib.einsum('mnef,mnf->e', imds.Woovv[km, kn, kshift], r2[km, kn])

    for ki, kj in itertools.product(range(nkpts), repeat=2):
        ka = kconserv[ki, kshift, kj]
        Hr2[ki, kj] += 0.5*lib.einsum('e,ijae->ija', tmp, imds.t2[ki, kj, ka])

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, kshift, imds=None):
    #TODO: find a way to check this
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nkpts, nocc, nvir = t1.shape
    kconserv = imds.kconserv

    Hr1 = -np.diag(imds.Foo[kshift])

    Hr2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir), dtype=t1.dtype)
    if eom.partition == 'mp':
        foo = eom.eris.fock[:,:nocc,:nocc]
        fvv = eom.eris.fock[:,nocc:,nocc:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kconserv[ki,kshift,kj]
                Hr2[ki,kj]  = fvv[ka].diagonal()
                Hr2[ki,kj] -= foo[ki].diagonal()[:,None,None]
                Hr2[ki,kj] -= foo[kj].diagonal()[None,:,None]
    else:
        idx = np.arange(nocc)
        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = kconserv[ki,kshift,kj]
                Hr2[ki,kj]  = imds.Fvv[ka].diagonal()
                Hr2[ki,kj] -= imds.Foo[ki].diagonal()[:,None,None]
                Hr2[ki,kj] -= imds.Foo[kj].diagonal()[None,:,None]

                if ki == kconserv[ki,kj,kj]:
                    Hr2[ki,kj] += 0.5 * np.einsum('ijij->ij', imds.Woooo[ki, kj, ki])[:,:,None]

                Wovvo = np.einsum('iaai->ia', imds.Wovvo[ki,ka,ka])
                Hr2[ki,kj] += Wovvo[:, None, :]
                if ki == kj:  # and i == j
                    Hr2[ki,ki,idx,idx] -= Wovvo

                Hr2[ki, kj] += 0.5 * lib.einsum('ijea,ijae->ija', imds.Woovv[ki, kj, kshift],
                                                imds.t2[ki, kj, ka])

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None, **kwargs):
    '''See `kernel()` for a description of arguments.'''
    if partition:
        eom.partition = partition.lower()
        assert eom.partition in ['mp','full']
        if eom.partition in ['mp', 'full']:
            raise NotImplementedError
    eom.converged, eom.e, eom.v \
            = kernel(eom, nroots, koopmans, guess, left, eris=eris, imds=imds,
                     partition=partition, kptlist=kptlist, dtype=dtype)
    return eom.e, eom.v

class EOMIP(eom_rccsd.EOM):
    def __init__(self, cc):
        self.kpts = cc.kpts
        eom_rccsd.EOM.__init__(self, cc)

    kernel = ipccsd
    ipccsd = ipccsd
    get_diag = ipccsd_diag
    matvec = ipccsd_matvec

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc-n-1] = 1.0
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess

    @property
    def nkpts(self):
        return len(self.kpts)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nkpts=None, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes_ip(vector, nkpts, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds

########################################
# EOM-EA-CCSD
########################################

def vector_to_amplitudes_ea(vector, nkpts, nmo, nocc):
    nvir = nmo - nocc

    # TODO: some redundancy; can use symmetry of operator to reduce size
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nkpts,nkpts,nocc,nvir,nvir)
    return [r1,r2]

def amplitudes_to_vector_ea(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def eaccsd(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, partition=None, kptlist=None,
           dtype=None):
    '''See `ipccsd()` for a description of arguments.'''
    return ipccsd(eom, nroots, koopmans, guess, left, eris, imds,
                  partition, kptlist, dtype)

def eaccsd_matvec(eom, vector, kshift, imds=None, diag=None):
    '''2ph operators are of the form s_{ij}^{a }, i.e. 'ia' indices are coupled.
    This differs from the restricted case that uses s_{ij}^{ b}.'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    nkpts = eom.nkpts
    kconserv = imds.kconserv
    r1, r2 = vector_to_amplitudes_ea(vector, nkpts, nmo, nocc)
    print np.linalg.norm(r1), np.linalg.norm(r2)

    Hr1 = np.einsum('ac,c->a', imds.Fvv[kshift], r1)

    Hr2 = np.zeros_like(r2)

    vector = amplitudes_to_vector_ea(Hr1, Hr2)
    return vector

class EOMEA(eom_rccsd.EOM):
    def __init__(self, cc):
        self.kpts = cc.kpts
        eom_rccsd.EOM.__init__(self, cc)
        self.kshift = 0

    kernel = eaccsd
    eaccsd = eaccsd
    #get_diag = eaccsd_diag

    @property
    def nkpts(self):
        return len(self.kpts)

    def gen_matvec(self, kshift, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(kshift, imds)
        if left:
            raise NotImplementedError
            matvec = lambda xs: [self.l_matvec(x, kshift, imds, diag) for x in xs]
        else:
            raise NotImplementedError
            matvec = lambda xs: [self.matvec(x, kshift, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nkpts=None, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        if nkpts is None: nkpts = self.nkpts
        return vector_to_amplitudes_ea(vector, nkpts, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nocc + nkpts**2*nocc*nocc*nvir

    def make_imds(self, eris=None, t1=None, t2=None):
        imds = _IMDS(self._cc, eris, t1, t2)
        imds.make_ea()
        return imds

class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> gintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc, eris=None, t1=None, t2=None):
        self._cc = cc
        self.verbose = cc.verbose
        self.kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)
        self.stdout = cc.stdout
        if t1 is None:
            t1 = cc.t1
        self.t1 = t1
        if t2 is None:
            t2 = cc.t2
        self.t2 = t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        self.Foo = imd.Foo(self._cc, t1, t2, eris, kconserv)
        self.Fvv = imd.Fvv(self._cc, t1, t2, eris, kconserv)
        self.Fov = imd.Fov(self._cc, t1, t2, eris, kconserv)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(self._cc, t1, t2, eris, kconserv)
        self.Woovv = eris.oovv

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-CCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(self._cc, t1, t2, eris, kconserv)
        self.Wooov = imd.Wooov(self._cc, t1, t2, eris, kconserv)
        self.Wovoo = imd.Wovoo(self._cc, t1, t2, eris, kconserv)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(self._cc, t1, t2, eris, kconserv)
        self.Wvvvv = imd.Wvvvv(self._cc, t1, t2, eris, kconserv)
        self.Wvvvo = imd.Wvvvo(self._cc, t1, t2, eris, self.Wvvvv, kconserv)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        kconserv = self.kconserv
        t1, t2, eris = self.t1, self.t2, self.eris

        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.Woooo = imd.Woooo(self._cc, t1, t2, eris, kconserv)
            self.Wooov = imd.Wooov(self._cc, t1, t2, eris, kconserv)
            self.Wovoo = imd.Wovoo(self._cc, t1, t2, eris, kconserv)
        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.Wvovv = imd.Wvovv(self._cc, t1, t2, eris, kconserv)
            self.Wvvvv = imd.Wvvvv(self._cc, t1, t2, eris, kconserv)
            self.Wvvvo = imd.Wvvvo(self._cc, t1, t2, eris, self.Wvvvv, kconserv)

        self.made_ee_imds = True
        logger.timer(self, 'EOM-CCSD EE intermediates', *cput0)
        return self

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = { 'C': [[0, (0.8, 1.0)],
                         [1, (1.0, 1.0)]]}
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2]), exxdiv=None)
    kmf.conv_tol_grad = 1e-8
    ehf = kmf.kernel()

    mycc = cc.KGCCSD(kmf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-9
    eris = mycc.ao2mo(mycc.mo_coeff)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.155298393321855)

    eom = EOMIP(mycc)
    e, v = eom.ipccsd(nroots=3, kptlist=[0])
    print(e[0] + 0.8268853970451141)

    #e, v = eom.eaccsd(nroots=3, kptlist=[0])
    #print(e[0] - 1.073716355462168)