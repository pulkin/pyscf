#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
JK with analytic Fourier transformation
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC
from pyscf.pbc.df.df_jk import is_zero, gamma_point, _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    dmsR = dms.real.reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.reshape(nset,nkpts,nao**2)
    kpt_allow = numpy.zeros(3)
    coulG = mydf.weighted_coulG(kpt_allow, False, mydf.gs)
    ngs = len(coulG)
    vR = numpy.zeros((nset,ngs))
    vI = numpy.zeros((nset,ngs))
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts, max_memory=max_memory):
        #:rho = numpy.einsum('lkL,lk->L', pqk.conj(), dm)
        for i in range(nset):
            rhoR = numpy.dot(dmsR[i,k], pqkR)
            rhoR+= numpy.dot(dmsI[i,k], pqkI)
            rhoI = numpy.dot(dmsI[i,k], pqkR)
            rhoI-= numpy.dot(dmsR[i,k], pqkI)
            vR[i,p0:p1] += rhoR * coulG[p0:p1]
            vI[i,p0:p1] += rhoI * coulG[p0:p1]
    pqkR = pqkI = coulG = None
    weight = 1./len(kpts)
    vR *= weight
    vI *= weight

    t1 = log.timer_debug1('get_j pass 1 to compute J(G)', *t1)

    kpts_band, single_kpt_band = _format_kpts_band(kpts_band, kpts)
    gamma_point = abs(kpts_band).sum() < 1e-9
    nband = len(kpts_band)

    vjR = numpy.zeros((nset,nband,nao*nao))
    vjI = numpy.zeros((nset,nband,nao*nao))
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts_band,
                            max_memory=max_memory):
        for i in range(nset):
            vjR[i,k] += numpy.dot(pqkR, vR[i,p0:p1])
            vjR[i,k] -= numpy.dot(pqkI, vI[i,p0:p1])
        if not gamma_point:
            for i in range(nset):
                vjI[i,k] += numpy.dot(pqkI, vR[i,p0:p1])
                vjI[i,k] += numpy.dot(pqkR, vI[i,p0:p1])
    pqkR = pqkI = coulG = None

    if gamma_point:
        vj_kpts = vjR
    else:
        vj_kpts = vjR + vjI*1j
    vj_kpts = vj_kpts.reshape(nset,nband,nao,nao)
    t1 = log.timer_debug1('get_j pass 2', *t1)

    return _format_jks(vj_kpts, dm_kpts, kpts_band, kpts, single_kpt_band)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    swap_2e = (kpts_band is None)
    kpts_band, single_kpt_band = _format_kpts_band(kpts_band, kpts)
    nband = len(kpts_band)
    kk_table = kpts_band.reshape(-1,1,3) - kpts.reshape(1,-1,3)
    kk_todo = numpy.ones(kk_table.shape[:2], dtype=bool)
    vkR = numpy.zeros((nset,nband,nao,nao))
    vkI = numpy.zeros((nset,nband,nao,nao))
    dmsR = numpy.asarray(dms.real, order='C')
    dmsI = numpy.asarray(dms.imag, order='C')

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now)) * .8
    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    def make_kpt(kpt):  # kpt = kptj - kpti
        # search for all possible ki and kj that has ki-kj+kpt=0
        kk_match = numpy.einsum('ijx->ij', abs(kk_table + kpt)) < 1e-9
        kpti_idx, kptj_idx = numpy.where(kk_todo & kk_match)
        nkptj = len(kptj_idx)
        log.debug1('kpt = %s', kpt)
        log.debug2('kpti_idx = %s', kpti_idx)
        log.debug2('kptj_idx = %s', kptj_idx)
        kk_todo[kpti_idx,kptj_idx] = False
        if swap_2e and not is_zero(kpt):
            kk_todo[kptj_idx,kpti_idx] = False

        max_memory1 = max_memory * (nkptj+1)/(nkptj+5)
        blksize = max(int(max_memory1*4e6/(nkptj+5)/16/nao**2), 16)
        bufR = numpy.empty((blksize*nao**2))
        bufI = numpy.empty((blksize*nao**2))
        # Use DF object to mimic KRHF/KUHF object in function get_coulG
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt, True, mydf.gs)
        kptjs = kpts[kptj_idx]
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        for k, pqkR, pqkI, p0, p1 \
                in mydf.ft_loop(mydf.gs, kpt, kptjs, max_memory=max_memory1):
            ki = kpti_idx[k]
            kj = kptj_idx[k]
            coulG = numpy.sqrt(vkcoulG[p0:p1])

# case 1: k_pq = (pi|iq)
#:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            pqkR *= coulG
            pqkI *= coulG
            pLqR = lib.transpose(pqkR.reshape(nao,nao,-1), axes=(0,2,1), out=bufR)
            pLqI = lib.transpose(pqkI.reshape(nao,nao,-1), axes=(0,2,1), out=bufI)
            iLkR = numpy.empty((nao*(p1-p0),nao))
            iLkI = numpy.empty((nao*(p1-p0),nao))
            for i in range(nset):
                iLkR, iLkI = zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                                    dmsR[i,kj], dmsI[i,kj], 1, iLkR, iLkI)
                zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                       pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                       1, vkR[i,ki], vkI[i,ki], 1)

# case 2: k_pq = (iq|pi)
#:v4 = numpy.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,li->kj', v4, dm)
            if swap_2e and not is_zero(kpt):
                iLkR = iLkR.reshape(nao,-1)
                iLkI = iLkI.reshape(nao,-1)
                for i in range(nset):
                    iLkR, iLkI = zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                                        pLqI.reshape(nao,-1), 1, iLkR, iLkI)
                    zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                           iLkR.reshape(-1,nao), iLkI.reshape(-1,nao),
                           1, vkR[i,kj], vkI[i,kj], 1)
            pqkR = pqkI = coulG = pLqR = pLqI = iLkR = iLkI = None

    for ki, kpti in enumerate(kpts_band):
        for kj, kptj in enumerate(kpts):
            if kk_todo[ki,kj]:
                make_kpt(kptj-kpti)

    if (gamma_point(kpts) and gamma_point(kpts_band) and
        not numpy.iscomplexobj(dm_kpts)):
        vk_kpts = vkR
    else:
        vk_kpts = vkR + vkI * 1j
    vk_kpts *= 1./nkpts

    # G=0 was not included in the non-uniform grids
    if cell.dimension != 3 and exxdiv:
        assert(exxdiv.lower() == 'ewald')
        _ewald_exxdiv_for_G0(cell, kpts_band, dms, vk_kpts, kpts_band)

    return _format_jks(vk_kpts, dm_kpts, kpts_band, kpts, single_kpt_band)


##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
    vj = vk = None
    if kpt_band is not None and abs(kpt-kpt_band).sum() > 1e-9:
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = get_k_kpts(mydf, dm, hermi, kpt, kpt_band, exxdiv)
        if with_j:
            vj = get_j_kpts(mydf, dm, hermi, kpt, kpt_band)
        return vj, vk

    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not numpy.iscomplexobj(dms)

    kptii = numpy.asarray((kpt,kpt))
    kpt_allow = numpy.zeros(3)

    if with_j:
        vjcoulG = mydf.weighted_coulG(kpt_allow, False, mydf.gs)
        vjR = numpy.zeros((nset,nao,nao))
        vjI = numpy.zeros((nset,nao,nao))
    if with_k:
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt_allow, True, mydf.gs)
        vkR = numpy.zeros((nset,nao,nao))
        vkI = numpy.zeros((nset,nao,nao))
    dmsR = numpy.asarray(dms.real.reshape(nset,nao,nao), order='C')
    dmsI = numpy.asarray(dms.imag.reshape(nset,nao,nao), order='C')
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now)) * .8
    log.debug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
    t2 = t1

    # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
    #                 == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
    blksize = max(int(max_memory*.25e6/16/nao**2), 16)
    bufR = numpy.empty(blksize*nao**2)
    bufI = numpy.empty(blksize*nao**2)
    for pqkR, pqkI, p0, p1 in mydf.pw_loop(mydf.gs, kptii, max_memory=max_memory):
        t2 = log.timer_debug1('%d:%d ft_aopair'%(p0,p1), *t2)
        pqkR = pqkR.reshape(nao,nao,-1)
        pqkI = pqkI.reshape(nao,nao,-1)
        if with_j:
            #:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
            #:vj += numpy.einsum('ijkl,lk->ij', v4, dm)
            for i in range(nset):
                rhoR = numpy.einsum('pq,pqk->k', dmsR[i], pqkR)
                rhoR+= numpy.einsum('pq,pqk->k', dmsI[i], pqkI)
                rhoI = numpy.einsum('pq,pqk->k', dmsI[i], pqkR)
                rhoI-= numpy.einsum('pq,pqk->k', dmsR[i], pqkI)
                rhoR *= vjcoulG[p0:p1]
                rhoI *= vjcoulG[p0:p1]
                vjR[i] += numpy.einsum('pqk,k->pq', pqkR, rhoR)
                vjR[i] -= numpy.einsum('pqk,k->pq', pqkI, rhoI)
                if not j_real:
                    vjI[i] += numpy.einsum('pqk,k->pq', pqkR, rhoI)
                    vjI[i] += numpy.einsum('pqk,k->pq', pqkI, rhoR)
        #t2 = log.timer_debug1('        with_j', *t2)

        if with_k:
            coulG = numpy.sqrt(vkcoulG[p0:p1])
            pqkR *= coulG
            pqkI *= coulG
            #:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
            #:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            pLqR = lib.transpose(pqkR, axes=(0,2,1), out=bufR).reshape(-1,nao)
            pLqI = lib.transpose(pqkI, axes=(0,2,1), out=bufI).reshape(-1,nao)
            iLkR = numpy.ndarray((nao*(p1-p0),nao), buffer=pqkR)
            iLkI = numpy.ndarray((nao*(p1-p0),nao), buffer=pqkI)
            for i in range(nset):
                if k_real:
                    lib.dot(pLqR, dmsR[i], 1, iLkR)
                    lib.dot(pLqI, dmsR[i], 1, iLkI)
                    lib.dot(iLkR.reshape(nao,-1), pLqR.reshape(nao,-1).T, 1, vkR[i], 1)
                    lib.dot(iLkI.reshape(nao,-1), pLqI.reshape(nao,-1).T, 1, vkR[i], 1)
                else:
                    zdotNN(pLqR, pLqI, dmsR[i], dmsI[i], 1, iLkR, iLkI)
                    zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                           pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                           1, vkR[i], vkI[i])
            #t2 = log.timer_debug1('        with_k', *t2)
        pqkR = pqkI = coulG = pLqR = pLqI = iLkR = iLkI = None
        #t2 = log.timer_debug1('%d:%d'%(p0,p1), *t2)
    bufR = bufI = None
    t1 = log.timer_debug1('aft_jk.get_jk', *t1)

    if with_j:
        if j_real:
            vj = vjR
        else:
            vj = vjR + vjI * 1j
        vj = vj.reshape(dm.shape)
    if with_k:
        if k_real:
            vk = vkR
        else:
            vk = vkR + vkI * 1j
        if cell.dimension != 3 and exxdiv:
            assert(exxdiv.lower() == 'ewald')
            _ewald_exxdiv_for_G0(cell, kpt, dms, vk)
        vk = vk.reshape(dm.shape)
    return vj, vk


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import aft

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    df = aft.AFTDF(cell)
    df.gs = (15,)*3
    dm = pscf.RHF(cell).get_init_guess()
    vj, vk = df.get_jk(dm)
    print(numpy.einsum('ij,ji->', df.get_nuc(), dm), 'ref=-10.577490961074622')
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=5.3766911667862516')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=8.2255177602309022')

