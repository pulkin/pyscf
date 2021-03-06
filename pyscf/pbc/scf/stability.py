#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Wave Function Stability Analysis

Ref.
JCP, 66, 3045
JCP, 104, 9047

See also tddft/rhf.py and scf/newton_ah.py
'''

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import newton_ah
from pyscf.pbc.scf.newton_ah import _gen_rhf_response, _gen_uhf_response

def rhf_stability(mf, internal=True, external=False, verbose=None):
    mo_i = mo_e = None
    if internal:
        mo_i = rhf_internal(mf, verbose=verbose)
    if external:
        mo_e = rhf_external(mf, verbose=verbose)
    return mo_i, mo_e

def uhf_stability(mf, internal=True, external=False, verbose=None):
    mo_i = mo_e = None
    if internal:
        mo_i = uhf_internal(mf, verbose=verbose)
    if external:
        mo_e = uhf_external(mf, verbose=verbose)
    return mo_i, mo_e

def rhf_internal(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    e, v = lib.davidson(hop, x0, precond, tol=1e-4, verbose=log)
    if e < -1e-5:
        log.log('KRHF/KRKS wavefunction has an internal instablity')
        mo = _rotate_mo(mf.mo_coeff, mf.mo_occ, v)
    else:
        log.log('KRHF/KRKS wavefunction is stable in the intenral stablity analysis')
        mo = mf.mo_coeff
    return mo

def _rotate_mo(mo_coeff, mo_occ, dx):
    nkpts = len(mo_occ)
    nmo = mo_occ[0].size
    nocc = numpy.sum(mo_occ > 0, axis=1)
    mo = []
    ip = 0
    for k, no in enumerate(nocc):
        nv = nmo - no
        dr = numpy.zeros((nmo,nmo), dtype=numpy.complex128)
        dr[no:,:no] = dx[ip:ip+nv*no].reshape(nv,no)
        dr[:no,no:] =-dx[ip:ip+nv*no].reshape(nv,no).T.conj()
        mo.append(numpy.dot(mo_coeff[k], scipy.linalg.expm(dr)))
        ip += nv * no
    return mo

def _gen_hop_rhf_external(mf, verbose=None):
#FIXME: numerically unstable with small gs?
#TODO: Add a warning message for small gs.
    from pyscf.pbc.dft import numint
    from pyscf.pbc.tddft.rhf import _unpack
    cell = mf.cell
    kpts = mf.kpts

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nkpts = len(mo_occ)
    occidx = [numpy.where(mo_occ[k]==2)[0] for k in range(nkpts)]
    viridx = [numpy.where(mo_occ[k]==0)[0] for k in range(nkpts)]
    orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
    orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpts)]

    h1e = mf.get_hcore()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = h1e + mf.get_veff(cell, dm0)
    fock = [reduce(numpy.dot, (mo_coeff[k].T.conj(), fock_ao[k], mo_coeff[k]))
            for k in range(nkpts)]
    foo = [fock[k][occidx[k][:,None],occidx[k]] for k in range(nkpts)]
    fvv = [fock[k][viridx[k][:,None],viridx[k]] for k in range(nkpts)]

    hdiag = [(fvv[k].diagonal().reshape(-1,1)-foo[k].diagonal()) * 2
             for k in range(nkpts)]
    hdiag = numpy.hstack([x.ravel() for x in hdiag])

    vresp1 = _gen_rhf_response(mf, singlet=False, hermi=1)
    def hop_rhf2uhf(x1):
        x1 = _unpack(x1, mo_occ)
        dmvo = []
        for k in range(nkpts):
            # *2 for double occupancy
            dm1 = reduce(numpy.dot, (orbv[k], x1[k]*2, orbo[k].T.conj()))
            dmvo.append(dm1 + dm1.T.conj())
        dmvo = lib.asarray(dmvo)

        v1ao = vresp1(dmvo)
        x2 = [0] * nkpts
        for k in range(nkpts):
            x2[k] = numpy.einsum('ps,sq->pq', fvv[k], x1[k])
            x2[k]-= numpy.einsum('ps,rp->rs', foo[k], x1[k])
            x2[k]+= reduce(numpy.dot, (orbv[k].T.conj(), v1ao[k], orbo[k]))
        return numpy.hstack([x.ravel() for x in x2])

    return hop_rhf2uhf, hdiag


def rhf_external(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    hop2, hdiag2 = _gen_hop_rhf_external(mf)

    def precond(dx, e, x0):
        hdiagd = hdiag2 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag2)
    x0[hdiag2>1e-5] = 1. / hdiag2[hdiag2>1e-5]
    e3, v3 = lib.davidson(hop2, x0, precond, tol=1e-4, verbose=log)
    if e3 < -1e-5:
        log.log('KRHF/KRKS wavefunction has an KRHF/KRKS -> KUHF/KRKS instablity.')
        nocca = numpy.count_nonzero(mf.mo_occ[0]> 0)
        nvira = numpy.count_nonzero(mf.mo_occ[0]==0)
        mo = (_rotate_mo(mf.mo_coeff, mf.mo_occ, v3), mf.mo_coeff)
    else:
        log.log('KRHF/KRKS wavefunction is stable in the KRHF/KRKS -> KUHF/KRKS stablity analysis')
        mo = (mf.mo_coeff, mf.mo_coeff)
    return mo

def uhf_internal(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
    def precond(dx, e, x0):
        hdiagd = hdiag - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd

    x0 = numpy.zeros_like(g)
    x0[g!=0] = 1. / hdiag[g!=0]
    e, v = lib.davidson(hop, x0, precond, tol=1e-4, verbose=log)
    if e < -1e-5:
        log.log('KUHF/KUKS wavefunction has an internal instablity.')
        tot_x_a = sum((occ>0).sum()*(occ==0).sum() for occ in mf.mo_occ[0])
        mo = (_rotate_mo(mf.mo_coeff[0], mf.mo_occ[0], v[:tot_x_a]),
              _rotate_mo(mf.mo_coeff[1], mf.mo_occ[1], v[tot_x_a:]))
    else:
        log.log('KUHF/KUKS wavefunction is stable in the intenral stablity analysis')
        mo = mf.mo_coeff
    return mo

def _gen_hop_uhf_external(mf, verbose=None):
    cell = mf.cell
    kpts = mf.kpts

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nkpts = len(mo_occ[0])
    occidxa = [numpy.where(mo_occ[0][k]>0)[0] for k in range(nkpts)]
    occidxb = [numpy.where(mo_occ[1][k]>0)[0] for k in range(nkpts)]
    viridxa = [numpy.where(mo_occ[0][k]==0)[0] for k in range(nkpts)]
    viridxb = [numpy.where(mo_occ[1][k]==0)[0] for k in range(nkpts)]
    nocca = [len(occidxa[k]) for k in range(nkpts)]
    noccb = [len(occidxb[k]) for k in range(nkpts)]
    nvira = [len(viridxa[k]) for k in range(nkpts)]
    nvirb = [len(viridxb[k]) for k in range(nkpts)]
    moa, mob = mo_coeff
    orboa = [moa[k][:,occidxa[k]] for k in range(nkpts)]
    orbva = [moa[k][:,viridxa[k]] for k in range(nkpts)]
    orbob = [mob[k][:,occidxb[k]] for k in range(nkpts)]
    orbvb = [mob[k][:,viridxb[k]] for k in range(nkpts)]

    h1e = mf.get_hcore()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = h1e + mf.get_veff(cell, dm0)
    focka = [reduce(numpy.dot, (moa[k].T.conj(), fock_ao[0][k], moa[k]))
             for k in range(nkpts)]
    fockb = [reduce(numpy.dot, (mob[k].T.conj(), fock_ao[1][k], mob[k]))
             for k in range(nkpts)]
    fooa = [focka[k][occidxa[k][:,None],occidxa[k]] for k in range(nkpts)]
    fvva = [focka[k][viridxa[k][:,None],viridxa[k]] for k in range(nkpts)]
    foob = [fockb[k][occidxb[k][:,None],occidxb[k]] for k in range(nkpts)]
    fvvb = [fockb[k][viridxb[k][:,None],viridxb[k]] for k in range(nkpts)]

    h_diag = ([fvva[k].diagonal().reshape(-1,1)-fooa[k].diagonal() for k in range(nkpts)] +
              [fvvb[k].diagonal().reshape(-1,1)-foob[k].diagonal() for k in range(nkpts)])

    hdiagab = [fvva[k].diagonal().reshape(-1,1) - foob[k].diagonal() for k in range(nkpts)]
    hdiagba = [fvvb[k].diagonal().reshape(-1,1) - fooa[k].diagonal() for k in range(nkpts)]
    hdiag2 = numpy.hstack([x.ravel() for x in (hdiagab + hdiagba)])

    vresp1 = _gen_uhf_response(mf, with_j=False, hermi=0)
    def hop_uhf2ghf(x1):
        x1ab = []
        x1ba = []
        ip = 0
        for k in range(nkpts):
            nv = nvira[k]
            no = noccb[k]
            x1ab.append(x1[ip:ip+nv*no].reshape(nv,no))
            ip += nv * no
        for k in range(nkpts):
            nv = nvirb[k]
            no = nocca[k]
            x1ba.append(x1[ip:ip+nv*no].reshape(nv,no))
            ip += nv * no

        dm1ab = []
        dm1ba = []
        for k in range(nkpts):
            d1ab = reduce(numpy.dot, (orbva[k], x1ab[k], orbob[k].T.conj()))
            d1ba = reduce(numpy.dot, (orbvb[k], x1ba[k], orboa[k].T.conj()))
            dm1ab.append(d1ab+d1ba.T.conj())
            dm1ba.append(d1ba+d1ab.T.conj())

        v1ao = vresp1(lib.asarray([dm1ab,dm1ba]))
        x2ab = [0] * nkpts
        x2ba = [0] * nkpts
        for k in range(nkpts):
            x2ab[k] = numpy.einsum('pr,rq->pq', fvva[k], x1ab[k])
            x2ab[k]-= numpy.einsum('sq,ps->pq', foob[k], x1ab[k])
            x2ba[k] = numpy.einsum('pr,rq->pq', fvvb[k], x1ba[k])
            x2ba[k]-= numpy.einsum('qs,ps->pq', fooa[k], x1ba[k])
            x2ab[k] += reduce(numpy.dot, (orbva[k].T.conj(), v1ao[0][k], orbob[k]))
            x2ba[k] += reduce(numpy.dot, (orbvb[k].T.conj(), v1ao[1][k], orboa[k]))
        return numpy.hstack([x.ravel() for x in (x2ab+x2ba)])

    return hop_uhf2ghf, hdiag2


def uhf_external(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    hop2, hdiag2 = _gen_hop_uhf_external(mf)

    def precond(dx, e, x0):
        hdiagd = hdiag2 - e
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        return dx/hdiagd
    x0 = numpy.zeros_like(hdiag2)
    x0[hdiag2>1e-5] = 1. / hdiag2[hdiag2>1e-5]
    e3, v = lib.davidson(hop2, x0, precond, tol=1e-4, verbose=log)
    log.debug('uhf_external: lowest eigs of H = %s', e3)
    mo = None
    if e3 < -1e-5:
        log.log('KUHF/KUKS wavefunction has an KUHF/KUKS -> KGHF/KGKS instablity.')
    else:
        log.log('KUHF/KUKS wavefunction is stable in the KUHF/KUKS -> KGHF/KGKS stablity analysis')
    return mo


if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import scf, dft
    from pyscf.pbc import df
    cell = gto.Cell()
    cell.unit = 'B'
    cell.atom = '''
    C  0.          0.          0.        
    C  1.68506879  1.68506879  1.68506879
    '''
    cell.a = '''
    0.          3.37013758  3.37013758
    3.37013758  0.          3.37013758
    3.37013758  3.37013758  0.
    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = [12]*3
    cell.build()
    kpts = cell.make_kpts([2,1,1])
    mf = scf.KRHF(cell, kpts).set(exxdiv=None)
    #mf.with_df = df.DF(cell, kpts)
    #mf.with_df.auxbasis = 'weigend'
    #mf.with_df._cderi = 'eri3d-df.h5'
    #mf.with_df.build(with_j3c=False)
    mf.run()
    rhf_stability(mf, True, True, verbose=5)

    mf = scf.KUHF(cell, kpts).set(exxdiv=None).run()
    uhf_stability(mf, True, True, verbose=5)

    mf = dft.KRKS(cell, kpts).set(xc='bp86').run()
    rhf_stability(mf, True, True, verbose=5)

    mf = dft.KUKS(cell, kpts).set(xc='bp86').run()
    uhf_stability(mf, True, True, verbose=5)
