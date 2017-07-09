#!/usr/bin/env python

import unittest
import numpy
from pyscf import lib, gto, scf
from pyscf import ao2mo
from pyscf.scf import stability

def gen_hop_rhf_external(mf):
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nmo = mo_coeff.shape[1]
    nocc = numpy.count_nonzero(mo_occ)
    nvir = nmo - nocc
    nov = nocc * nvir

    eri_mo = ao2mo.full(mol, mo_coeff)
    eri_mo = ao2mo.restore(1, eri_mo, nmo)
    eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
    # A
    h = numpy.einsum('ckld->kcld', eri_mo[nocc:,:nocc,:nocc,nocc:]) * 2
    h-= numpy.einsum('cdlk->kcld', eri_mo[nocc:,nocc:,:nocc,:nocc])
    for a in range(nvir):
        for i in range(nocc):
            h[i,a,i,a] += eai[a,i]
    # B
    h-= numpy.einsum('ckdl->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc]) * 2
    h+= numpy.einsum('cldk->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc])
    h1 = h.transpose(1,0,3,2).reshape(nov,nov)
    def hop1(x):
        return h1.dot(x)

    h =-numpy.einsum('cdlk->kcld', eri_mo[nocc:,nocc:,:nocc,:nocc])
    for a in range(nvir):
        for i in range(nocc):
            h[i,a,i,a] += eai[a,i]
    h-= numpy.einsum('cldk->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc])
    h2 = h.transpose(1,0,3,2).reshape(nov,nov)
    def hop2(x):
        return h2.dot(x)
    return hop1, hop2

def gen_hop_uhf_external(mf):
    mol = mf.mol
    mo_a, mo_b = mf.mo_coeff
    mo_ea, mo_eb = mf.mo_energy
    mo_occa, mo_occb = mf.mo_occ
    nmo = mo_a.shape[1]
    nocca = numpy.count_nonzero(mo_occa)
    noccb = numpy.count_nonzero(mo_occb)
    nvira = nmo - nocca
    nvirb = nmo - noccb

    eri_aa = ao2mo.restore(1, ao2mo.full(mol, mo_a), nmo)
    eri_ab = ao2mo.restore(1, ao2mo.general(mol, [mo_a,mo_a,mo_b,mo_b]), nmo)
    eri_bb = ao2mo.restore(1, ao2mo.full(mol, mo_b), nmo)
    # alpha -> alpha
    haa =-numpy.einsum('abji->iajb', eri_aa[nocca:,nocca:,:nocca,:nocca])
    haa+= numpy.einsum('ajbi->iajb', eri_aa[nocca:,:nocca,nocca:,:nocca])
    for a in range(nvira):
        for i in range(nocca):
            haa[i,a,i,a] += mo_ea[nocca+a] - mo_ea[i]
    # beta -> beta
    hbb =-numpy.einsum('abji->iajb', eri_bb[noccb:,noccb:,:noccb,:noccb])
    hbb+= numpy.einsum('ajbi->iajb', eri_bb[noccb:,:noccb,noccb:,:noccb])
    for a in range(nvirb):
        for i in range(noccb):
            hbb[i,a,i,a] += mo_eb[noccb+a] - mo_eb[i]

    nova = nocca * nvira
    novb = noccb * nvirb
    h1 = numpy.zeros((nova+novb,nova+novb))
    h1[:nova,:nova] = haa.transpose(1,0,3,2).reshape(nova,nova)
    h1[nova:,nova:] = hbb.transpose(1,0,3,2).reshape(novb,novb)
    def hop1(x):
        return h1.dot(x)

    h11 =-numpy.einsum('abji->iajb', eri_ab[nocca:,nocca:,:noccb,:noccb])
    for a in range(nvira):
        for i in range(noccb):
            h11[i,a,i,a] += mo_ea[nocca+a] - mo_eb[i]
    h22 =-numpy.einsum('jiab->iajb', eri_ab[:nocca,:nocca,noccb:,noccb:])
    for a in range(nvirb):
        for i in range(nocca):
            h22[i,a,i,a] += mo_eb[noccb+a] - mo_ea[i]
    h12 =-numpy.einsum('ajbi->iajb', eri_ab[nocca:,:nocca,noccb:,:noccb])
    h21 =-numpy.einsum('biaj->iajb', eri_ab[nocca:,:nocca,noccb:,:noccb])

    n1 = noccb * nvira
    n2 = nocca * nvirb
    h2 = numpy.empty((n1+n2,n1+n2))
    h2[:n1,:n1] = h11.transpose(1,0,3,2).reshape(n1,n1)
    h2[n1:,n1:] = h22.transpose(1,0,3,2).reshape(n2,n2)
    h2[:n1,n1:] = h12.transpose(1,0,3,2).reshape(n1,n2)
    h2[n1:,:n1] = h21.transpose(1,0,3,2).reshape(n2,n1)
    def hop2(x):
        return h2.dot(x)
    return hop1, hop2


class KnowValues(unittest.TestCase):
    def test_rhf_external_hop(self):
        mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*')
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        hop1ref, hop2ref = gen_hop_rhf_external(mf)
        hop1, hdiag1, hop2, hdiag2 = stability._gen_hop_rhf_external(mf)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag1.size)
        self.assertAlmostEqual(numpy.linalg.norm(hop1(x1) - hop1ref(x1)), 0, 7)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag2.size)
        self.assertAlmostEqual(numpy.linalg.norm(hop2(x1) - hop2ref(x1)), 0, 7)

        mol = gto.M(atom='N 0 0 0; N 0 0 1.2222', basis='631g*', symmetry=1)
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        hop1ref, hop2ref = gen_hop_rhf_external(mf)
        hop1, hdiag1, hop2, hdiag2 = stability._gen_hop_rhf_external(mf)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag1.size)
        x1[hdiag1==0] = 0
        xref = hop1ref(x1)
        xref[hdiag1==0] = 0
        self.assertAlmostEqual(numpy.linalg.norm(hop1(x1) - xref), 0, 7)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag2.size)
        x1[hdiag2==0] = 0
        xref = hop2ref(x1)
        xref[hdiag2==0] = 0
        self.assertAlmostEqual(numpy.linalg.norm(hop2(x1) - xref), 0, 7)

    def test_uhf_external_hop(self):
        mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*', spin=2)
        mf = scf.UHF(mol).run(conv_tol=1e-14)
        hop1ref, hop2ref = gen_hop_uhf_external(mf)
        hop1, hdiag1, hop2, hdiag2 = stability._gen_hop_uhf_external(mf)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag1.size)
        self.assertAlmostEqual(numpy.linalg.norm(hop1(x1) - hop1ref(x1)), 0, 7)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag2.size)
        self.assertAlmostEqual(numpy.linalg.norm(hop2(x1) - hop2ref(x1)), 0, 7)

        mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*', symmetry=1, spin=2)
        mf = scf.UHF(mol).run(conv_tol=1e-14)
        hop1ref, hop2ref = gen_hop_uhf_external(mf)
        hop1, hdiag1, hop2, hdiag2 = stability._gen_hop_uhf_external(mf)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag1.size)
        x1[hdiag1==0] = 0
        xref = hop1ref(x1)
        xref[hdiag1==0] = 0
        self.assertAlmostEqual(numpy.linalg.norm(hop1(x1) - xref), 0, 7)

        numpy.random.seed(1)
        x1 = numpy.random.random(hdiag2.size)
        x1[hdiag2==0] = 0
        xref = hop2ref(x1)
        xref[hdiag2==0] = 0
        self.assertAlmostEqual(numpy.linalg.norm(hop2(x1) - xref), 0, 7)

if __name__ == "__main__":
    print("Full Tests for stability")
    unittest.main()

