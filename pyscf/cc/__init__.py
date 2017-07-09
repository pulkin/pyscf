'''
Coupled Cluster
===============

Simple usage::

    >>> from pyscf import gto, scf, cc
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> cc.CCSD(mf).run()

:func:`cc.CCSD` returns an instance of CCSD class.  Followings are parameters
to control CCSD calculation.

    verbose : int
        Print level.  Default value equals to :class:`Mole.verbose`
    max_memory : float or int
        Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
    conv_tol : float
        converge threshold.  Default is 1e-7.
    conv_tol_normt : float
        converge threshold for norm(t1,t2).  Default is 1e-5.
    max_cycle : int
        max number of iterations.  Default is 50.
    diis_space : int
        DIIS space size.  Default is 6.
    diis_start_cycle : int
        The step to start DIIS.  Default is 0.
    direct : bool
        AO-direct CCSD. Default is False.
    frozen : int or list
        If integer is given, the inner-most orbitals are frozen from CC
        amplitudes.  Given the orbital indices (0-based) in a list, both
        occupied and virtual orbitals can be frozen in CC calculation.


Saved results

    converged : bool
        CCSD converged or not
    e_tot : float
        Total CCSD energy (HF + correlation)
    t1, t2 : 
        t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    l1, l2 : 
        Lambda amplitudes l1[i,a], l2[i,j,a,b]  (i,j in occ, a,b in virt)
'''

from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm
from pyscf.cc import addons

def CCSD(mf, frozen=[], mo_coeff=None, mo_occ=None):
    __doc__ = ccsd.CCSD.__doc__
    try:
        import sys
        from pyscf import dft
        if isinstance(mf, (dft.rks.RKS, dft.uks.UKS, dft.roks.ROKS,
                           dft.rks_symm.RKS, dft.uks_symm.UKS, dft.rks_symm.ROKS)):
            sys.stderr.write('CCSD Warning: The first argument mf is a DFT object. '
                             'CCSD calculation should be used with HF object')
    except:
        pass
    return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ)

def RCCSD(mf, frozen=[], mo_coeff=None, mo_occ=None):
    from pyscf.cc import rccsd
    mf = _convert_to_rhf(mf)
    return rccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

def UCCSD(mf, frozen=[[],[]], mo_coeff=None, mo_occ=None):
    from pyscf.cc import uccsd
    mf = _convert_to_uhf(mf)
    return uccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)

def _convert_to_rhf(mf):
    from pyscf import scf
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_rhf(mf)
    return mf

def _convert_to_uhf(mf):
    from pyscf import scf
    if not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
    return mf
