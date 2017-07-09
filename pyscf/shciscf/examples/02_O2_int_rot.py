#!/usr/bin/env python
#
# Author: James Smith <james.smith9113@gmail.com>
#
# All diatomic bond lengths taken from:
# http://cccbdb.nist.gov/diatomicexpbondx.asp

'''
Comparing determining the effect of acitve-active orbital rotations. All output
is deleted after the run to keep the directory neat. Comment out the cleanup
section to view output.
'''

from pyscf import gto, scf, mcscf, dmrgscf
from pyscf.future.shciscf import shci
import os

# O2 molecule Parameters
b = 1.208 # Bond length.
dimer_atom = 'O'
norb = 12 # Number of orbitals included in the active space.
nelec = 12 # Number of electrons included in the active space.
nfrozen = 2 # Number of orbitals that won't be rotated/optimized.


mol = gto.Mole()
mol.build(
verbose = 4,
output = None,
atom = [
    [dimer_atom,(  0.000000,  0.000000, -b/2)],
    [dimer_atom,(  0.000000,  0.000000,  b/2)], ],
basis = {dimer_atom: 'ccpvdz', },
symmetry = True
)

# Create HF molecule
mf = scf.RHF( mol )
mf.conv_tol = 1e-9
mf.scf()


# Calculate energy of the molecules with frozen core.
mch = shci.SHCISCF( mf, norb, nelec )
mch.frozen = nfrozen # Freezes the innermost 2 orbitals.
mch.fcisolver.sweep_iter = [ 0]
mch.fcisolver.sweep_epsilon = [ 1e-3 ] # Loose variational tolerances.
mch.fcisolver.nPTiter = 0
e_noaa = mch.mc1step()[0]


# Calculate energy of the molecule with frozen core and active-active rotations
mch = shci.SHCISCF( mf, norb, nelec )
mch.frozen = nfrozen # Freezes the innermost 2 orbitals.
mch.internal_rotation = True # Do active-active orbital rotations.
mch.fcisolver.sweep_iter = [ 0 ]
mch.fcisolver.sweep_epsilon = [ 1e-3 ]
mch.fcisolver.nPTiter = 0
mch.max_cycle_macro = 20
e_aa = mch.mc1step()[0]

# Comparison Calculations
del_aa = e_aa - e_noaa

print( '\n\nEnergies for %s2 give in E_h.' % dimer_atom )
print( '=====================================' )
print( 'SHCI w/o Act.-Act. Rotations: %6.12f' %e_noaa )
print( 'SHCI w/ Act.-Act. Rotations: %6.12f' %e_aa )
print( 'Change w/ Act.-Act. Rotations: %6.12f' %del_aa )

# File cleanup. Comment out to help debugging.
os.system("rm *.bkp")
os.system("rm *.txt")
os.system("rm shci.e")
os.system("rm *.dat")
os.system("rm FCIDUMP")
