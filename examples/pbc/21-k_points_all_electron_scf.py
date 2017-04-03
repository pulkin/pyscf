#!/usr/bin/env python

'''
Hartree-Fock/DFT with k-points sampling for all-electron calculation
MDF (mixed density fitting) can also be used in k-points sampling.
'''

import numpy
from pyscf.pbc import gto, scf, dft
from pyscf.pbc import df

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = '6-31g',
    gs = [10]*3,
    verbose = 4,
)

nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)

mydf = df.MDF(cell, kpts)
mydf.auxbasis = 'weigend'

kmf = scf.KRHF(cell, kpts)
kmf.with_df = mydf
kmf.kernel()

kmf = dft.KRKS(cell, kpts)
kmf.with_df = mydf
kmf.xc = 'bp86'
kmf.kernel()

