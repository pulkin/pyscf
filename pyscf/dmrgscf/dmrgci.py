#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         Sheng Guo <shengg@princeton.edu>
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
DMRG solver for CASCI and CASSCF.
'''

import ctypes
import os
import sys
import struct
import time
import tempfile
from subprocess import check_call, check_output, STDOUT, CalledProcessError
import numpy
import pyscf.tools
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib import chkfile
from pyscf import mcscf
from pyscf.dmrgscf import dmrg_sym

import pyscf.lib

libunpack = pyscf.lib.load_library('libicmpspt')


try:
    from pyscf.dmrgscf import settings
except ImportError:
    import sys
    sys.stderr.write('''settings.py not found.  Please create %s
''' % os.path.join(os.path.dirname(__file__), 'settings.py'))
    raise ImportError



class DMRGCI(pyscf.lib.StreamObject):
    '''Block program interface and the object to hold Block program input parameters.

    Attributes:
        outputlevel : int
            Noise level for Block program output.
        maxIter : int

        approx_maxIter : int
            To control the DMRG-CASSCF approximate DMRG solver accuracy.
        twodot_to_onedot : int
            When to switch from two-dot algroithm to one-dot algroithm.
        nroots : int

        weights : list of floats
            Use this attribute with "nroots" attribute to set state-average calculation.
        restart : bool
            To control whether to restart a DMRG calculation.
        tol : float
            DMRG convergence tolerence
        maxM : int
            Bond dimension
        scheduleSweeps, scheduleMaxMs, scheduleTols, scheduleNoises : list
            DMRG sweep scheduler.  See also Block documentation
        wfnsym : str or int
            Wave function irrep label or irrep ID
        orbsym : list of int
            irrep IDs of each orbital
        groupname : str
            groupname, orbsym together can control whether to employ symmetry in
            the calculation.  "groupname = None and orbsym = []" requires the
            Block program using C1 symmetry.

    Examples:

    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASCI(mf, 4, 4)
    >>> mc.fcisolver = DMRGCI(mol)
    >>> mc.kernel()
    -74.379770619390698
    '''
    def __init__(self, mol, maxM=None, tol=None, num_thrds=1, memory=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.outputlevel = 2

        self.executable = settings.BLOCKEXE
        self.scratchDirectory = settings.BLOCKSCRATCHDIR
        self.mpiprefix = settings.MPIPREFIX
        self.memory = memory

        self.integralFile = "FCIDUMP"
        self.configFile = "dmrg.conf"
        self.outputFile = "dmrg.out"
        if hasattr(settings, 'BLOCKRUNTIMEDIR'):
            self.runtimeDir = settings.BLOCKRUNTIMEDIR
        else:
            self.runtimeDir = '.'
        self.maxIter = 20
        self.approx_maxIter = 4
        self.twodot_to_onedot = 15
        self.dmrg_switch_tol = 1e-3
        self.nroots = 1
        self.weights = []
        self.wfnsym = 1
        self.extraline = []

        if tol is None:
            self.tol = 1e-8
        else:
            self.tol = tol/10
        if maxM is None:
            self.maxM = 1000
        else:
            self.maxM = maxM

        self.num_thrds= num_thrds
        self.startM =  None
        self.restart = False
        self.nonspinAdapted = False
        self.scheduleSweeps = []
        self.scheduleMaxMs  = []
        self.scheduleTols   = []
        self.scheduleNoises = []
        self.onlywriteIntegral = False
        self.spin = 0

        self.orbsym = []
        if mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None

##################################################
# don't modify the following attributes, if you do not finish part of calculation, which can be reused.

        #DO NOT CHANGE these parameters, unless you know the code in details
        self.twopdm = True #By default, 2rdm is calculated after the calculations of wave function.
        self.block_extra_keyword = [] #For Block advanced user only.
        self.has_fourpdm = False
        self.has_threepdm = False
        self.has_nevpt = False
# This flag _restart is set by the program internally, to control when to make
# Block restart calculation.
        self._restart = False
        self.generate_schedule()

        self._keys = set(self.__dict__.keys())


    def generate_schedule(self):

        if self.startM is None:
            if self.maxM < 200:
                self.startM = 50
            else:
                self.startM = 200
        if len(self.scheduleSweeps) == 0:
            startM = self.startM
            N_sweep = 0
            if self.restart or self._restart :
                Tol = self.tol / 10.0
            else:
                Tol = 1.0e-5
            Noise = Tol
            while startM < int(self.maxM):
                self.scheduleSweeps.append(N_sweep)
                N_sweep += 4
                self.scheduleMaxMs.append(startM)
                startM *= 2
                self.scheduleTols.append(Tol)
                self.scheduleNoises.append(Noise)
            while Tol > float(self.tol):
                self.scheduleSweeps.append(N_sweep)
                N_sweep += 2
                self.scheduleMaxMs.append(self.maxM)
                self.scheduleTols.append(Tol)
                Tol /= 10.0
                self.scheduleNoises.append(5.0e-5)
            self.scheduleSweeps.append(N_sweep)
            N_sweep += 2
            self.scheduleMaxMs.append(self.maxM)
            self.scheduleTols.append(self.tol)
            self.scheduleNoises.append(0.0)
            self.twodot_to_onedot = N_sweep + 2
            self.maxIter = self.twodot_to_onedot + 12
        return self


    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = logger.Logger(self.stdout, verbose)
        log.info('******** Block flags ********')
        log.info('executable = %s', self.executable)
        log.info('Block version %s', block_version(self.executable))
        log.info('BLOCKEXE_COMPRESS_NEVPT = %s', settings.BLOCKEXE_COMPRESS_NEVPT)
        log.info('mpiprefix = %s', self.mpiprefix)
        log.info('scratchDirectory = %s', self.scratchDirectory)
        log.info('integralFile = %s', os.path.join(self.runtimeDir, self.integralFile))
        log.info('configFile = %s', os.path.join(self.runtimeDir, self.configFile))
        log.info('outputFile = %s', os.path.join(self.runtimeDir, self.outputFile))
        log.info('maxIter = %d', self.maxIter)
        log.info('scheduleSweeps = %s', str(self.scheduleSweeps))
        log.info('scheduleMaxMs = %s', str(self.scheduleMaxMs))
        log.info('scheduleTols = %s', str(self.scheduleTols))
        log.info('scheduleNoises = %s', str(self.scheduleNoises))
        log.info('twodot_to_onedot = %d', self.twodot_to_onedot)
        log.info('tol = %g', self.tol)
        log.info('maxM = %d', self.maxM)
        log.info('fullrestart = %s', str(self.restart or self._restart))
        log.info('dmrg switch tol =%s', self.dmrg_switch_tol)
        log.info('wfnsym = %s', self.wfnsym)
        log.info('num_thrds = %d', self.num_thrds)
        log.info('memory = %s', self.memory)
        return self

    def make_rdm1s(self, state, norb, nelec, link_index=None, **kwargs):
# Ref: IJQC, 109, 3552 Eq (3)
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = (nelec-self.spin) // 2
            neleca = nelec - nelecb
        else :
            neleca, nelecb = nelec
        dm1, dm2 = DMRGCI.make_rdm12(self, state, norb, nelec, link_index, **kwargs)
        dm1n = (2-(neleca+nelecb)/2.) * dm1 - numpy.einsum('pkkq->pq', dm2)
        dm1n *= 1./(neleca-nelecb+1)
        dm1a, dm1b = (dm1+dm1n)*.5, (dm1-dm1n)*.5
        return dm1a, dm1b

    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
# Avoid calling self.make_rdm12 because it may be overloaded
        return DMRGCI.make_rdm12(self, state, norb, nelec, link_index, **kwargs)[0]

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        file2pdm = "spatial_twopdm.%d.%d.txt" %(state, state)
        with open(os.path.join(self.scratchDirectory, "node0", file2pdm), "r") as f:
            norb_read = int(f.readline().split()[0])
            assert(norb_read == norb)

            for line in f:
                linesp = line.split()
                i, k, l, j = [int(x) for x in linesp[:4]]
                twopdm[i,j,k,l] = 2.0 * float(linesp[4])

        onepdm = numpy.einsum('ikjj->ik', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm

    def trans_rdm1s(self, statebra, stateket, norb, nelec, link_index=None, **kwargs):
# Ref: IJQC, 109, 3552 Eq (3)
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = (nelec-self.spin) // 2
            neleca = nelec - nelecb
        else :
            neleca, nelecb = nelec
        dm1, dm2 = DMRGCI.trans_rdm12(self, statebra, stateket, norb, nelec, link_index, **kwargs)
        dm1n = (2-(neleca+nelecb)/2.) * dm1 - numpy.einsum('pkkq->pq', dm2)
        dm1n *= 1./(neleca-nelecb+1)
        dm1a, dm1b = (dm1+dm1n)*.5, (dm1-dm1n)*.5
        return dm1a, dm1b

    def trans_rdm1(self, statebra, stateket, norb, nelec, link_index=None, **kwargs):
        return DMRGCI.trans_rdm12(self, statebra, stateket, norb, nelec, link_index, **kwargs)[0]

    def trans_rdm12(self, statebra, stateket, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        writeDMRGConfFile(self, nelec, True, with_2pdm=False,
                          extraline=['restart_tran_twopdm',
                                     'specificpdm %d %d' % (statebra, stateket)])
        executeBLOCK(self)

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        file2pdm = "spatial_twopdm.%d.%d.txt" %(statebra, stateket)
        with open(os.path.join(self.scratchDirectory, "node0", file2pdm), "r") as f:
            norb_read = int(f.readline().split()[0])
            assert(norb_read == norb)

            for line in f:
                linesp = line.split()
                i, k, l, j = [int(x) for x in linesp[:4]]
                twopdm[i,j,k,l] = 2.0 * float(linesp[4])

        onepdm = numpy.einsum('ikjj->ik', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm

    def make_rdm123(self, state, norb, nelec, link_index=None, **kwargs):
        if self.has_threepdm == False:
            writeDMRGConfFile(self, nelec, True,
                              with_2pdm=False, extraline=['restart_threepdm'])
            if self.verbose >= logger.DEBUG1:
                inFile = os.path.join(self.runtimeDir, self.configFile)
                logger.debug1(self, 'Block Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeBLOCK(self)
            if self.verbose >= logger.DEBUG1:
                outFile = os.path.join(self.runtimeDir, self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_threepdm = True

        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        threepdm = numpy.zeros( (norb, norb, norb, norb, norb, norb) )
        file3pdm = "spatial_threepdm.%d.%d.txt" %(state, state)
        with open(os.path.join(self.scratchDirectory, "node0", file3pdm), "r") as f:
            norb_read = int(f.readline().split()[0])
            assert(norb_read == norb)

            for line in f:
                linesp = line.split()
                i, j, k, l, m, n = [int(x) for x in linesp[:6]]
                threepdm[i,j,k,l,m,n] = float(linesp[6])

        twopdm = numpy.einsum('ijkklm->ijlm',threepdm)
        twopdm /= (nelectrons-2)
        onepdm = numpy.einsum('ijjk->ik', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm, threepdm

    def _make_dm123(self, state, norb, nelec, link_index=None, **kwargs):
        r'''Note this function does NOT compute the standard density matrix.
        The density matrices are reordered to match the the fci.rdm.make_dm123
        function (used by NEVPT code).
        The returned "2pdm" is :math:`\langle p^\dagger q r^\dagger s\rangle`;
        The returned "3pdm" is :math:`\langle p^\dagger q r^\dagger s t^\dagger u\rangle`.
        '''
        onepdm, twopdm, threepdm = self.make_rdm123(state, norb, nelec, None, **kwargs)
        threepdm = numpy.einsum('mkijln->ijklmn',threepdm).copy()
        threepdm += numpy.einsum('jk,lm,in->ijklmn',numpy.identity(norb),numpy.identity(norb),onepdm)
        threepdm += numpy.einsum('jk,miln->ijklmn',numpy.identity(norb),twopdm)
        threepdm += numpy.einsum('lm,kijn->ijklmn',numpy.identity(norb),twopdm)
        threepdm += numpy.einsum('jm,kinl->ijklmn',numpy.identity(norb),twopdm)

        twopdm =(numpy.einsum('iklj->ijkl',twopdm)
               + numpy.einsum('il,jk->ijkl',onepdm,numpy.identity(norb)))

        return onepdm, twopdm, threepdm

    def make_rdm3(self, state, norb, nelec, dt=numpy.dtype('Float64'), filetype = "binary", link_index=None, **kwargs):
        import os

        if self.has_threepdm == False:
            self.twopdm = False
            self.extraline.append('threepdm\n')

            writeDMRGConfFile(self, nelec, False)
            if self.verbose >= logger.DEBUG1:
                inFile = self.configFile
                #inFile = os.path.join(self.scratchDirectory,self.configFile)
                logger.debug1(self, 'Block Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeBLOCK(self)
            if self.verbose >= logger.DEBUG1:
                outFile = self.outputFile
                #outFile = os.path.join(self.scratchDirectory,self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_threepdm = True
            self.extraline.pop()

        # Remove everything
        os.system("rm -f %s/node0/*twopdm*"%(self.scratchDirectory))
        os.system("rm -f %s/node0/*tmp"%(self.scratchDirectory))

        # The binary files coming from STACKBLOCK and BLOCK are different
        # - STACKBLOCK uses the 6-fold symmetry, this must unpacked using "libunpack.unpackE3" (see lib/icmpspt/icmpspt.c)
        # - BLOCK just writes a list of all values, this is directly read by "unpackE3_BLOCK"
        if (filetype == "binary") :
            fname = os.path.join(self.scratchDirectory,"node0", "spatial_threepdm.%d.%d.bin" %(state, state))
            if 'stackblock' in settings.BLOCKEXE:
              print 'Reading binary 3RDM from STACKBLOCK'
              fnameout = os.path.join(self.scratchDirectory,"node0", "spatial_threepdm.%d.%d.bin.unpack" %(state, state))
              libunpack.unpackE3(ctypes.c_char_p(fname), ctypes.c_char_p(fnameout), ctypes.c_int(norb))
              E3 = numpy.fromfile(fnameout, dtype=numpy.dtype('Float64'))
              E3 = numpy.reshape(E3, (norb, norb, norb, norb, norb, norb), order='F')
            else:
              print 'Reading binary 3RDM from BLOCK'
              E3 = DMRGCI.unpackE3_BLOCK(self,fname,norb)


        # Both text files are the same: a list of "a,b,c,d,e,f,E3[a,b,c,f,e,d]"
        else:
            print 'Reading text-file 3RDM'
            fname = os.path.join(self.scratchDirectory,"node0", "spatial_threepdm.%d.%d.txt" %(state, state))
            f = open(fname, 'r')
            lines = f.readlines()
            E3 = numpy.zeros(shape=(norb, norb, norb, norb, norb, norb), dtype=dt, order='F')
            for line in lines[1:]:
                linesp = line.split()
                if (len(linesp) != 7) :
                    continue
                a, b, c, d, e, f, integral = int(linesp[0]), int(linesp[1]), int(linesp[2]), int(linesp[3]), int(linesp[4]), int(linesp[5]), float(linesp[6])
                E3[a,b,c, f,e,d] = integral
                E3[a,c,b, f,d,e] = integral
                E3[b,a,c, e,f,d] = integral
                E3[b,c,a, e,d,f] = integral
                E3[c,a,b, d,f,e] = integral
                E3[c,b,a, d,e,f] = integral

        #for i1 in range(norb):
        #  for i2 in range(norb):
        #    for i3 in range(norb):
        #      for i4 in range(norb):
        #        for i5 in range(norb):
        #          for i6 in range(norb):
        #            if (abs(E3[i1,i2,i3,i4,i5,i6])>0.0001):
        #              print "[BM] ({:5} {:5} {:5}|{:5} {:5} {:5}) {:20.4f}".format(i1,i2,i3,i4,i5,i6,E3[i1,i2,i3,i4,i5,i6])
        print ''
        return E3

    def make_rdm4(self, state, norb, nelec, dt=numpy.dtype('Float64'), filetype = "binary", link_index=None, **kwargs):
        import os

        if self.has_fourpdm == False:
            self.twopdm = False
            self.threepdm = False
            self.extraline.append('threepdm\n')
            self.extraline.append('fourpdm\n')

            writeDMRGConfFile(self, nelec, False)
            if self.verbose >= logger.DEBUG1:
                inFile = self.configFile
                #inFile = os.path.join(self.scratchDirectory,self.configFile)
                logger.debug1(self, 'Block Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeBLOCK(self)
            if self.verbose >= logger.DEBUG1:
                outFile = self.outputFile
                #outFile = os.path.join(self.scratchDirectory,self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_fourpdm = True
            self.has_threepdm = True
            self.extraline.pop()

        # Remove everything
        os.system("rm -f %s/node0/*twopdm*"%(self.scratchDirectory))
        os.system("rm -f %s/node0/*tmp"%(self.scratchDirectory))

        # The binary files coming from STACKBLOCK and BLOCK are different:
        # - STACKBLOCK does not have 4RDM
        #   if it had, it's probably gonna come in a 8-fold symmetry, which must unpacked using "libunpack.unpackE4" (see lib/icmpspt/icmpspt.c)
        # - BLOCK just writes a list of all values, this is directly read by "unpackE4_BLOCK"
        if (filetype == "binary") :
            fname = os.path.join(self.scratchDirectory,"node0", "spatial_fourpdm.%d.%d.bin" %(state, state))
            if 'stackblock' in settings.BLOCKEXE:
              print 'Reading binary 4RDM from STACKBLOCK'
              fnameout = os.path.join(self.scratchDirectory,"node0", "spatial_fourpdm.%d.%d.bin.unpack" %(state, state))
              libunpack.unpackE4(ctypes.c_char_p(fname), ctypes.c_char_p(fnameout), ctypes.c_int(norb))
              E4 = numpy.fromfile(fnameout, dtype=numpy.dtype('Float64'))
              E4 = numpy.reshape(E4, (norb, norb, norb, norb, norb, norb, norb, norb), order='F')
            else:
              print 'Reading binary 4RDM from BLOCK'
              E4 = DMRGCI.unpackE4_BLOCK(self,fname,norb)

        # Both text files are the same: a list of "a,b,c,d,e,f,g,h,E4[a,b,c,d,h,g,f,e]"
        else:
            print 'Reading text-file 4RDM'
            fname = os.path.join(self.scratchDirectory,"node0", "spatial_fourpdm.%d.%d.txt" %(state, state))
            f = open(fname, 'r')
            lines = f.readlines()
            E4 = numpy.zeros(shape=(norb, norb, norb, norb, norb, norb, norb, norb), dtype=dt, order='F')
            for line in lines[1:]:
              linesp = line.split()
              if (len(linesp) != 9) :
                  continue
              a, b, c, d, e, f, g, h, integral = int(linesp[0]), int(linesp[1]), int(linesp[2]), int(linesp[3]), int(linesp[4]), int(linesp[5]), int(linesp[6]), int(linesp[7]), float(linesp[8])
              up_indexes=[a,b,c,d]
              dn_indexes=[h,g,f,e]
              for i in range(4):
                for j in range(4):
                  if (i==j):
                    continue
                  for k in range(4):
                    if ((i==k)or(j==k)):
                      continue
                    for l in range(4):
                      if ((i==l)or(j==l)or(k==l)):
                        continue
                      E4[up_indexes[i],up_indexes[j],up_indexes[k],up_indexes[l],\
                         dn_indexes[i],dn_indexes[j],dn_indexes[k],dn_indexes[l]] = integral
        #for i1 in range(norb):
        #  for i2 in range(norb):
        #    for i3 in range(norb):
        #      for i4 in range(norb):
        #        for i5 in range(norb):
        #          for i6 in range(norb):
        #            for i7 in range(norb):
        #              for i8 in range(norb):
        #                if abs(E4[i1,i2,i3,i4,i5,i6,i7,i8]>0.0001):
        #                  print "[BM] ({:5} {:5} {:5} {:5}|{:5} {:5} {:5} {:5}) {:20.4f}".format(i1,i2,i3,i4,i5,i6,i7,i8,E4[i1,i2,i3,i4,i5,i6,i7,i8])
        print ''
        return E4

    def unpackE3_BLOCK(self,fname,norb):
        E3=numpy.zeros((norb,norb,norb,norb,norb,norb), order='F')
        fil=open(fname,"rb")
        fil.seek(93)
        for a in range(norb):
          for b in range(norb):
            for c in range(norb):
              for d in range(norb):
                for e in range(norb):
                  for f in range(norb):
                    (value,)=struct.unpack('d',fil.read(8))
                    E3[a,b,c,  f,e,d]=value
        fil.close()
        return E3

    def unpackE4_BLOCK(self,fname,norb):
        E4=numpy.zeros((norb,norb,norb,norb,norb,norb,norb,norb), order='F')
        fil=open(fname,"rb")
        fil.seek(109)
        for a in range(norb):
          for b in range(norb):
            for c in range(norb):
              for d in range(norb):
                for e in range(norb):
                  for f in range(norb):
                    for g in range(norb):
                      for h in range(norb):
                        (value,)=struct.unpack('d',fil.read(8))
                        E4[a,b,c,d,  h,g,f,e]=value
        fil.close()
        return E4

    def clearSchedule(self):
        self.scheduleSweeps = []
        self.scheduleMaxMs = []
        self.scheduleTols = []
        self.scheduleNoises = []

    def nevpt_intermediate(self, tag, norb, nelec, state, **kwargs):

        if self.has_nevpt == False:
            writeDMRGConfFile(self, nelec, True,
                              with_2pdm=False, extraline=['restart_nevpt2_npdm'])
            if self.verbose >= logger.DEBUG1:
                inFile = os.path.join(self.runtimeDir, self.configFile)
                logger.debug1(self, 'Block Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeBLOCK(self)
            if self.verbose >= logger.DEBUG1:
                outFile = os.path.join(self.runtimeDir, self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_nevpt = True

        a16 = numpy.zeros( (norb, norb, norb, norb, norb, norb) )
        filename = "%s_matrix.%d.%d.txt" % (tag, state, state)
        with open(os.path.join(self.scratchDirectory, "node0", filename), "r") as f:
            norb_read = int(f.readline().split()[0])
            assert(norb_read == norb)

            for line in f:
                linesp = line.split()
                i, j, k, l, m, n = [int(x) for x in linesp[:6]]
                a16[i,j,k,l,m,n] = float(linesp[6])

        return a16

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)
        if fciRestart is None:
            fciRestart = self.restart or self._restart

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        writeDMRGConfFile(self, nelec, fciRestart)
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'Block Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        if self.onlywriteIntegral:
            logger.info(self, 'Only write integral')
            try:
                calc_e = readEnergy(self)
            except IOError:
                if self.nroots == 1:
                    calc_e = 0.0
                else :
                    calc_e = [0.0] * self.nroots
            return calc_e, roots

        executeBLOCK(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)

        return calc_e, roots

    def approx_kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
        fciRestart = True

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        writeDMRGConfFile(self, nelec, fciRestart, self.approx_maxIter)
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'Block Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        executeBLOCK(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)

        if self.nroots==1:
            roots = 0
        else:
            roots = range(self.nroots)
        return calc_e, roots

    def restart_scheduler_(self):
        def callback(envs):
            if (envs['norm_gorb'] < self.dmrg_switch_tol or
                ('norm_ddm' in envs and envs['norm_ddm'] < self.dmrg_switch_tol*10)):
                self._restart = True
            else :
                self._restart = False
        return callback

    def spin_square(self, civec, norb, nelec):
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else :
            neleca, nelecb = nelec
        s = (neleca - nelecb) * .5
        ss = s * (s+1)
        if isinstance(civec, int):
            return ss, s*2+1
        else:
            return [ss]*len(civec), [s*2+1]*len(civec)


def make_schedule(sweeps, Ms, tols, noises, twodot_to_onedot):
    if len(sweeps) == len(Ms) == len(tols) == len(noises):
        schedule = ['schedule']
        for i, s in enumerate(sweeps):
            schedule.append('%d %6d  %8.4e  %8.4e' % (s, Ms[i], tols[i], noises[i]))
        schedule.append('end')
        if (twodot_to_onedot != 0):
            schedule.append('twodot_to_onedot %i'%twodot_to_onedot)
        return '\n'.join(schedule)
    else:

        return 'schedule default\nmaxM %s'%Ms[-1]

def writeDMRGConfFile(DMRGCI, nelec, Restart,
                      maxIter=None, with_2pdm=True, extraline=[]):
    confFile = os.path.join(DMRGCI.runtimeDir, DMRGCI.configFile)

    f = open(confFile, 'w')

    if isinstance(nelec, (int, numpy.integer)):
        nelecb = (nelec-DMRGCI.spin) // 2
        neleca = nelec - nelecb
    else :
        neleca, nelecb = nelec
    f.write('nelec %i\n'%(neleca+nelecb))
    f.write('spin %i\n' %(neleca-nelecb))
    if DMRGCI.groupname is not None:
        if isinstance(DMRGCI.wfnsym, str):
            wfnsym = dmrg_sym.irrep_name2id(DMRGCI.groupname, DMRGCI.wfnsym)
        else:
            gpname = dmrg_sym.d2h_subgroup(DMRGCI.groupname)
            assert(DMRGCI.wfnsym in dmrg_sym.IRREP_MAP[gpname])
            wfnsym = DMRGCI.wfnsym
        f.write('irrep %i\n' % wfnsym)

    if (not Restart):
        #f.write('schedule\n')
        #f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-5, 10.0))
        #f.write('1 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-5, 1e-4))
        #f.write('10 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-6, 1e-5))
        #f.write('16 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol/10.0, 0e-6))
        #f.write('end\n')
        schedule = make_schedule(DMRGCI.scheduleSweeps,
                                 DMRGCI.scheduleMaxMs,
                                 DMRGCI.scheduleTols,
                                 DMRGCI.scheduleNoises,
                                 DMRGCI.twodot_to_onedot)
        f.write('%s\n' % schedule)
    else :
        f.write('schedule\n')
        #if approx == True :
        #    f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol*10.0, 0e-6))
        #else :
        #    f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol, 0e-6))
        f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol/10, 0e-6))
        f.write('end\n')
        f.write('fullrestart\n')
        f.write('onedot \n')

    if DMRGCI.groupname is not None:
        f.write('sym %s\n' % dmrg_sym.d2h_subgroup(DMRGCI.groupname).lower())
    f.write('orbitals %s\n' % DMRGCI.integralFile)
    if maxIter is None:
        maxIter = DMRGCI.maxIter
    f.write('maxiter %i\n'%maxIter)
    f.write('sweep_tol %8.4e\n'%DMRGCI.tol)

    f.write('outputlevel %s\n'%DMRGCI.outputlevel)
    f.write('hf_occ integral\n')
    if(with_2pdm and DMRGCI.twopdm):
        f.write('twopdm\n')
    if(DMRGCI.nonspinAdapted):
        f.write('nonspinAdapted\n')
    if(DMRGCI.scratchDirectory):
        f.write('prefix  %s\n'%DMRGCI.scratchDirectory)
    if (DMRGCI.nroots !=1):
        f.write('nroots %d\n'%DMRGCI.nroots)
        if (DMRGCI.weights==[]):
            DMRGCI.weights= [1.0/DMRGCI.nroots]* DMRGCI.nroots
        f.write('weights ')
        for weight in DMRGCI.weights:
            f.write('%f '%weight)
        f.write('\n')

    block_extra_keyword = DMRGCI.extraline + DMRGCI.block_extra_keyword + extraline
    if block_version(DMRGCI.executable).startswith('1.1'):
        for line in block_extra_keyword:
            if not ('num_thrds' in line or 'memory' in line):
                f.write('%s\n'%line)
    else:
        if DMRGCI.memory is not None:
            f.write('memory, %i, g\n'%(DMRGCI.memory))
        if DMRGCI.num_thrds > 1:
            f.write('num_thrds %d\n'%DMRGCI.num_thrds)
        for line in block_extra_keyword:
            f.write('%s\n'%line)
    f.close()
    #no reorder
    #f.write('noreorder\n')

def writeIntegralFile(DMRGCI, h1eff, eri_cas, ncas, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else :
        neleca, nelecb = nelec
    integralFile = os.path.join(DMRGCI.runtimeDir, DMRGCI.integralFile)
    if DMRGCI.groupname is not None and DMRGCI.orbsym is not []:
        orbsym = numpy.asarray(dmrg_sym.convert_orbsym(DMRGCI.groupname, DMRGCI.orbsym))
        pair_irrep = (orbsym.reshape(-1,1) ^ orbsym)[numpy.tril_indices(ncas)]
        sym_forbid = pair_irrep.reshape(-1,1) != pair_irrep.ravel()
        eri_cas = pyscf.ao2mo.restore(4, eri_cas, ncas)
        eri_cas[sym_forbid] = 0
        eri_cas = pyscf.ao2mo.restore(8, eri_cas, ncas)
    else:
        orbsym = []
        eri_cas = pyscf.ao2mo.restore(8, eri_cas, ncas)
    if not os.path.exists(DMRGCI.scratchDirectory):
        os.makedirs(DMRGCI.scratchDirectory)
    if not os.path.exists(DMRGCI.runtimeDir):
        os.makedirs(DMRGCI.runtimeDir)

    pyscf.tools.fcidump.from_integrals(integralFile, h1eff, eri_cas, ncas,
                                       neleca+nelecb, ecore, ms=abs(neleca-nelecb),
                                       orbsym=orbsym)


def executeBLOCK(DMRGCI):

    inFile  = DMRGCI.configFile
    outFile = DMRGCI.outputFile
    try:
        cmd = ' '.join((DMRGCI.mpiprefix, DMRGCI.executable, inFile))
        cmd = "%s > %s 2>&1" % (cmd, outFile)
        check_call(cmd, cwd=DMRGCI.runtimeDir, shell=True)
    except CalledProcessError as err:
        logger.error(DMRGCI, cmd)
        DMRGCI.stdout.write(check_output(['tail', '-100', outFile]))
        raise err

def readEnergy(DMRGCI):
    file1 = open(os.path.join(DMRGCI.scratchDirectory, "node0", "dmrg.e"), "rb")
    format = ['d']*DMRGCI.nroots
    format = ''.join(format)
    calc_e = struct.unpack(format, file1.read())
    file1.close()
    if DMRGCI.nroots == 1:
        return calc_e[0]
    else:
        return numpy.asarray(calc_e)


def DMRGSCF(mf, norb, nelec, maxM=1000, tol=1.e-8, *args, **kwargs):
    '''Shortcut function to setup CASSCF using the DMRG solver.  The DMRG
    solver is properly initialized in this function so that the 1-step
    algorithm can applied with DMRG-CASSCF.

    Examples:

    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = DMRGSCF(mf, 4, 4)
    >>> mc.kernel()
    -74.414908818611522
    '''
    if (hasattr(mf,'with_df')):
      mc = mcscf.DFCASSCF(mf, norb, nelec, *args, **kwargs)
    else:
      mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    mc.fcisolver = DMRGCI(mf.mol, maxM, tol=tol)
    mc.callback = mc.fcisolver.restart_scheduler_()
    if mc.chkfile == mc._scf._chkfile.name:
        # Do not delete chkfile after mcscf
        mc.chkfile = tempfile.mktemp(dir=settings.BLOCKSCRATCHDIR)
        if not os.path.exists(settings.BLOCKSCRATCHDIR):
            os.makedirs(settings.BLOCKSCRATCHDIR)
    return mc


def dryrun(mc, mo_coeff=None):
    '''Generate FCIDUMP and dmrg config file'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    mc.fcisolver.onlywriteIntegral, bak = True, mc.fcisolver.onlywriteIntegral
    mc.casci(mo_coeff)
    mc.fcisolver.onlywriteIntegral = bak

def block_version(blockexe):
    try:
        msg = check_output([blockexe, '-v'], stderr=STDOUT)
        version = '1.1.0'
        for line in msg.split('\n'):
            if line.startswith('Block '):
                version = line.split()[1]
                break
        return version
    except CalledProcessError:
        f1 = tempfile.NamedTemporaryFile()
        f1.write('memory 1 m\n')
        f1.flush()
        try:
            msg = check_output([blockexe, f1.name], stderr=STDOUT)
        except CalledProcessError as err:
            if 'Unrecognized option :: memory' in err.output:
                version = '1.1.1'
            elif 'need to specify hf_occ' in err.output:
                version = '1.5'
            else:
                sys.stderr.write(err.output)
                raise err
        f1.close()
        return version


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    settings.MPIPREFIX =''
    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-dmrgci',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True
    )
    m = scf.RHF(mol)
    m.scf()

    mc = DMRGSCF(m, 4, 4)
    mc.fcisolver.tol = 1e-9
    emc_1 = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver = DMRGCI(mol)
    mc.fcisolver.scheduleSweeps = []
    emc_0 = mc.casci()[0]

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-casscf',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    emc_1ref = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    emc_0ref = mc.casci()[0]

    print('DMRGCI  = %.15g CASCI  = %.15g' % (emc_0, emc_0ref))
    print('DMRGSCF = %.15g CASSCF = %.15g' % (emc_1, emc_1ref))

