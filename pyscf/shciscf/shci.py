#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         James Smith <james.smith9113@gmail.com>
#

'''
SHCI solver for CASCI and CASSCF.
'''
import ctypes
import os
import sys
import struct
import time
import tempfile
from subprocess import check_call, CalledProcessError
import numpy
import pyscf.tools
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib import chkfile
from pyscf.lib import load_library
from pyscf import mcscf
ndpointer = numpy.ctypeslib.ndpointer

# Settings
try:
   from pyscf.future.shciscf import settings
except ImportError:
    import sys
    sys.stderr.write('''settings.py not found.  Please create %s
''' % os.path.join(os.path.dirname(__file__), 'settings.py'))
    raise ImportError

# Libraries
libE3unpack = load_library('libicmpspt')
# TODO: Organize this better.
shciLib = load_library('libshciscf')

transformDinfh = shciLib.transformDinfh
transformDinfh.restyp = None
transformDinfh.argtypes = [ctypes.c_int, ndpointer(ctypes.c_int32),
                      ndpointer(ctypes.c_int32), ndpointer(ctypes.c_double),
                      ndpointer(ctypes.c_double),ndpointer(ctypes.c_double)]

transformRDMDinfh = shciLib.transformRDMDinfh
transformRDMDinfh.restyp = None
transformRDMDinfh.argtypes = [ctypes.c_int, ndpointer(ctypes.c_int32),
                      ndpointer(ctypes.c_int32), ndpointer(ctypes.c_double),
                      ndpointer(ctypes.c_double),ndpointer(ctypes.c_double)]

writeIntNoSymm = shciLib.writeIntNoSymm
writeIntNoSymm.argtypes = [ctypes.c_int, ndpointer(ctypes.c_double),ndpointer(ctypes.c_double),
                           ctypes.c_double, ctypes.c_int, ndpointer(ctypes.c_int)]

fcidumpFromIntegral = shciLib.fcidumpFromIntegral
fcidumpFromIntegral.restype = None
fcidumpFromIntegral.argtypes = [ctypes.c_char_p,
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                ctypes.c_double,
                                ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                                ctypes.c_size_t,
                                ctypes.c_double
]

r2RDM = shciLib.r2RDM
r2RDM.restype = None
r2RDM.argtypes = [ ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
					ctypes.c_size_t,
					ctypes.c_char_p]


class SHCI(pyscf.lib.StreamObject):
    r'''SHCI program interface and object to hold SHCI program input parameters.

    Attributes:
        davidsonTol: double
        epsilon2: double
        epsilon2Large: double
        sampleN: int
        epsilon1: vector<double>
        onlyperturbative: bool
        restart: bool
        fullrestart: bool
        dE: double
        eps: double
        prefix: str
        stochastic: bool
        nblocks: int
        excitation: int
        nvirt: int
        singleList: bool
        io: bool
        nroots: int
        nPTiter: int
        DoRDM: bool
        sweep_iter: [int]
        sweep_epsilon: [float]
        groupname : str
            groupname, orbsym together can control whether to employ symmetry in
            the calculation.  "groupname = None and orbsym = []" requires the
            SHCI program using C1 symmetry.
        useExtraSymm : False
            if the symmetry of the molecule is Dooh or Cooh, then this keyword uses
            complex orbitals to make full use of this symmetry

    Examples:


    '''

    def __init__(self, mol=None, maxM=None, tol=None, num_thrds=1, memory=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2

        self.executable = settings.SHCIEXE
        #self.QDPTexecutable = settings.SHCIQDPTEXE
        self.scratchDirectory = settings.SHCISCRATCHDIR
        self.mpiprefix = settings.MPIPREFIX
        self.memory = memory

        self.integralFile = "FCIDUMP"
        self.configFile = "input.dat"
        self.outputFile = "output.dat"
        if hasattr(settings, 'SHCIRUNTIMEDIR'):
            self.runtimeDir = settings.SHCIRUNTIMEDIR
        else:
            self.runtimeDir = '.'

        self.spin = None
        # TODO: Organize into pyscf and SHCI parameters
        # Standard SHCI Input parameters
        self.davidsonTol = 5.e-5
        self.epsilon2 = 1.e-7
        self.epsilon2Large = 1000
        self.sampleN = 200
        self.epsilon1 = None
        self.onlyperturbative = False
        self.fullrestart = False
        self.dE = 1.e-8
        self.eps = None
        self.prefix = "."
        self.stochastic = True
        self.nblocks = 1
        self.excitation = 1000
        self.nvirt = 1e6
        self.singleList = True
        self.io = True
        self.nroots = 1
        self.nPTiter = 0
        self.DoRDM = True
        self.sweep_iter =[]
        self.sweep_epsilon = []
        self.maxIter = 6
        self.restart = False
        self.orbsym = []
        self.onlywriteIntegral = False
        self.orbsym = None
        if mol is None:
           self.groupname = None
        else:
           if mol.symmetry:
              self.groupname = mol.groupname
           else:
              self.groupname = None
        self._restart = False
        self.returnInt = False
        self._keys = set(self.__dict__.keys())
        self.irrep_nelec = None
        self.useExtraSymm = False

    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = logger.Logger(self.stdout, verbose)
        log.info( '******** SHCI flags ********' )
        log.info( 'executable = %s', self.executable)
        log.info( 'SHCIEXE_COMPRESS_NEVPT = %s', settings.SHCIEXE_COMPRESS_NEVPT )
        log.info( 'mpiprefix = %s', self.mpiprefix )
        log.info( 'scratchDirectory = %s', self.scratchDirectory )
        log.info( 'integralFile = %s', os.path.join(self.runtimeDir, self.integralFile) )
        log.info( 'configFile = %s', os.path.join(self.runtimeDir, self.configFile) )
        log.info( 'outputFile = %s', os.path.join(self.runtimeDir, self.outputFile) )
        log.info( 'maxIter = %d', self.maxIter )
        log.info( 'nPTiter = %i', self.nPTiter )
        log.info( 'Stochastic = %r', self.stochastic )
        log.info( 'num_thrds = %d', self.num_thrds )
        log.info( 'memory = %s', self.memory )
        return self

    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
# Avoid calling self.make_rdm12 because it may be overloaded
        return SHCI.make_rdm12(self, state, norb, nelec, link_index, **kwargs)[0]

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        file2pdm = "%s/spatialRDM.%d.%d.txt"%(self.prefix,state, state)
        r2RDM( twopdm, norb, file2pdm )

        #if (SHCI.groupname == 'Dooh' or SHCI.groupname == 'Cooh') and SHCI.useExtraSymm:
        if (self.groupname == 'Dooh' or self.groupname == 'Cooh') and self.useExtraSymm:
           nRows, rowIndex, rowCoeffs = DinfhtoD2h(self, norb, nelec)
           twopdmcopy = 1.*twopdm
           twopdm = 0.*twopdm
           transformRDMDinfh(norb, numpy.ascontiguousarray(nRows, numpy.int32),
                             numpy.ascontiguousarray(rowIndex, numpy.int32),
                             numpy.ascontiguousarray(rowCoeffs, numpy.float64),
                             numpy.ascontiguousarray(twopdmcopy, numpy.float64),
                             numpy.ascontiguousarray(twopdm, numpy.float64))
           twopdmcopy = None



        onepdm = numpy.einsum('ikjj->ik', twopdm)
        onepdm /= (nelectrons-1)

        return onepdm, twopdm

    def trans_rdm1(self, statebra, stateket, norb, nelec, link_index=None, **kwargs):
        return self.trans_rdm12(statebra, stateket, norb, nelec, link_index, **kwargs)[0]

    def trans_rdm12(self, statebra, stateket, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        writeSHCIConfFile(self, nelec, True)
        executeSHCI(self)

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        file2pdm = "spatialRDM.%d.%d.txt" % ( root, root )
        r2RDM( twopdm, norb, file2pdm )

        onepdm = numpy.einsum('ikjj->ik', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm

    def make_rdm123(self, state, norb, nelec, link_index=None, **kwargs):
        if self.has_threepdm == False:
            writeSHCIConfFile(self, nelec, True )
            if self.verbose >= logger.DEBUG1:
                inFile = os.path.join(self.runtimeDir, self.configFile)
                logger.debug1(self, 'SHCI Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeSHCI(self)
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

            writeSHCIConfFile(self, nelec, True)
            if self.verbose >= logger.DEBUG1:
                inFile = self.configFile
                #inFile = os.path.join(self.scratchDirectory,self.configFile)
                logger.debug1(self, 'SHCI Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeSHCI(self)
            if self.verbose >= logger.DEBUG1:
                outFile = self.outputFile
                #outFile = os.path.join(self.scratchDirectory,self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_threepdm = True
            self.extraline.pop()

        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        if (filetype == "binary") :
            fname = os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_threepdm.%d.%d.bin" %(state, state))
            fnameout = os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_threepdm.%d.%d.bin.unpack" %(state, state))
            libE3unpack.unpackE3(ctypes.c_char_p(fname), ctypes.c_char_p(fnameout), ctypes.c_int(norb))

            E3 = numpy.fromfile(fnameout, dtype=numpy.dtype('Float64'))
            E3 = numpy.reshape(E3, (norb, norb, norb, norb, norb, norb), order='F')

        else :
            fname = os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_threepdm.%d.%d.txt" %(state, state))
            f = open(fname, 'r')
            lines = f.readlines()
            E3 = numpy.zeros(shape=(norb, norb, norb, norb, norb, norb), dtype=dt)
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

        return E3

    def clearSchedule(self):
        self.scheduleSweeps = []
        self.scheduleMaxMs = []
        self.scheduleTols = []
        self.scheduleNoises = []


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
        writeSHCIConfFile(self, nelec, fciRestart)
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'SHCI Input conf')
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
        if self.returnInt:
            return h1e, eri

        executeSHCI(self)
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
        writeSHCIConfFile(self, nelec, fciRestart )
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'SHCI Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        executeSHCI(self)
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
            if (envs['norm_gorb'] < self.shci_switch_tol or
                ('norm_ddm' in envs and envs['norm_ddm'] < self.shci_switch_tol*10)):
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


def print1Int(h1,name):
 with open('%s.X'%(name), 'w') as fout:
    fout.write('%d\n'%h1[0].shape[0])
    for i in range(h1[0].shape[0]):
        for j in range(h1[0].shape[0]):
            fout.write('%16.10g %4d %4d\n'%(h1[0,i,j], i+1, j+1))

 with open('%s.Y'%(name), 'w') as fout:
    fout.write('%d\n'%h1[1].shape[0])
    for i in range(h1[1].shape[0]):
        for j in range(h1[1].shape[0]):
            fout.write('%16.10g %4d %4d\n'%(h1[1,i,j], i+1, j+1))

 with open('%s.Z'%(name), 'w') as fout:
    fout.write('%d\n'%h1[2].shape[0])
    for i in range(h1[2].shape[0]):
        for j in range(h1[2].shape[0]):
            fout.write('%16.10g %4d %4d\n'%(h1[2,i,j], i+1, j+1))

 with open('%sZ'%(name), 'w') as fout:
    fout.write('%d\n'%h1[2].shape[0])
    for i in range(h1[2].shape[0]):
        for j in range(h1[2].shape[0]):
            fout.write('%16.10g %4d %4d\n'%(h1[2,i,j], i+1, j+1))


def make_sched( SHCI ):

    nIter = len( SHCI.sweep_iter )
    # Check that the number of different epsilons match the number of iter steps.
    assert( nIter == len( SHCI.sweep_epsilon ) )

    if( nIter == 0 ):
        SHCI.sweep_iter = [0]
        SHCI.sweep_epsilon = [1.e-3]

    schedStr = 'schedule '
    for it, eps in zip( SHCI.sweep_iter, SHCI.sweep_epsilon ):
        schedStr += '\n' + str( it ) + '\t' + str( eps )

    schedStr += '\nend\n'

    return schedStr

def writeSHCIConfFile( SHCI, nelec, Restart ):
    confFile = os.path.join(SHCI.runtimeDir, SHCI.configFile)

    f = open(confFile, 'w')

    # Reference determinant section
    f.write('nocc %i\n'%(nelec[0]+nelec[1]))
    if SHCI.__class__.__name__ == 'FakeCISolver':
       for i in range(nelec[0]):
          f.write('%i '%(2*i))
       for i in range(nelec[1]):
          f.write('%i '%(2*i+1))
    else:
       if SHCI.irrep_nelec is None:
          for i in range(nelec[0]):
             f.write('%i '%(2*i))
          for i in range(nelec[1]):
             f.write('%i '%(2*i+1))
       else:
          from pyscf import symm
          from pyscf.dmrgscf import dmrg_sym
          from pyscf.symm.basis import DOOH_IRREP_ID_TABLE
          if SHCI.groupname is not None and SHCI.orbsym is not []:
             orbsym = dmrg_sym.convert_orbsym(SHCI.groupname, SHCI.orbsym)
          else:
             orbsym = [1]*norb

          for k,v in SHCI.irrep_nelec.items():

             irrep, nalpha, nbeta = [dmrg_sym.irrep_name2id(SHCI.groupname, k)],\
                                    v[0], v[1]

             for i in range(len(orbsym)):  #loop over alpha electrons
                if (orbsym[i] == irrep[0] and nalpha != 0):
                   f.write('%i '%(i*2))
                   nalpha -= 1
                if (orbsym[i] == irrep[0] and nbeta != 0):
                   f.write('%i '%(i*2+1))
                   nbeta -= 1
             if (nalpha != 0):
                print "number of irreps %s in active space = %d"%(k, v[0] - nalpha)
                print "number of irreps %s alpha electrons = %d"%(k, v[0])
                exit(1)
             if (nbeta != 0):
                print "number of irreps %s in active space = %d"%(k, v[1] - nbeta)
                print "number of irreps %s beta  electrons = %d"%(k, v[1])
                exit(1)
    f.write('\nend\n')
    f.write( 'nroots %r\n' % SHCI.nroots )

    # Variational Keyword Section
    if (not Restart):
       schedStr = make_sched( SHCI )
       f.write( schedStr )
    else:
       f.write('schedule\n')
       f.write('%d  %g\n'%(0, SHCI.sweep_epsilon[-1]))
       f.write('end\n')

    f.write( 'davidsonTol %g\n' %SHCI.davidsonTol )
    f.write( 'dE %g\n' %SHCI.dE )
    if (SHCI.DoRDM):
       f.write( 'DoRDM\n')

    # Perturbative Keyword Section
    if( SHCI.stochastic == False ):
        f.write( 'deterministic \n' )
    else:
       f.write( 'nPTiter %d\n' %SHCI.nPTiter )

    f.write( 'epsilon2 %g\n' %SHCI.epsilon2 )
    f.write( 'sampleN %i\n' %SHCI.sampleN )

    # Miscellaneous Keywords
    f.write( 'noio \n' )

    if (SHCI.prefix != "") :
       if not os.path.exists(SHCI.prefix):
          os.makedirs(SHCI.prefix)
       f.write( 'prefix %s\n' %(SHCI.prefix))

    # Sets maxiter to 6 more than the last iter in sweep_iter[] if restarted.
    if (not Restart):
       f.write( 'maxiter %i\n' %(SHCI.sweep_iter[-1]+6) )
    else:
       f.write('maxiter 6\n')
       f.write('fullrestart\n')

    f.write('\n') # SHCI requires that there is an extra line.

    f.close()


def D2htoDinfh(SHCI, norb, nelec):
   from pyscf import symm
   from pyscf.dmrgscf import dmrg_sym

   coeffs = numpy.zeros(shape=(norb, norb)).astype(complex);
   nRows = numpy.zeros(shape=(norb,), dtype=int)
   rowIndex = numpy.zeros(shape=(2*norb,), dtype=int)
   rowCoeffs = numpy.zeros(shape=(2*norb,), dtype=float)

   i, orbsym, ncore = 0, [0]*len(SHCI.orbsym), len(SHCI.orbsym)-norb

   while i < norb:
      symbol = symm.basis.linearmole_irrep_id2symb(SHCI.groupname, SHCI.orbsym[i+ncore])
      if (symbol[0] == 'A'):
         coeffs[i, i] = 1.0
         orbsym[i] = 1
         nRows[i] = 1
         rowIndex[2*i] = i
         rowCoeffs[2*i] = 1.
         if len(symbol) == 3 and symbol[2] == 'u':
            orbsym[i] = 2
      else:
         if (i == norb-1):
            print "the orbitals dont have dinfh symmetry"
            exit(0)
         l = int(symbol[1])
         orbsym[i], orbsym[i+1] = 2*l+3, -(2*l+3)
         if ( len(symbol) == 4 and symbol[2] == 'u'):
            orbsym[i], orbsym[i+1] = orbsym[i]+1, orbsym[i+1]-1
         if (symbol[3] == 'x'):
            m1, m2 = 1., -1.
         else:
            m1, m2 = -1., 1.


         nRows[i] = 2
         if m1 > 0 :
            coeffs[i, i], coeffs[i, i+1] = ((-1)**l)*1.0/(2.0**0.5), ((-1)**l)*1.0j/(2.0**0.5)
            rowIndex[2*i], rowIndex[2*i+1] = i, i+1
         else:
            coeffs[i, i+1], coeffs[i, i] = ((-1)**l)*1.0/(2.0**0.5), ((-1)**l)*1.0j/(2.0**0.5)
            rowIndex[2*i], rowIndex[2*i+1] = i+1, i

         rowCoeffs[2*i], rowCoeffs[2*i+1] = ((-1)**l)*1.0/(2.0**0.5), ((-1)**l)*1.0/(2.0**0.5)
         i = i+1

         nRows[i] = 2
         if (m1 > 0):  #m2 is the complex number
            rowIndex[2*i] = i-1
            rowIndex[2*i+1] = i
            rowCoeffs[2*i], rowCoeffs[2*i+1] = 1.0/(2.0**0.5), -1.0/(2.0**0.5)
            coeffs[i, i-1], coeffs[i, i] = 1.0/(2.0**0.5), -1.0j/(2.0**0.5)
         else:
            rowIndex[2*i] = i
            rowIndex[2*i+1] = i-1
            rowCoeffs[2*i], rowCoeffs[2*i+1] = 1.0/(2.0**0.5), -1.0/(2.0**0.5)
            coeffs[i, i], coeffs[i, i-1] = 1.0/(2.0**0.5), -1.0j/(2.0**0.5)

      i = i+1

   return coeffs, nRows, rowIndex, rowCoeffs, orbsym


def DinfhtoD2h(SHCI, norb, nelec):
   from pyscf import symm
   from pyscf.dmrgscf import dmrg_sym

   nRows = numpy.zeros(shape=(norb,), dtype=int)
   rowIndex = numpy.zeros(shape=(2*norb,), dtype=int)
   rowCoeffs = numpy.zeros(shape=(4*norb,), dtype=float)

   i, ncore = 0, len(SHCI.orbsym)-norb

   while i < norb:
      symbol = symm.basis.linearmole_irrep_id2symb(SHCI.groupname, SHCI.orbsym[i+ncore])
      if (symbol[0] == 'A'):
         nRows[i] = 1
         rowIndex[2*i] = i
         rowCoeffs[4*i] = 1.
      else:
         l = int(symbol[1])

         if (symbol[3] == 'x'):
            m1, m2 = 1., -1.
         else:
            m1, m2 = -1., 1.


         nRows[i] = 2
         rowIndex[2*i], rowIndex[2*i+1] = i, i+1
         if m1 > 0 :
            rowCoeffs[4*i], rowCoeffs[4*i+2] = ((-1)**l)*1.0/(2.0**0.5), 1.0/(2.0**0.5)
         else:
            rowCoeffs[4*i+1], rowCoeffs[4*i+3] = -((-1)**l)*1.0/(2.0**0.5), 1.0/(2.0**0.5)

         i = i+1

         nRows[i] = 2
         rowIndex[2*i], rowIndex[2*i+1] = i-1, i
         if (m1 > 0):  #m2 is the complex number
            rowCoeffs[4*i+1], rowCoeffs[4*i+3] = -((-1)**l)*1.0/(2.0**0.5), 1.0/(2.0**0.5)
         else:
            rowCoeffs[4*i], rowCoeffs[4*i+2] = ((-1)**l)*1.0/(2.0**0.5), 1.0/(2.0**0.5)


      i = i+1

   return nRows, rowIndex, rowCoeffs

def writeIntegralFile(SHCI, h1eff, eri_cas, norb, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else :
        neleca, nelecb = nelec

    # The name of the FCIDUMP file, default is "FCIDUMP".
    integralFile = os.path.join(SHCI.runtimeDir, SHCI.integralFile)

    if not os.path.exists(SHCI.scratchDirectory):
        os.makedirs(SHCI.scratchDirectory)


    from pyscf import symm
    from pyscf.dmrgscf import dmrg_sym

    if (SHCI.groupname == 'Dooh' or SHCI.groupname == 'Cooh') and SHCI.useExtraSymm:
       eri_cas = pyscf.ao2mo.restore(1, eri_cas, norb)
       coeffs, nRows, rowIndex, rowCoeffs, orbsym = D2htoDinfh(SHCI, norb, nelec)

       newintt = numpy.tensordot(coeffs.conj(), h1eff, axes=([1],[0]))
       newint1 = numpy.tensordot(newintt, coeffs, axes=([1],[1]))
       newint1r = numpy.zeros(shape=(norb, norb), order='C')
       for i in range(norb):
          for j in range(norb):
             newint1r[i,j] = newint1[i,j].real
       eri_cas = pyscf.ao2mo.restore(1, eri_cas, norb)
       int2 = 1.0*eri_cas
       eri_cas = 0.0*eri_cas

       transformDinfh(norb, numpy.ascontiguousarray(nRows, numpy.int32),
                      numpy.ascontiguousarray(rowIndex, numpy.int32),
                      numpy.ascontiguousarray(rowCoeffs, numpy.float64),
                      numpy.ascontiguousarray(int2, numpy.float64),
                      numpy.ascontiguousarray(eri_cas, numpy.float64))


       writeIntNoSymm(norb, numpy.ascontiguousarray(newint1r, numpy.float64),
                      numpy.ascontiguousarray(eri_cas, numpy.float64), ecore, neleca+nelecb,
                      numpy.asarray(orbsym, dtype=numpy.int32))

    else:
       if SHCI.groupname is not None and SHCI.orbsym is not []:
          orbsym = dmrg_sym.convert_orbsym(SHCI.groupname, SHCI.orbsym)
       else:
          eri_cas = pyscf.ao2mo.restore(8, eri_cas, norb)
          orbsym = [1]*norb

       eri_cas = pyscf.ao2mo.restore(8, eri_cas, norb)
       # Writes the FCIDUMP file using functions in SHCI_tools.cpp.
       fcidumpFromIntegral( integralFile, h1eff, eri_cas, norb, neleca+nelecb,
                            ecore, numpy.asarray(orbsym, dtype=numpy.int32),
                            abs(neleca-nelecb), 1e-15 )


def executeSHCI(SHCI):
    file1 = os.path.join(SHCI.runtimeDir, "%s/shci.e"%(SHCI.prefix))
    if os.path.exists(file1):
       os.remove(file1)
    inFile  = os.path.join(SHCI.runtimeDir, SHCI.configFile)
    outFile = os.path.join(SHCI.runtimeDir, SHCI.outputFile)
    try:
        cmd = ' '.join((SHCI.mpiprefix, SHCI.executable, inFile))
        cmd = "%s > %s 2>&1" % (cmd, outFile)
        check_call(cmd, shell=True)
    except CalledProcessError as err:
        logger.error(SHCI, cmd)
        raise err

def readEnergy(SHCI):
    file1 = open(os.path.join(SHCI.runtimeDir, "%s/shci.e"%(SHCI.prefix)), "rb")
    format = ['d']*SHCI.nroots
    format = ''.join(format)
    calc_e = struct.unpack(format, file1.read())
    file1.close()
    if SHCI.nroots == 1 :
       return calc_e[0]
    else:
       return list(calc_e)

def SHCISCF(mf, norb, nelec, maxM=1000, tol=1.e-8, *args, **kwargs):
    '''Shortcut function to setup CASSCF using the SHCI solver.  The SHCI
    solver is properly initialized in this function so that the 1-step
    algorithm can applied with SHCI-CASSCF.

    Examples:

    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = SHCISCF(mf, 4, 4)
    >>> mc.kernel()
    -74.414908818611522
    '''

    mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    mc.fcisolver = SHCI(mf.mol, maxM, tol=tol)
    # TODO
    #mc.callback = mc.fcisolver.restart_scheduler_()
    if mc.chkfile == mc._scf._chkfile.name:
        # Do not delete chkfile after mcscf
        mc.chkfile = tempfile.mktemp(dir=settings.SHCISCRATCHDIR)
        if not os.path.exists(settings.SHCISCRATCHDIR):
            os.makedirs(settings.SHCISCRATCHDIR)

    return mc



def writeSOCIntegrals(mc):
       from pyscf.lib.parameters import LIGHT_SPEED
       alpha = 1.0/LIGHT_SPEED
       rdm1ao = mc.make_rdm1()

       ncore, ncas = mc.ncore, mc.ncas
       mo_coeff = mc.mo_coeff
       # CAS space orbitals
       cas_orb = mc.mo_coeff[:,ncore:ncore+ncas]

       h2ao = -(alpha)**2*0.5*mc.mol.intor('cint2e_p1vxp1_sph', comp=3, aosym='s1')
       h2ao = h2ao.reshape(3,rdm1ao.shape[0],rdm1ao.shape[0],rdm1ao.shape[0],rdm1ao.shape[0])

       h1ao = 1.*(numpy.einsum('ijklm,lm->ijk', h2ao, rdm1ao) - 1.5*(numpy.einsum('ijklm, kl->ijm', h2ao, rdm1ao)+numpy.einsum('ijklm,mj->ilk', h2ao, rdm1ao) ) )

       for i in range(mc.mol.natm):
          r = mc.mol.atom_coord(i)
          Z = mc.mol.atom_charge(i)
          mc.mol.set_rinv_origin(r)  # set the gauge origin on second atom
          h1ao += (alpha)**2*0.5*Z*mc.mol.intor('cint1e_prinvxp_sph', comp=3)

       h1 = numpy.einsum('xpq,pi,qj->xij', h1ao,
                         mo_coeff, mo_coeff)[:, ncore:ncore+ncas, ncore:ncore+ncas]
       print1Int(h1, 'SOC')

def dryrun(mc, mo_coeff=None):
    '''Generate FCIDUMP and SHCI config file'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    #mc.fcisolver.onlywriteIntegral, bak = True, mc.fcisolver.onlywriteIntegral
    mc.casci(mo_coeff)
    #mc.fcisolver.onlywriteIntegral = bak

#mc is the CASSCF object
#nroots is the number of roots that will be used to calcualte the SOC matrix
def runQDPT(mc, gtensor):
   if mc.fcisolver.__class__.__name__ == 'FakeCISolver':
       SHCI = mc.fcisolver.fcisolvers[0]
       outFile = os.path.join(SHCI.runtimeDir, SHCI.outputFile)
       writeSHCIConfFile(SHCI, mc.nelecas, False)
       confFile = os.path.join(SHCI.runtimeDir, SHCI.configFile)
       f = open(confFile, 'a')
       for SHCI2 in mc.fcisolver.fcisolvers[1:] :
          if (SHCI2.prefix != "") :
             f.write( 'prefix %s\n' %(SHCI2.prefix))
       if (gtensor):
          f.write("dogtensor\n")
       f.close()
       try:
          cmd = ' '.join((SHCI.mpiprefix, SHCI.QDPTexecutable, confFile))
          cmd = "%s > %s 2>&1" % (cmd, outFile)
          check_call(cmd, shell=True)
          check_call('cat %s|grep -v "#"'%(outFile), shell=True)
       except CalledProcessError as err:
          logger.error(mc.fcisolver, cmd)
          raise err

   else:
       writeSHCIConfFile(mc.fcisolver, mc.nelecas, False)
       confFile = os.path.join(mc.fcisolver.runtimeDir, mc.fcisolver.configFile)
       outFile = os.path.join(mc.fcisolver.runtimeDir, mc.fcisolver.outputFile)
       if (gtensor):
          f = open(confFile, 'a')
          f.write("dogtensor\n")
          f.close()
       try:
          cmd = ' '.join((mc.fcisolver.mpiprefix, mc.fcisolver.QDPTexecutable, confFile))
          cmd = "%s > %s 2>&1" % (cmd, outFile)
          check_call(cmd, shell=True)
          check_call('cat %s|grep -v "#"'%(outFile), shell=True)
       except CalledProcessError as err:
          logger.error(mc.fcisolver, cmd)
          raise err

def doSOC(mc, gtensor=False):
   writeSOCIntegrals(mc)
   dryrun(mc)
   if (gtensor or True):
      ncore, ncas = mc.ncore, mc.ncas
      charge_center = numpy.einsum('z,zx->x',mc.mol.atom_charges(),mc.mol.atom_coords())
      h1ao = mc.mol.intor('cint1e_cg_irxp_sph', comp=3)
      h1 = numpy.einsum('xpq,pi,qj->xij', h1ao,
                        mc.mo_coeff, mc.mo_coeff)[:, ncore:ncore+ncas, ncore:ncore+ncas]
      print1Int(h1, 'GTensor')

   runQDPT(mc, gtensor)

if __name__ == '__main__':
    from pyscf import gto, scf, mcscf, dmrgscf
    from pyscf.future.shciscf import shci

    # Initialize N2 molecule
    b = 1.098
    mol = gto.Mole()
    mol.build(
    verbose = 5,
    output = None,
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    )

    # Create HF molecule
    mf = scf.RHF( mol )
    mf.conv_tol = 1e-9
    mf.scf()

    # Number of orbital and electrons
    norb = 8
    nelec = 10
    dimer_atom = 'N'

    mc = mcscf.CASSCF( mf, norb, nelec )
    e_CASSCF = mc.mc2step()[0]

    # Create SHCI molecule for just variational opt.
    # Active spaces chosen to reflect valence active space.
    mch = shci.SHCISCF( mf, norb, nelec )
    mch.fcisolver.nPTiter = 0 # Turn off perturbative calc.
    mch.fcisolver.outputFile = 'no_PT.dat'
    mch.fcisolver.sweep_iter = [ 0, 3 ]
    mch.fcisolver.DoRDM = True
    # Setting large epsilon1 thresholds highlights improvement from perturbation.
    mch.fcisolver.sweep_epsilon = [ 1e-2, 1e-2 ]
    e_noPT = mch.mc1step()[0]

    # Run a single SHCI iteration with perturbative correction.
    mch.fcisolver.stochastic = False # Turns on deterministic PT calc.
    mch.fcisolver.outputFile = 'PT.dat' # Save output under different name.
    shci.writeSHCIConfFile( mch.fcisolver, [nelec/2,nelec/2] , True )
    shci.executeSHCI( mch.fcisolver )

    # Open and get the energy from the binary energy file hci.e.
    # Open and get the energy from the
    with open( mch.fcisolver.outputFile, 'r' ) as f:
    	lines = f.readlines()

    e_PT = float( lines[ len(lines) - 1 ].split()[2] )

    # e_PT = shci.readEnergy( mch.fcisolver )

    # Comparison Calculations
    del_PT = e_PT - e_noPT
    del_shci = e_CASSCF - e_PT

    print( '\n\nEnergies for %s2 give in E_h.' % dimer_atom )
    print( '=====================================' )
    print( 'SHCI Variational: %6.12f' %e_noPT )
    # Prints the total energy including the perturbative component.
    print( 'SHCI Perturbative: %6.12f' %e_PT )
    print( 'Perturbative Change: %6.12f' %del_PT )
    print( 'CASSCF Total Energy: %6.12f' %e_CASSCF )
    print( 'E(CASSCF) - E(SHCI): %6.12f' %del_shci )
