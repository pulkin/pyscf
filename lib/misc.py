#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Some hacky functions
'''

import os, sys
import tempfile
import shutil
import functools
import itertools
import math
import ctypes
import numpy
import h5py
from pyscf.lib import param

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

def load_library(libname):
# numpy 1.6 has bug in ctypeslib.load_library, see numpy/distutils/misc_util.py
    if '1.6' in numpy.__version__:
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            raise OSError('Unknown platform')
        libname_so = libname + so_ext
        return ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname_so))
    else:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)

#Fixme, the standard resouce module gives wrong number when objects are released
#see http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/#fn:1
#or use slow functions as memory_profiler._get_memory did
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
PAGESIZE = os.sysconf("SC_PAGE_SIZE")
def current_memory():
    #import resource
    #return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    if sys.platform.startswith('linux'):
        with open("/proc/%s/statm" % os.getpid()) as f:
            vms, rss = [int(x)*PAGESIZE for x in f.readline().split()[:2]]
            return rss/1e6, vms/1e6
    else:
        return 0, 0

def num_threads():
    if 'OMP_NUM_THREADS' in os.environ:
        return int(os.environ['OMP_NUM_THREADS'])
    else:
        import multiprocessing
        return multiprocessing.cpu_count()


def c_int_arr(m):
    npm = numpy.array(m).flatten('C')
    arr = (ctypes.c_int * npm.size)(*npm)
    # cannot return LP_c_double class,
    #Xreturn npm.ctypes.data_as(c_int_p), which destructs npm before return
    return arr
def f_int_arr(m):
    npm = numpy.array(m).flatten('F')
    arr = (ctypes.c_int * npm.size)(*npm)
    return arr
def c_double_arr(m):
    npm = numpy.array(m).flatten('C')
    arr = (ctypes.c_double * npm.size)(*npm)
    return arr
def f_double_arr(m):
    npm = numpy.array(m).flatten('F')
    arr = (ctypes.c_double * npm.size)(*npm)
    return arr


def member(test, x, lst):
    for l in lst:
        if test(x, l):
            return True
    return False

def remove_dup(test, lst, from_end=False):
    if test is None:
        return set(lst)
    else:
        if from_end:
            lst = list(reversed(lst))
        seen = []
        for l in lst:
            if not member(test, l, seen):
                seen.append(l)
        return seen

def remove_if(test, lst):
    return [x for x in lst if not test(x)]

def find_if(test, lst):
    for l in lst:
        if test(l):
            return l
    raise ValueError('No element of the given list matches the test condition.')

def arg_first_match(test, lst):
    for i,x in enumerate(lst):
        if test(x):
            return i
    raise ValueError('No element of the given list matches the test condition.')

def tril_equal_pace(n, base=0, npace=0, minimal=1):
    idx = numpy.arange(n+1)
    cum = idx * (idx+1) // 2
    if base == 0:
        assert(npace > 0)
        segsize = float(cum[-1]) / (npace-.5)
        displs = _blocksize_partition(cum, segsize)
        if len(displs) != npace+1:
            displs = _balanced_partition(cum, npace)
    else:
        displs = _blocksize_partition(cum, base)
    for p0, p1 in zip(displs[:-1], displs[1:]):
        yield p0, p1

def _balanced_partition(cum, ntasks):
    segsize = float(cum[-1]) / ntasks
    bounds = numpy.arange(ntasks+1) * segsize
    displs = abs(bounds[:,None] - cum).argmin(axis=1)
    return displs

def _blocksize_partition(cum, blocksize):
    n = len(cum) - 1
    displs = [0]
    p0 = 0
    for i in range(1, n):
        if cum[i+1]-cum[p0] > blocksize:
            displs.append(i)
            p0 = i
    displs.append(n)
    return displs

def flatten(lst):
    '''flatten nested lists
    x[0] + x[1] + x[2] + ...

    Examples:

    >>> flatten([[0, 2], [1], [[9, 8, 7]]])
    [0, 2, 1, [9, 8, 7]]
    '''
    return list(itertools.chain.from_iterable(lst))

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


class ctypes_stdout(object):
    '''make c-printf output to string, but keep python print in /dev/pts/1.
    Note it cannot correctly handle c-printf with GCC, don't know why.
    Usage:
        with ctypes_stdout() as stdout:
            ...
        print(stdout.read())'''
    def __enter__(self):
        sys.stdout.flush()
        self._contents = None
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.bak_stdout = sys.stdout
        self.fd, self.ftmp = tempfile.mkstemp(dir='/dev/shm')
        os.dup2(self.fd, self.old_stdout_fileno)
        sys.stdout = os.fdopen(self.bak_stdout_fd, 'w')
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        os.fsync(self.fd)
        self._contents = open(self.ftmp, 'r').read()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        sys.stdout = self.bak_stdout # self.bak_stdout_fd is closed
        #os.close(self.fd) is closed when os.fdopen is closed
        os.remove(self.ftmp)
    def read(self):
        if self._contents:
            return self._contents
        else:
            sys.stdout.flush()
            #f = os.fdopen(self.fd, 'r') # need to rewind(0) before reading
            #f.seek(0)
            return open(self.ftmp, 'r').read()

class capture_stdout(object):
    '''redirect all stdout (c printf & python print) into a string
    Usage:
        with capture_stdout() as stdout:
            ...
        print(stdout.read())
    '''
    def __enter__(self):
        sys.stdout.flush()
        self._contents = None
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.fd, self.ftmp = tempfile.mkstemp(dir='/dev/shm')
        os.dup2(self.fd, self.old_stdout_fileno)
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        self._contents = open(self.ftmp, 'r').read()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        os.close(self.bak_stdout_fd)
        #os.close(self.fd) will be closed when os.fdopen is closed
        os.remove(self.ftmp)
    def read(self):
        if self._contents:
            return self._contents
        else:
            sys.stdout.flush()
            #f = os.fdopen(self.fd, 'r') # need to rewind(0) before reading
            #f.seek(0)
            return open(self.ftmp, 'r').read()

class quite_run(object):
    '''output nothing

    Examples
    --------
    with quite_run():
        ...
    '''
    def __enter__(self):
        sys.stdout.flush()
        self.dirnow = os.getcwd()
        self.tmpdir = tempfile.mkdtemp(dir='/dev/shm')
        os.chdir(self.tmpdir)
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.fnull = open(os.devnull, 'wb')
        os.dup2(self.fnull.fileno(), self.old_stdout_fileno)
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        self.fnull.close()
        shutil.rmtree(self.tmpdir)
        os.chdir(self.dirnow)


# from pygeocoder
# this decorator lets me use methods as both static and instance methods
# In contrast to classmethod, when obj.function() is called, the first
# argument is obj in omnimethod rather than obj.__class__ in classmethod
class omnimethod(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return functools.partial(self.func, instance)


class StreamObject(object):
    '''For most methods, there are three stream functions to pipe computing stream:

    1 ``.set_`` function to update object attributes, eg
    ``mf = scf.RHF(mol).set(conv_tol=1e-5)`` is identical to proceed in two steps
    ``mf = scf.RHF(mol); mf.conv_tol=1e-5``

    2 ``.run`` function to execute the kenerl function (the function arguments
    are passed to kernel function).  If keyword arguments is given, it will first
    call ``.set`` function to update object attributes then execute the kernel
    function.  Eg
    ``mf = scf.RHF(mol).run(dm_init, conv_tol=1e-5)`` is identical to three steps
    ``mf = scf.RHF(mol); mf.conv_tol=1e-5; mf.kernel(dm_init)``

    3 ``.apply`` function to apply the given function/class to the current object
    (function arguments and keyword arguments are passed to the given function).
    Eg
    ``mol.apply(scf.RHF).run().apply(mcscf.CASSCF, 6, 4, frozen=4)`` is identical to
    ``mf = scf.RHF(mol); mf.kernel(); mcscf.CASSCF(mf, 6, 4, frozen=4)``
    '''

    verbose = 0
    stdout = sys.stdout
    _keys = set(['verbose', 'stdout'])

    def run(self, *args, **kwargs):
        '''Call the kernel function of current object.  `args` will be passed
        to kernel function.  `kwargs` will be used to update the attributes of
        current object.
        '''
        self.set(**kwargs)
        self.kernel(*args)
        return self

    def set(self, **kwargs):
        '''Update the attributes of the current object.
        '''
        #if hasattr(self, '_keys'):
        #    for k,v in kwargs.items():
        #        setattr(self, k, v)
        #        if k not in self._keys:
        #            sys.stderr.write('Warning: %s does not have attribute %s\n'
        #                             % (self.__class__, k))
        #else:
        for k,v in kwargs.items():
            setattr(self, k, v)
        return self

    def apply(self, fn, *args, **kwargs):
        '''Apply the fn to rest arguments:  return fn(*args, **kwargs)
        '''
        return fn(self, *args, **kwargs)

#    def _format_args(self, args, kwargs, kernel_kw_lst):
#        args1 = [kwargs.pop(k, v) for k, v in kernel_kw_lst]
#        return args + args1[len(args):], kwargs

    def check_sanity(self):
        '''Check misinput of class attributes, check whether a class method is
        overwritten.  It does not check the attributes which are prefixed with
        "_".
        '''
        if (self.verbose > 0 and  # logger.QUIET
            hasattr(self, '_keys')):
            check_sanity(self, self._keys, self.stdout)
        return self

_warn_once_registry = {}
def check_sanity(obj, keysref, stdout=sys.stdout):
    '''Check misinput of class attributes, check whether a class method is
    overwritten.  It does not check the attributes which are prefixed with
    "_".
    '''
    objkeys = [x for x in obj.__dict__ if not x.startswith('_')]
    keysub = set(objkeys) - set(keysref)
    if keysub:
        class_attr = set(dir(obj.__class__))
        keyin = keysub.intersection(class_attr)
        if keyin:
            msg = ('Overwrite attributes  %s  of %s\n' %
                   (' '.join(keyin), obj.__class__))
            if msg not in _warn_once_registry:
                _warn_once_registry[msg] = 1
                sys.stderr.write(msg)
                if stdout is not sys.stdout:
                    stdout.write(msg)
        keydiff = keysub - class_attr
        if keydiff:
            msg = ('%s does not have attributes  %s\n' %
                   (obj.__class__, ' '.join(keydiff)))
            if msg not in _warn_once_registry:
                _warn_once_registry[msg] = 1
                sys.stderr.write(msg)
                if stdout is not sys.stdout:
                    stdout.write(msg)
    return obj

def with_doc(doc):
    '''Use this decorator to add doc string for function

        @with_doc(doc)
        def fn:
            ...

    makes

        fn.__doc__ = doc
    '''
    def make_fn(fn):
        fn.__doc__ = doc
        return fn
    return make_fn

def overwrite_mro(obj, mro):
    '''A hacky function to overwrite the __mro__ attribute'''
    class HackMRO(type):
        pass
# Overwrite type.mro function so that Temp class can use the given mro
    HackMRO.mro = lambda self: mro
    if sys.version_info < (3,):
        class Temp(obj.__class__):
            __metaclass__ = HackMRO
    else:
        #class Temp(obj.__class__, metaclass=HackMRO):
        #    pass
        raise NotImplementedError()
    obj = Temp()
# Delete mro function otherwise all subclass of Temp are not able to
# resolve the right mro
    del(HackMRO.mro)
    return obj

def izip(*args):
    '''python2 izip == python3 zip'''
    if sys.version_info < (3,):
        return itertools.izip(*args)
    else:
        return zip(*args)

from threading import Thread
from multiprocessing import Queue, Process
class ProcessWithReturnValue(Process):
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None):
        self._q = Queue()
        def qwrap(*args, **kwargs):
            self._q.put(target(*args, **kwargs))
        Process.__init__(self, group, qwrap, name, args, kwargs)
    def join(self):
        Process.join(self)
        return self._q.get()
    get = join

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None):
        self._q = Queue()
        def qwrap(*args, **kwargs):
            self._q.put(target(*args, **kwargs))
        Thread.__init__(self, group, qwrap, name, args, kwargs)
    def join(self):
        Thread.join(self)
        return self._q.get()
    get = join

def background_thread(func, *args, **kwargs):
    '''applying function in background'''
    thread = ThreadWithReturnValue(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread

def background_process(func, *args, **kwargs):
    '''applying function in background'''
    thread = ProcessWithReturnValue(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread

bg = background = bg_thread = background_thread
bp = bg_process = background_process


class H5TmpFile(h5py.File):
    def __init__(self, filename=None, *args, **kwargs):
        if filename is None:
            tmpfile = tempfile.NamedTemporaryFile(dir=param.TMPDIR)
            filename = tmpfile.name
        h5py.File.__init__(self, filename, *args, **kwargs)
    def __del__(self):
        self.close()


if __name__ == '__main__':
    for i,j in tril_equal_pace(90, 30):
        print('base=30', i, j, j*(j+1)//2-i*(i+1)//2)
    for i,j in tril_equal_pace(90, npace=5):
        print('npace=5', i, j, j*(j+1)//2-i*(i+1)//2)
