#!/usr/bin/env python
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import ctypes
import numpy
import h5py
from pyscf.dft.numint import _dot_ao_ao, _dot_ao_dm, BLKSIZE
from pyscf import lib
from pyscf import dft
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import is_zero, gamma_point, member

libpbc = lib.load_library('libpbc')

def eval_ao(cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0, shl_slice=None,
            non0tab=None, out=None, verbose=None):
    '''Collocate AO crystal orbitals (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Cell`

        coords : (nx*ny*nz, 3) ndarray
            The real-space grid point coordinates.

    Kwargs:
        kpt : (3,) ndarray
            The k-point corresponding to the crystal AO.
        deriv : int
            AO derivative order.  It affects the shape of the return array.
            If deriv=0, the returned AO values are stored in a (N,nao) array.
            Otherwise the AO values are stored in an array of shape (M,N,nao).
            Here N is the number of grids, nao is the number of AO functions,
            M is the size associated to the derivative deriv.

    Returns:
        aoR : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If deriv=1, also contains the value of the orbitals gradient in the
            x, y, and z directions.  It can be either complex or float array,
            depending on the kpt argument.  If kpt is not given (gamma point),
            aoR is a float array.

    See Also:
        pyscf.dft.numint.eval_ao

    '''
    ao_kpts = eval_ao_kpts(cell, coords, numpy.reshape(kpt, (-1,3)), deriv,
                           relativity, shl_slice, non0tab, out, verbose)
    return ao_kpts[0]

def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shl_slice=None, non0tab=None, out=None, verbose=None, kpt=None):
    '''
    Returns:
        ao_kpts: (nkpts, ngs, nao) ndarray
            AO values at each k-point
    '''
    if kpts is None:
        if not kpt is None:
            sys.stderr.write('WARN: _KNumInt.eval_ao function finds keyword '
                             'argument "kpt" and converts it to "kpts"\n')
            kpts = kpt
        else:
            kpts = numpy.zeros((1,3))
    kpts = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts)
    ngrids = len(coords)

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE, cell.nbas),
                             dtype=numpy.int8)

    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    ao_kpts = [numpy.zeros((ngrids,nao,comp), dtype=numpy.complex128, order='F')
               for k in range(nkpts)]
    out_ptrs = (ctypes.c_void_p*nkpts)(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in ao_kpts])
    coords = numpy.asarray(coords, order='C')
    Ls = cell.get_lattice_Ls()
    expLk = numpy.exp(1j * numpy.asarray(numpy.dot(Ls, kpts.T), order='C'))

    drv = getattr(libpbc, 'PBCval_sph_deriv%d' % deriv)
    drv(ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
        Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(Ls)),
        expLk.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts),
        (ctypes.c_int*2)(0, cell.nbas),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        out_ptrs, coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        cell._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
        cell._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
        cell._env.ctypes.data_as(ctypes.c_void_p))

    for k, kpt in enumerate(kpts):
        mat = ao_kpts[k].transpose(2,0,1)
        if comp == 1:
            aos = lib.transpose(mat[0].T)
        else:
            aos = numpy.empty((comp,ngrids,nao), dtype=numpy.complex128)
            for i in range(comp):
                lib.transpose(mat[i].T, out=aos[i])

        if abs(kpt).sum() < 1e-9:  # gamma point
            aos = aos.real.copy()

        ao_kpts[k] = aos
    return ao_kpts


def eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', verbose=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Args:
        cell : instance of :class:`Mole` or :class:`Cell`

        ao : ([4,] nx*ny*nz, nao=cell.nao_nr()) ndarray
            The value of the AO crystal orbitals on the real-space grid by default.
            If xctype='GGA', also contains the value of the gradient in the x, y,
            and z directions.

    Returns:
        rho : ([4,] nx*ny*nz) ndarray
            The value of the density on the real-space grid. If xctype='GGA',
            also contains the value of the gradient in the x, y, and z
            directions.

    See Also:
        pyscf.dft.numint.eval_rho

    '''

    assert(ao.flags.c_contiguous)
    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                             dtype=numpy.int8)

    # complex orbitals or density matrix
    if numpy.iscomplexobj(ao) or numpy.iscomplexobj(dm):

        def re_im(a):
            return (numpy.asarray(a.real, order='C'),
                    numpy.asarray(a.imag, order='C'))
        dm_re, dm_im = re_im(dm)
        def dot_dm_ket(ket_re, ket_im):
            # DM * ket: e.g. ir denotes dm_im | ao_re >
            c0_rr = _dot_ao_dm(cell, ket_re, dm_re, nao, ngrids, non0tab)
            c0_ri = _dot_ao_dm(cell, ket_im, dm_re, nao, ngrids, non0tab)
            c0_ir = _dot_ao_dm(cell, ket_re, dm_im, nao, ngrids, non0tab)
            c0_ii = _dot_ao_dm(cell, ket_im, dm_im, nao, ngrids, non0tab)
            return c0_ri, c0_rr, c0_ir, c0_ii
        def dot_bra(bra_re, bra_im, c0):
            # bra * DM
            c0_ri, c0_rr, c0_ir, c0_ii = c0
            rho = (numpy.einsum('pi,pi->p', bra_im, c0_ri) +
                   numpy.einsum('pi,pi->p', bra_re, c0_rr) +
                   numpy.einsum('pi,pi->p', bra_im, c0_ir) -
                   numpy.einsum('pi,pi->p', bra_re, c0_ii))
            return rho

        if xctype == 'LDA':
            ao_re, ao_im = re_im(ao)
            c0 = dot_dm_ket(ao_re, ao_im)
            rho = dot_bra(ao_re, ao_im, c0)

        elif xctype == 'GGA':
            rho = numpy.empty((4,ngrids))
            ao0_re, ao0_im = re_im(ao[0])
            c0 = dot_dm_ket(ao0_re, ao0_im)
            rho[0] = dot_bra(ao0_re, ao0_im, c0)

            for i in range(1, 4):
                ao_re, ao_im = re_im(ao[i])
                rho[i] = dot_bra(ao_re, ao_im, c0) * 2

        else:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            ao0_re, ao0_im = re_im(ao[0])
            c0 = dot_dm_ket(ao0_re, ao0_im)
            rho[0] = dot_bra(ao0_re, ao0_im, c0)

            rho[5] = 0
            for i in range(1, 4):
                ao_re, ao_im = re_im(ao[i])
                rho[i] = dot_bra(ao_re, ao_im, c0) * 2 # *2 for +c.c.
                c1 = dot_dm_ket(ao_re, ao_im)
                rho[5] += dot_bra(ao_re, ao_im, c1)
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            ao_re, ao_im = re_im(ao2)
            rho[4] = dot_bra(ao_re, ao_im, c0)
            rho[4] += rho[5]
            rho[4] *= 2 # *2 for +c.c.
            rho[5] *= .5

    # real orbitals and real DM
    else:
        rho = dft.numint.eval_rho(cell, ao, dm, non0tab, xctype, verbose)

    return rho

def eval_mat(cell, ao, weight, rho, vxc,
             non0tab=None, xctype='LDA', spin=0, verbose=None):
    '''Calculate the XC potential AO matrix.

    Args:
        cell : instance of :class:`Mole` or :class:`Cell`

        ao : ([4/10,] ngrids, nao) ndarray
            2D array of shape (N,nao) for LDA,
            3D array of shape (4,N,nao) for GGA
            or (10,N,nao) for meta-GGA.
            N is the number of grids, nao is the number of AO functions.
            If xctype is GGA, ao[0] is AO value and ao[1:3] are the real space
            gradients.  If xctype is meta-GGA, ao[4:10] are second derivatives
            of ao values.
        rho : ([4/6,] ngrids) ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))
        vxc : ([4,] ngrids) ndarray
            XC potential value on each grid = (vrho, vsigma, vlapl, vtau)
            vsigma is GGA potential value on each grid.
            If the kwarg spin is not 0, a list [vsigma_uu,vsigma_ud] is required.

    See Also:
        dft.numint.eval_mat

    '''

    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                             dtype=numpy.int8)

    if numpy.iscomplexobj(ao):
        def dot(ao1, ao2):
            ao1_re = numpy.asarray(ao1.real, order='C')
            ao1_im = numpy.asarray(ao1.imag, order='C')
            ao2_re = numpy.asarray(ao2.real, order='C')
            ao2_im = numpy.asarray(ao2.imag, order='C')

            mat_re  = _dot_ao_ao(cell, ao1_re, ao2_re, nao, ngrids, non0tab)
            mat_re += _dot_ao_ao(cell, ao1_im, ao2_im, nao, ngrids, non0tab)
            mat_im  = _dot_ao_ao(cell, ao1_re, ao2_im, nao, ngrids, non0tab)
            mat_im -= _dot_ao_ao(cell, ao1_im, ao2_re, nao, ngrids, non0tab)
            return mat_re + 1j*mat_im

        if xctype == 'LDA':
            if not isinstance(vxc, numpy.ndarray) or vxc.ndim == 2:
                vrho = vxc[0]
            else:
                vrho = vxc
            # *.5 because return mat + mat.T
            aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
            mat = dot(ao, aow)

        else:
            vrho, vsigma = vxc[:2]
            wv = numpy.empty((4,ngrids))
            if spin == 0:
                #wv = weight * vsigma * 2
                #aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
                #aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
                #aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
                #aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
                wv[0]  = weight * vrho * .5
                wv[1:4] = rho[1:4] * (weight * vsigma * 2)
            else:
                rho_a, rho_b = rho
                wv[0]  = weight * vrho * .5
                wv[1:4] = rho_a[1:4] * (weight * vsigma[0] * 2)  # sigma_uu
                wv[1:4]+= rho_b[1:4] * (weight * vsigma[1])      # sigma_ud
            aow = numpy.einsum('npi,np->pi', ao[:4], wv)
            mat = dot(ao[0], aow)

        if xctype == 'MGGA':
            vlapl, vtau = vxc[2:]
            if vlapl is None:
                vlpal = 0
            aow = numpy.einsum('npi,p->npi', ao[1:4], weight * (.25*vtau+vlapl))
            mat += dot(ao[1], aow[0])
            mat += dot(ao[2], aow[1])
            mat += dot(ao[3], aow[2])

            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            aow = numpy.einsum('pi,p->pi', ao2, .5 * weight * vlapl)
            mat += dot(ao[0], aow)

        return (mat + mat.T.conj())

    else:
        return dft.numint.eval_mat(cell, ao, weight, rho, vxc,
                                   non0tab, xctype, spin, verbose)


def nr_rks(ni, cell, grids, xc_code, dm, spin=0, relativity=0, hermi=1,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None, precomputed_ao = None):
    '''Calculate RKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.
    Faster function uses eval_rho2 which is not yet implemented.

    Args:
        ni : an instance of :class:`_NumInt` or :class:`_KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm : 2D array
            Density matrix

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        hermi : int
            No effects
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm)

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = [0]*nset
    if xctype == 'LDA':
        ao_deriv = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory, precomputed_ao = precomputed_ao):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1)[:2]
                vrho = vxc[0]
                den = rho*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory, precomputed_ao = precomputed_ao):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1)[:2]
                den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    else:
        assert(all(x not in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')))
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory, precomputed_ao = precomputed_ao):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, 0, relativity, 1)[:2]
                den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)

    #if kpts_band is not None:
    #    vmat = [v.reshape(nao,nao) for v in vmat]

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, vmat

def nr_uks(ni, cell, grids, xc_code, dm, spin=1, relativity=0, hermi=1,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None, precomputed_ao = None):
    '''Calculate UKS XC functional and potential matrix for given meshgrids and density matrix

    Note: This is a replica of pyscf.dft.numint.nr_rks_vxc with kpts added.
    This implemented uses slow function in numint, which only calls eval_rho, eval_mat.
    Faster function uses eval_rho2 which is not yet implemented.

    Args:
        ni : an instance of :class:`_NumInt` or :class:`_KNumInt`

        cell : instance of :class:`Mole` or :class:`Cell`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm : 2D array
            Density matrix

    Kwargs:
        spin : int
            spin polarized if spin = 1
        relativity : int
            No effects.
        hermi : int
            No effects
        max_memory : int or float
            The maximum size of cache to use (in MB).
        verbose : int or object of :class:`Logger`
            No effects.
        kpts : (3,) ndarray or (nkpts,3) ndarray
            Single or multiple k-points sampled for the DM.  Default is gamma point.
            kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evaluate the XC matrix.

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    if kpts is None:
        kpts = numpy.zeros((1,3))
    xctype = ni._xc_type(xc_code)
    dm = numpy.asarray(dm)
    nao = dm.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(cell, dm[0].reshape(-1,nao,nao))[:2]
    make_rhob       = ni._gen_rho_evaluator(cell, dm[1].reshape(-1,nao,nao))[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmata = [0]*nset
    vmatb = [0]*nset
    if xctype == 'LDA':
        ao_deriv = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b * weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                vmata[i] += ni.eval_mat(cell, ao_k1, weight, rho_a, vrho[:,0],
                                        mask, xctype, 1, verbose)
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, rho_b, vrho[:,1],
                                        mask, xctype, 1, verbose)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts,
                                 kpts_band, max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose)[:2]
                vrho, vsigma = vxc[:2]
                den = rho_a[0]*weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b[0]*weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                vmata[i] += ni.eval_mat(cell, ao_k1, weight, (rho_a,rho_b),
                                        (vrho[:,0], (vsigma[:,0],vsigma[:,1])),
                                        mask, xctype, 1, verbose)
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, (rho_b,rho_a),
                                        (vrho[:,1], (vsigma[:,2],vsigma[:,1])),
                                        mask, xctype, 1, verbose)
    else:
        assert(all(x not in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')))
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao_k2, mask, xctype)
                rho_b = make_rhob(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b),
                                      1, relativity, 1, verbose)[:2]
                vrho, vsigma, vlapl, vtau = vxc
                den = rho_a[0]*weight
                nelec[0,i] += den.sum()
                excsum[i] += (den*exc).sum()
                den = rho_b[0]*weight
                nelec[1,i] += den.sum()
                excsum[i] += (den*exc).sum()

                v = (vrho[:,0], (vsigma[:,0],vsigma[:,1]), None, vtau[:,0])
                vmata[i] += ni.eval_mat(cell, ao_k1, weight, (rho_a,rho_b), v,
                                        mask, xctype, 1, verbose)
                v = (vrho[:,1], (vsigma[:,2],vsigma[:,1]), None, vtau[:,1])
                vmatb[i] += ni.eval_mat(cell, ao_k1, weight, (rho_b,rho_a), v,
                                        mask, xctype, 1, verbose)
                v = None

    if kpts_band is not None:
        vmata = [v.reshape(nao,nao) for v in vmata]
        vmatb = [v.reshape(nao,nao) for v in vmatb]

    if nset == 1:
        nelec = nelec[:,0]
        excsum = excsum[0]
        vmata = vmata[0]
        vmatb = vmatb[0]
    return nelec, excsum, lib.asarray((vmata,vmatb))

nr_rks_vxc = nr_rks
nr_uks_vxc = nr_uks


def large_rho_indices(ni, cell, dm, grids, cutoff=1e-10, kpt=numpy.zeros(3),
                      max_memory=2000, precomputed_ao = None):
    '''Indices of density which are larger than given cutoff
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm)
    idx = []
    cutoff = cutoff / grids.weights.size
    for ao_k1, ao_k2, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, 0, kpt, kpt, max_memory, precomputed_ao = precomputed_ao):
        rho = make_rho(0, ao_k2, mask, 'LDA')
        idx.append(abs(rho*weight) > cutoff)
    return numpy.hstack(idx)


def get_default_block_size(max_memory, nao, ng, nk, deriv):
    """
    Retrieves defualt size of the block to process via a single thread.
    Makes a choice based on:
    
    Args:
    
        max_memory (float): maximal memory in Mb;
        
        nao (int): number of degrees of freedom (atomic orbitals);
        
        ng (int): number of grid points in real space;
        
        nk (int): number of grid points in reciprocal space (k-points);
        
        derive (int): maximal derivative order in atomic orbitals;
        
    Returns:
    
        An optimal block size: integer multiplier of `pyscf.dft.numint.BLKSIZE`.
    """
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    result = int(max_memory*1e6/(comp*2*nk*nao*16*BLKSIZE))*BLKSIZE
    return max(min(result,ng), BLKSIZE)
    
def block_loop(self, cell, grids, nao, deriv=0, kpts=numpy.zeros((1,3)),
               kpts_band=None, max_memory=2000, non0tab=None, blksize=None, precomputed_ao = None):
    '''Define this macro to loop over grids by blocks.
    '''
    ngrids = grids.weights.size

    if blksize is None:
        blksize = get_default_block_size(max_memory, nao, ngrids, len(kpts), deriv)
        
    if kpts_band is not None:
        kpts_band = numpy.reshape(kpts_band, (-1,3))
        where = [member(k, kpts) for k in kpts_band]
        where = [k_id[0] if len(k_id)>0 else None for k_id in where]

    for ip0 in range(0, ngrids, blksize):
        ip1 = min(ngrids, ip0+blksize)
        coords = grids.coords[ip0:ip1]
        weight = grids.weights[ip0:ip1]
        non0 = non0tab[ip0//BLKSIZE:] if not non0tab is None else None
        if precomputed_ao is None:
            ao_k2 = self.eval_ao(cell, coords, kpts, deriv=deriv, non0tab=non0)
        else:
            ao_k2 = list(i[...,ip0:ip1,:] for i in precomputed_ao)
        if kpts_band is None:
            ao_k1 = ao_k2
        else:
            new_kpts = [k for k,w in zip(kpts_band, where) if w is None]
            if len(new_kpts)>0:
                new_ao = iter(self.eval_ao(cell, coords, new_kpts, deriv=deriv, non0tab=non0))
            old_ao = (ao_k2[w] for w in where if not w is None)
            ao_k1 = []
            for w in where:
                ao_k1.append(next(new_ao) if w is None else next(old_ao))
        yield ao_k1, ao_k2, non0, weight, coords
        ao_k1 = ao_k2 = None

class _NumInt(dft.numint._NumInt):
    '''Generalization of pyscf's _NumInt class for a single k-point shift and
    periodic images.
    '''
    def eval_ao(self, cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0,
                shl_slice=None, non0tab=None, out=None, verbose=None):
        return eval_ao(cell, coords, kpt, deriv, relativity, shl_slice,
                       non0tab, out, verbose)

    def eval_rho(self, cell, ao, dm, non0tab=None, xctype='LDA', verbose=None):
        return eval_rho(cell, ao, dm, non0tab, xctype, verbose)

    def eval_rho2(self, cell, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        dm = numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())
        return eval_rho(cell, ao, dm, non0tab, xctype, verbose)

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               kpt=None, kpt_band=None, max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix.
        See :func:`nr_rks` and :func:`nr_uks` for more details.
        '''
        if spin == 0:
            return self.nr_rks(cell, grids, xc_code, dms, hermi,
                               kpt, kpt_band, max_memory, verbose)
        else:
            return self.nr_uks(cell, grids, xc_code, dms, hermi,
                               kpt, kpt_band, max_memory, verbose)

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, hermi=1,
               kpt=numpy.zeros(3), kpt_band=None, max_memory=2000, verbose=None, precomputed_ao = None):
        return nr_rks(self, cell, grids, xc_code, dms,
                      0, 0, 1, kpt, kpt_band, max_memory, verbose, precomputed_ao)

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, hermi=1,
               kpt=numpy.zeros(3), kpt_band=None, max_memory=2000, verbose=None, precomputed_ao = None):
        return nr_uks(self, cell, grids, xc_code, dms,
                      1, 0, 1, kpt, kpt_band, max_memory, verbose, precomputed_ao)

    def eval_mat(self, cell, ao, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        # use local function for complex eval_mat
        return eval_mat(cell, ao, weight, rho, vxc,
                        non0tab, xctype, spin, verbose)

    def block_loop(self, cell, grids, nao, deriv=0, kpt=numpy.zeros(3),
                   kpt_band=None, max_memory=2000, non0tab=None, blksize=None, precomputed_ao = None):
        '''Define this macro to loop over grids by blocks.
        '''
        ngrids = grids.weights.size
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index ni.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = min(int(max_memory*1e6/(comp*2*nao*16*BLKSIZE))*BLKSIZE, ngrids)
            blksize = max(blksize, BLKSIZE)
        if non0tab is None:
            non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                 dtype=numpy.int8)
        kpt = numpy.reshape(kpt, 3)
        if kpt_band is None:
            kpt1 = kpt2 = kpt
        else:
            kpt1 = kpt_band
            kpt2 = kpt

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            if precomputed_ao is None:
                ao_k2 = self.eval_ao(cell, coords, kpt2, deriv=deriv, non0tab=non0)
            else:
                ao_k2 = list(i[...,ip0:ip1,:] for i in precomputed_ao)
            if abs(kpt1-kpt2).sum() < 1e-9:
                ao_k1 = ao_k2
            else:
                ao_k1 = self.eval_ao(cell, coords, kpt1, deriv=deriv)
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

    def _gen_rho_evaluator(self, cell, dms, hermi=0):
        return dft.numint._NumInt._gen_rho_evaluator(self, cell, dms, 0)

    large_rho_indices = large_rho_indices


class _KNumInt(dft.numint._NumInt):
    '''Generalization of pyscf's _NumInt class for k-point sampling and
    periodic images.
    '''
    def __init__(self, kpts=numpy.zeros((1,3))):
        dft.numint._NumInt.__init__(self)
        self.kpts = numpy.reshape(kpts, (-1,3))

    def eval_ao(self, cell, coords, kpts=numpy.zeros((1,3)), deriv=0, relativity=0,
                shl_slice=None, non0tab=None, out=None, verbose=None, **kwargs):
        return eval_ao_kpts(cell, coords, kpts, deriv,
                            relativity, shl_slice, non0tab, out, verbose)

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 verbose=None):
        '''
        Args:
            cell : Mole or Cell object
            ao_kpts : (nkpts, ngs, nao) ndarray
                AO values at each k-point
            dm_kpts: (nkpts, nao, nao) ndarray
                Density matrix at each k-point

        Returns:
           rhoR : (ngs,) ndarray
        '''
        nkpts = len(ao_kpts)
        ngs = ao_kpts[0].shape[-2]
        rhoR = 0
        for k in range(nkpts):
            rhoR += 1./nkpts*eval_rho(cell, ao_kpts[k], dm_kpts[k], non0tab,
                                      xctype, verbose)
        return rhoR

    def eval_rho2(self, cell, ao, dm, non0tab=None, xctype='LDA', verbose=None):
        raise NotImplementedError

    def nr_vxc(self, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        '''Evaluate RKS/UKS XC functional and potential matrix.
        See :func:`nr_rks` and :func:`nr_uks` for more details.
        '''
        if spin == 0:
            return self.nr_rks(cell, grids, xc_code, dms, hermi,
                               kpts, kpts_band, max_memory, verbose)
        else:
            return self.nr_uks(cell, grids, xc_code, dms, hermi,
                               kpts, kpts_band, max_memory, verbose)

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, hermi=1, kpts=None, kpts_band=None,
               max_memory=2000, verbose=None, precomputed_ao = None, kpt = None):
        if kpts is None:
            if not kpt is None:
                sys.stderr.write('WARN: _KNumInt.nr_rks function finds keyword '
                                 'argument "kpt" and converts it to "kpts"\n')
                kpts = kpt
            else:
                kpts = numpy.zeros((1,3))
        kpts = kpts.reshape(-1,3)

        return nr_rks(self, cell, grids, xc_code, dms, 0, 0,
                      hermi, kpts, kpts_band, max_memory, verbose, precomputed_ao)

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, hermi=1, kpts=None, kpts_band=None,
               max_memory=2000, verbose=None, precomputed_ao = None, kpt = None):
        if kpts is None:
            if not kpt is None:
                sys.stderr.write('WARN: _KNumInt.nr_uks function finds keyword '
                                 'argument "kpt" and converts it to "kpts"\n')
                kpts = kpt
            else:
                kpts = numpy.zeros((1,3))
        kpts = kpts.reshape(-1,3)

        return nr_uks(self, cell, grids, xc_code, dms, 1, 0,
                      hermi, kpts, kpts_band, max_memory, verbose, precomputed_ao)

    def eval_mat(self, cell, ao_kpts, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        nkpts = len(ao_kpts)
        mat = [eval_mat(cell, ao_kpts[k], weight, rho, vxc,
                        non0tab, xctype, spin, verbose)
               for k in range(nkpts)]
        return lib.asarray(mat)

    block_loop = block_loop
    
    def _gen_rho_evaluator(self, cell, dms, hermi=1):
        if isinstance(dms, numpy.ndarray) and dms.ndim == 3:
            nao = dms.shape[-1]
            dms = [dms]
        else:
            nao = dms[0].shape[-1]
        ndms = len(dms)
        def make_rho(idm, ao_kpts, non0tab, xctype):
            return self.eval_rho(cell, ao_kpts, dms[idm], non0tab, xctype)
        return make_rho, ndms, nao

    large_rho_indices = large_rho_indices


def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)
