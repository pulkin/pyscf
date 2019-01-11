from pyscf.pbc.tdscf.krhf_slow_supercell import k_nocc

import numpy
from numpy import testing


def retrieve_m(model, **kwargs):
    """Retrieves TDSCF matrix."""
    vind, hdiag = model.gen_vind(model._scf, **kwargs)
    size = model.init_guess(model._scf, 1).shape[1]
    return vind(numpy.eye(size)).T


def sign(x):
    return x / abs(x)


def pull_dim(a, dim):
    """Pulls the specified dimension forward and reshapes array into a 2D matrix."""
    a = a.transpose(*(
            (dim,) + tuple(range(dim)) + tuple(range(dim + 1, len(a.shape)))
    ))
    a = a.reshape(len(a), -1)
    return a


def phase_difference(a, b, axis=0, threshold=1e-5):
    """The phase difference between vectors."""
    v1, v2 = numpy.asarray(a), numpy.asarray(b)
    testing.assert_equal(v1.shape, v2.shape)
    v1, v2 = pull_dim(v1, axis), pull_dim(v2, axis)
    g1 = abs(v1) > threshold
    g2 = abs(v2) > threshold
    g12 = numpy.logical_and(g1, g2)
    if numpy.any(g12.sum(axis=1) == 0):
        desired_threshold = numpy.minimum(abs(v1), abs(v2)).max(axis=1).min()
        raise ValueError("Cannot find an anchor for the rotation, maximal value for the threshold is: {:.3e}".format(
            desired_threshold
        ))
    anchor_index = tuple(numpy.where(i)[0][0] for i in g12)
    return sign(v2[numpy.arange(len(v2)), anchor_index]) / sign(v1[numpy.arange(len(v1)), anchor_index])


def adjust_mf_phase(model1, model2, threshold=1e-5):
    """Tunes the phase of the 2 mean-field models to a common value."""
    signatures = []
    orders = []

    for m in (model1, model2):
        if "kpts" in dir(m):
            signatures.append(numpy.concatenate(m.mo_coeff, axis=1))
            orders.append(numpy.argsort(numpy.concatenate(m.mo_energy)))
        else:
            signatures.append(m.mo_coeff)
            orders.append(numpy.argsort(m.mo_energy))

    m1, m2 = signatures
    o1, o2 = orders
    mdim = min(m1.shape[0], m2.shape[0])
    m1, m2 = m1[:mdim, :][:, o1], m2[:mdim, :][:, o2]

    p = phase_difference(m1, m2, axis=1, threshold=threshold)

    if "kpts" in dir(model2):
        fr = 0
        for k, i in enumerate(model2.mo_coeff):
            to = fr + i.shape[1]
            slc = numpy.logical_and(fr <= o2, o2 < to)
            i[:, o2[slc] - fr] /= p[slc][numpy.newaxis, :]
            fr = to
    else:
        model2.mo_coeff[:, o2] /= p[numpy.newaxis, :]


def adjust_td_phase(model1, model2, threshold=1e-5):
    """Tunes the phase of the 2 time-dependent models to a common value."""
    signatures = []
    orders = []

    for m in (model1, model2):
        # Are there k-points?
        if "kpts" in dir(m._scf):
            # Is it a supercell model, Gamma model or a true k-model?
            if isinstance(m.xy, dict):
                # A true k-model, take k = 0
                raise NotImplementedError("Implement me")
            elif len(m.xy.shape) == 6:
                # A supercell model
                xy = m.xy.reshape(len(m.e), -1)
                xy = xy[:, ov_order(m._scf)]
                signatures.append(xy)
                orders.append(numpy.argsort(m.e))
            elif len(m.xy.shape) == 5:
                # Gamma model
                raise NotImplementedError("Implement me")
            else:
                raise ValueError("Unknown vectors: {}".format(repr(m.xy)))
        else:
            signatures.append(m.xy.reshape(len(m.e), -1))
            orders.append(numpy.argsort(m.e))

    m1, m2 = signatures
    o1, o2 = orders
    m1, m2 = m1[o1, :], m2[o2, :]

    p = phase_difference(m1, m2, axis=0, threshold=threshold)

    if "kpts" in dir(model2._scf):
        # Is it a supercell model, Gamma model or a true k-model?
        if isinstance(m.xy, dict):
            # A true k-model, take k = 0
            raise NotImplementedError("Implement me")
        elif len(m.xy.shape) == 6:
            # A supercell model
            model2.xy[o2, ...] /= p[(slice(None),) + (numpy.newaxis,) * 5]
        elif len(m.xy.shape) == 5:
            # Gamma model
            raise NotImplementedError("Implement me")
        else:
            raise ValueError("Unknown vectors: {}".format(repr(m.xy)))
    else:
        model2.xy[o2, ...] /= p[(slice(None),) + (numpy.newaxis,) * 5]


def remove_phase_difference(v1, v2, axis=0, threshold=1e-5):
    """Removes the phase difference between two vectors."""
    dtype = numpy.common_type(numpy.asarray(v1), numpy.asarray(v2))
    v1, v2 = numpy.array(v1, dtype=dtype), numpy.array(v2, dtype=dtype)
    v1, v2 = pull_dim(v1, axis), pull_dim(v2, axis)
    v2 /= phase_difference(v1, v2, threshold=threshold)[:, numpy.newaxis]
    return v1, v2


def assert_vectors_close(v1, v2, axis=0, threshold=1e-5, atol=1e-8):
    """Compares two vectors up to a phase difference."""
    v1, v2 = remove_phase_difference(v1, v2, axis=axis, threshold=threshold)
    delta = abs(v1 - v2).max(axis=1)
    wrong = delta > atol
    if any(wrong):
        raise AssertionError("Vectors are not close to tolerance atol={}\n\n({:d} roots mismatch)\ndelta {}".format(
            str(atol),
            sum(wrong),
            ", ".join("#{:d}: {:.3e}".format(i, delta[i]) for i in numpy.argwhere(wrong)[:, 0]),
        ))


def ov_order(model):
    nocc = k_nocc(model)
    e_occ = tuple(e[:o] for e, o in zip(model.mo_energy, nocc))
    e_virt = tuple(e[o:] for e, o in zip(model.mo_energy, nocc))
    sort_o = []
    sort_v = []
    for o in e_occ:
        for v in e_virt:
            _v, _o = numpy.meshgrid(v, o)
            sort_o.append(_o.reshape(-1))
            sort_v.append(_v.reshape(-1))
    sort_o, sort_v = numpy.concatenate(sort_o), numpy.concatenate(sort_v)
    vals = numpy.array(
        list(zip(sort_o, sort_v)),
        dtype=[('o', sort_o[0].dtype), ('v', sort_v[0].dtype)]
    )
    result = numpy.argsort(vals, order=('o', 'v'))
    # Double for other blocks
    return numpy.concatenate([result, result + len(result)])
