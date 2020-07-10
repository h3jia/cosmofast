cimport cython
from libc.stdlib cimport malloc, free
# from libc.math cimport log
# from cython.parallel import prange

__all__ = ['_get_binned_cls', '_smica_f', '_smica_j', '_smica_fj']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _get_binned_cls(const double[::1] cls, double[::1] lens_b,
                    double[::1] cmb_b, const double[:, ::1] F,
                    const size_t n_cmb, size_t l_min=2, size_t l_max=2500,
                    size_t n_bin=9):
    for i in range(n_bin):
        lens_b[i] = 0.
        for j in range(l_min, l_max + 1):
            lens_b[i] += F[i, j] * cls[j]
        if n_cmb:
            cmb_b[i] = 0.
            for j in range(1, n_cmb + 1):
                for k in range(l_min + j * (l_max + 1), (j + 1) * (l_max + 1)):
                    cmb_b[i] += F[i, k] * cls[k]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _smica_f(const double[::1] lens_b, const double[::1] cmb_b, const double ap,
             double[::1] out_f, const double[::1] mu,
             const double[:, ::1] siginv, const size_t n_cmb, size_t n_bin=9):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i, j
    if not y0:
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_bin):
            if n_cmb:
                y0[i] = lens_b[i] + cmb_b[i] / ap2 - mu[i]
            else:
                y0[i] = lens_b[i] - mu[i]
        out_f[0] = 0.
        for i in range(n_bin):
            out_f[0] += siginv[i, i] * y0[i] * y0[i]
            for j in range(i + 1, n_bin):
                out_f[0] += 2. * siginv[i, j] * y0[i] * y0[j]
        out_f[0] *= -0.5
    finally:
        free(y0)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _smica_j(const double[::1] lens_b, const double[::1] cmb_b, const double ap,
             double[:, ::1] out_j, const double[::1] mu,
             const double[:, ::1] siginv, const size_t n_cmb, size_t n_bin=9):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i, j
    if not y0:
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_bin):
            if n_cmb:
                y0[i] = lens_b[i] + cmb_b[i] / ap2 - mu[i]
            else:
                y0[i] = lens_b[i] - mu[i]
        for i in range(n_bin):
            out_j[0, i] = 0.
            for j in range(n_bin):
                out_j[0, i] -= siginv[i, j] * y0[j]
        if n_cmb:
            out_j[0, 2 * n_bin] = 0.
            for i in range(n_bin):
                out_j[0, i + n_bin] = out_j[0, i] / ap2
                out_j[0, 2 * n_bin] -= 2 * out_j[0, i] * cmb_b[i] / ap2 / ap
    finally:
        free(y0)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _smica_fj(const double[::1] lens_b, const double[::1] cmb_b, const double ap,
             double[::1] out_f, double[:, ::1] out_j, const double[::1] mu,
             const double[:, ::1] siginv, const size_t n_cmb, size_t n_bin=9):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i, j
    if not y0:
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_bin):
            if n_cmb:
                y0[i] = lens_b[i] + cmb_b[i] / ap2 - mu[i]
            else:
                y0[i] = lens_b[i] - mu[i]
        for i in range(n_bin):
            out_j[0, i] = 0.
            for j in range(n_bin):
                out_j[0, i] -= siginv[i, j] * y0[j]
        out_f[0] = 0.
        for i in range(n_bin):
            out_f[0] += out_j[0, i] * y0[i]
        out_f[0] *= 0.5
        if n_cmb:
            out_j[0, 2 * n_bin] = 0.
            for i in range(n_bin):
                out_j[0, i + n_bin] = out_j[0, i] / ap2
                out_j[0, 2 * n_bin] -= 2 * out_j[0, i] * cmb_b[i] / ap2 / ap
    finally:
        free(y0)
