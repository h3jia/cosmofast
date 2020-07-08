cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport log
from cython.parallel import prange

__all__ = ['_smica_f', '_smica_j', '_smica_fj']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _smica_f(const double[::1] cls, const double ap, double[::1] out_f,
             const double[:, ::1] F, const double[::1] mu,
             const double[:, ::1] siginv, size_t n_cmb, size_t l_min=2,
             size_t l_max=2500, size_t n_bin=9):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i, j, k
    if not y0:
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_bin):
            y0[i] = 0.
            for j in prange(l_min, l_max + 1, nogil=True, schedule='static'):
                y0[i] += F[i, j] * cls[j]
            for k in range(1, n_cmb + 1):
                for j in prange(
                    l_min + k * (l_max + 1), (k + 1) * (l_max + 1), nogil=True, 
                    schedule='static'):
                    y0[i] += F[i, j] * cls[j] / ap2
            y0[i] -= mu[i]
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
def _smica_j(const double[::1] cls, const double ap, double[:, ::1] out_j,
             const double[:, ::1] F, const double[::1] mu,
             const double[:, ::1] siginv, size_t n_cmb, size_t l_min=2,
             size_t l_max=2500, size_t n_bin=9):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double *y1 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef double t = 0.
    cdef size_t i, j, k, l
    l = (n_cmb + 1) * (l_max + 1)
    if not (y0 and y1):
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_bin):
            y0[i] = 0.
            for j in prange(l_min, l_max + 1, nogil=True, schedule='static'):
                y0[i] += F[i, j] * cls[j]
            for k in range(1, n_cmb + 1):
                for j in prange(
                    l_min + k * (l_max + 1), (k + 1) * (l_max + 1), nogil=True, 
                    schedule='static'):
                    y0[i] += F[i, j] * cls[j] / ap2
            y0[i] -= mu[i]
        for i in range(n_bin):
            y1[i] = 0
            for j in range(n_bin):
                y1[i] += siginv[i, j] * y0[j]
        for i in range(l + 1):
            out_j[0, i] = 0.
        for i in prange(l_min, l_max + 1, nogil=True, schedule='static'):
            for j in range(n_bin):
                out_j[0, i] -= F[j, i] * y1[j]
        for i in range(1, n_cmb + 1):
            for j in prange(
                l_min + i * (l_max + 1), (i + 1) * (l_max + 1), nogil=True, 
                schedule='static'):
                for k in range(n_bin):
                    out_j[0, j] -= F[k, j] * y1[k]
                out_j[0, j] /= ap2
        for i in prange(l_max + 1, l, nogil=True, schedule='static'):
            t -= 2 * cls[i] / ap * out_j[0, i]
        out_j[0, l] = t
    finally:
        free(y0)
        free(y1)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _smica_fj(const double[::1] cls, const double ap, double[::1] out_f,
              double[:, ::1] out_j, const double[:, ::1] F,
              const double[::1] mu, const double[:, ::1] siginv, size_t n_cmb,
              size_t l_min=2, size_t l_max=2500, size_t n_bin=9):
    cdef double *y0 = <double *> malloc(n_bin * sizeof(double))
    cdef double *y1 = <double *> malloc(n_bin * sizeof(double))
    cdef double ap2 = ap * ap
    cdef double t = 0.
    cdef size_t i, j, k, l
    l = (n_cmb + 1) * (l_max + 1)
    if not (y0 and y1):
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_bin):
            y0[i] = 0.
            for j in prange(l_min, l_max + 1, nogil=True, schedule='static'):
                y0[i] += F[i, j] * cls[j]
            for k in range(1, n_cmb + 1):
                for j in prange(
                    l_min + k * (l_max + 1), (k + 1) * (l_max + 1), nogil=True, 
                    schedule='static'):
                    y0[i] += F[i, j] * cls[j] / ap2
            y0[i] -= mu[i]
        for i in range(n_bin):
            y1[i] = 0
            for j in range(n_bin):
                y1[i] += siginv[i, j] * y0[j]
        out_f[0] = 0
        for i in range(l + 1):
            out_j[0, i] = 0.
        for i in range(n_bin):
            out_f[0] += y0[i] * y1[i]
        out_f[0] *= -0.5
        for i in prange(l_min, l_max + 1, nogil=True, schedule='static'):
            for j in range(n_bin):
                out_j[0, i] -= F[j, i] * y1[j]
        for i in range(1, n_cmb + 1):
            for j in prange(
                l_min + i * (l_max + 1), (i + 1) * (l_max + 1), nogil=True, 
                schedule='static'):
                for k in range(n_bin):
                    out_j[0, j] -= F[k, j] * y1[k]
                out_j[0, j] /= ap2
        for i in prange(l_max + 1, l, nogil=True, schedule='static'):
            t -= 2 * cls[i] / ap * out_j[0, i]
        out_j[0, l] = t
    finally:
        free(y0)
        free(y1)
