cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport log
from cython.parallel import prange


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _smica_f(const double[::1] cls, const double ap, double[::1] out_f,
             const double[:, ::1] F, const double[::1] mu,
             const double[:, ::1] siginv, size_t n_cmb, size_t lmin=2,
             size_t lmax=2500, size_t n_b=9):
    cdef double *y0 = <double *> malloc(n_b * sizeof(double))
    cdef double ap2 = ap * ap
    cdef size_t i, j, k
    if not y0:
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_b):
            y0[i] = 0.
            for j in prange(lmin, lmax + 1, nogil=True, schedule='static'):
                y0[i] += F[i, j] * cls[j]
            for k in range(n_cmb):
                for j in prange(
                    lmin + k * (lmax + 1), (k + 1) * (lmax + 1), nogil=True, 
                    schedule='static'):
                    y0[i] += F[i, j] * cls[j] / ap2
            y0[i] -= mu[i]
        for i in range(n_b):
            out_f[0] = 2. * siginv[i, i] * y0[i] * y0[i]
            for j in range(i + 1, n_b):
                out_f[0] += siginv[i, i] * y0[i] * y0[j]
        out_f[0] *= -0.5
    finally:
        free(y0)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _smica_j(const double[::1] cls, const double ap, double[:, ::1] out_j,
             const double[:, ::1] F, const double[::1] mu,
             const double[:, ::1] siginv, size_t n_cmb, size_t lmin=2,
             size_t lmax=2500, size_t n_b=9):
    cdef double *y0 = <double *> malloc(n_b * sizeof(double))
    cdef double *y1 = <double *> malloc(n_b * sizeof(double))
    cdef double ap2 = ap * ap
    cdef double t = 0.
    cdef size_t i, j, k, l
    l = (n_cmb + 1) * (lmax + 1)
    if not (y0 and y1):
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_b):
            y0[i] = 0.
            for j in prange(lmin, lmax + 1, nogil=True, schedule='static'):
                y0[i] += F[i, j] * cls[j]
            for k in range(1, n_cmb + 1):
                for j in prange(
                    lmin + k * (lmax + 1), (k + 1) * (lmax + 1), nogil=True, 
                    schedule='static'):
                    y0[i] += F[i, j] * cls[j] / ap2
            y0[i] -= mu[i]
        for i in range(n_b):
            y1[i] = 0
            for j in range(n_b):
                y1[i] += siginv[i, j] * y0[j]
        for i in range(l + 1):
            out_j[0, i] = 0.
        for i in prange(lmin, lmax + 1, nogil=True, schedule='static'):
            for j in range(n_b):
                out_j[0, i] -= F[j, i] * y1[j]
        for i in range(1, n_cmb + 1):
            for j in prange(
                lmin + i * (lmax + 1), (i + 1) * (lmax + 1), nogil=True, 
                schedule='static'):
                for k in range(n_b):
                    out_j[0, j] -= F[k, j] * y1[k]
                out_j[0, j] /= ap2
        for i in prange(lmax + 1, l, nogil=True, schedule='static'):
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
              size_t lmin=2, size_t lmax=2500, size_t n_b=9):
    cdef double *y0 = <double *> malloc(n_b * sizeof(double))
    cdef double *y1 = <double *> malloc(n_b * sizeof(double))
    cdef double ap2 = ap * ap
    cdef double t = 0.
    cdef size_t i, j, k, l
    l = (n_cmb + 1) * (lmax + 1)
    if not (y0 and y1):
        raise MemoryError('cannot malloc required array in _smica_f.')
    try:
        for i in range(n_b):
            y0[i] = 0.
            for j in prange(lmin, lmax + 1, nogil=True, schedule='static'):
                y0[i] += F[i, j] * cls[j]
            for k in range(1, n_cmb + 1):
                for j in prange(
                    lmin + k * (lmax + 1), (k + 1) * (lmax + 1), nogil=True, 
                    schedule='static'):
                    y0[i] += F[i, j] * cls[j] / ap2
            y0[i] -= mu[i]
        for i in range(n_b):
            y1[i] = 0
            for j in range(n_b):
                y1[i] += siginv[i, j] * y0[j]
        out_f[0] = 0
        for i in range(l + 1):
            out_j[0, i] = 0.
        for i in range(n_b):
            out_f[0] += y0[i] * y1[i]
        out_f[0] *= -0.5
        for i in prange(lmin, lmax + 1, nogil=True, schedule='static'):
            for j in range(n_b):
                out_j[0, i] -= F[j, i] * y1[j]
        for i in range(1, n_cmb + 1):
            for j in prange(
                lmin + i * (lmax + 1), (i + 1) * (lmax + 1), nogil=True, 
                schedule='static'):
                for k in range(n_b):
                    out_j[0, j] -= F[k, j] * y1[k]
                out_j[0, j] /= ap2
        for i in prange(lmax + 1, l, nogil=True, schedule='static'):
            t -= 2 * cls[i] / ap * out_j[0, i]
        out_j[0, l] = t
    finally:
        free(y0)
        free(y1)
