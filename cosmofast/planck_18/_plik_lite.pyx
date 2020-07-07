cimport cython
from libc.stdlib cimport malloc, free
#from libc.math cimport log
#from cython.parallel import prange

__all__ = ['_get_binned_cls', '_plik_lite_f', '_plik_lite_j', '_plik_lite_fj']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _get_binned_cls(const double[::1] cls, double[::1] cls_b,
                    const double[::1] weight, const int[::1] low,
                    const int[::1] width, const size_t n_bin):
    cdef size_t i, j
    for i in range(n_bin):
        cls_b[i] = 0
        for j in range(low[i], low[i] + width[i]):
            cls_b[i] += cls[j] * weight[j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _plik_lite_f(const double[::1] cls_bd, const double ap, double[::1] out_f,
                 const double[::1] mu, const size_t n_bin):
    cdef double tmp
    cdef double ap2 = ap * ap
    cdef size_t i
    out_f[0] = 0.
    for i in range(n_bin):
        tmp = cls_bd[i] / ap2
        tmp -= mu[i]
        out_f[0] -= tmp * tmp
    out_f[0] *= 0.5


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _plik_lite_j(const double[::1] cls_bd, const double ap,
                 double[:, ::1] out_j, const double[::1] mu,
                 const size_t n_bin):
    cdef double ap2 = ap * ap
    cdef size_t i
    for i in range(n_bin):
        out_j[0, i] = -cls_bd[i] / ap2
        out_j[0, i] += mu[i]
    out_j[0, n_bin] = 0.
    for i in range(n_bin):
        out_j[0, i] /= ap2
        out_j[0, n_bin] -= 2 * cls_bd[i] / ap * out_j[0, i]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _plik_lite_fj(const double[::1] cls_bd, const double ap, double[::1] out_f,
                  double[:, ::1] out_j, const double[::1] mu,
                  const size_t n_bin):
    cdef double ap2 = ap * ap
    cdef size_t i
    out_f[0] = 0.
    for i in range(n_bin):
        out_j[0, i] = -cls_bd[i] / ap2
        out_j[0, i] += mu[i]
        out_f[0] -= out_j[0, i] * out_j[0, i]
    out_f[0] *= 0.5
    out_j[0, n_bin] = 0.
    for i in range(n_bin):
        out_j[0, i] /= ap2
        out_j[0, n_bin] -= 2 * cls_bd[i] / ap * out_j[0, i]
