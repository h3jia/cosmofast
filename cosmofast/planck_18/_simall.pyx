cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport log, pi
cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"
import numpy as np
cimport numpy as np

__all__ = ['_simall_f', '_simall_j', '_simall_fj']

# NOTE: here the result can be a bit different (dlogp up to ~0.05) 
#       from the original Planck implementation
#       since we are doing cubic spline for the table
#       while the original implementation just round to the lower node


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _evaluate(const double* c, double x) nogil:
    return c[0] * x * x * x + c[1] * x * x + c[2] * x + c[3]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _derivative(const double* c, double x) nogil:
    return 3 * c[0] * x * x + 2 * c[1] * x + c[2]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _simall_f(const double[::1] cls, const double ap, double[::1] out_f,
              const double[:, :, ::1] c, const double step, const size_t n_step,
              size_t l_min=2, size_t l_max=29):
    cdef double power, tmp
    cdef double ap2 = ap * ap
    cdef size_t i, j, k
    cdef size_t n_l = l_max - l_min + 1
    out_f[0] = 0.
    for i in range(n_l):
        j = i + l_min
        tmp = j * (j + 1) / 2 / pi / ap2
        power = cls[i] * tmp
        k = <size_t>(power / step)
        if cls[i] < 0. or k >= n_step - 1:
            out_f[0] = nan
            return
        out_f[0] += _evaluate(&c[i, k, 0], power - k * step)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _simall_j(const double[::1] cls, const double ap, double[:, ::1] out_j,
              const double[:, :, ::1] c, const double step, const size_t n_step,
              size_t l_min=2, size_t l_max=29):
    cdef double power, tmp
    cdef double ap2 = ap * ap
    cdef size_t i, j, k
    cdef size_t n_l = l_max - l_min + 1
    out_j[0, n_l] = 0.
    for i in range(n_l):
        j = i + l_min
        tmp = j * (j + 1) / 2 / pi / ap2
        power = cls[i] * tmp
        k = <size_t>(power / step)
        if cls[i] < 0. or k >= n_step - 1:
            for i in range(n_l + 1):
                out_j[0, i] = nan
            return
        out_j[0, i] = _derivative(&c[i, k, 0], power - k * step) * tmp
        out_j[0, n_l] -= 2 * cls[i] / ap * out_j[0, i]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _simall_fj(const double[::1] cls, const double ap, double[::1] out_f,
               double[:, ::1] out_j, const double[:, :, ::1] c,
               const double step, const size_t n_step, size_t l_min=2,
               size_t l_max=29):
    cdef double power
    cdef double ap2 = ap * ap
    cdef size_t i, j, k
    cdef size_t n_l = l_max - l_min + 1
    out_f[0] = 0.
    out_j[0, n_l] = 0.
    for i in range(n_l):
        j = i + l_min
        tmp = j * (j + 1) / 2 / pi / ap2
        power = cls[i] * tmp
        k = <size_t>(power / step)
        if cls[i] < 0. or k >= n_step - 1:
            out_f[0] = nan
            for i in range(n_l + 1):
                out_j[0, i] = nan
            return
        out_f[0] += _evaluate(&c[i, k, 0], power - k * step)
        out_j[0, i] = _derivative(&c[i, k, 0], power - k * step) * tmp
        out_j[0, n_l] -= 2 * cls[i] / ap * out_j[0, i]
