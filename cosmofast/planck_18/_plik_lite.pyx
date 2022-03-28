cimport cython
# from libc.stdlib cimport malloc, free
# from libc.math cimport log
# from cython.parallel import prange

__all__ = ['_get_binned_cls']


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
