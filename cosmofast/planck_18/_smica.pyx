cimport cython
# from libc.stdlib cimport malloc, free
# from libc.math cimport log
# from cython.parallel import prange

__all__ = ['_get_binned_cls']


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
