cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport log, pi
cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"

__all__ = ['_commander_f', '_commander_j', '_commander_fj']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _find_interval(const double* x_s, const int m, const double xval,
                        int prev_interval=-1) nogil:
    cdef int high, low, mid, interval
    interval = prev_interval
    if interval < 0 or interval >= m:
        interval = (m - 1) // 2
    if not (x_s[0] <= xval < x_s[m - 1]):
        interval = -1
    else:
        # Find the interval the coordinate is in
        # x[0] i[0] x[1] ... x[m-2] i[m-2] x[m-1]
        # (binary search with locality)
        if xval >= x_s[interval]:
            low = interval
            high = m - 2
        else:
            low = 0
            high = interval - 1
        if xval < x_s[low + 1]:
            high = low
        while low < high:
            mid = (high + low) // 2
            if xval < x_s[mid]:
                high = mid - 1
            elif xval >= x_s[mid + 1]:
                low = mid + 1
            else:
                # x_s[mid] <= xval < x_s[mid+1]
                low = mid
                break
        interval = low
    return interval


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _quad_sym(const double[:, ::1] A, const double* x,
                      const int m) nogil:
    cdef size_t i, j
    cdef double xAx
    xAx = 0.
    for i in range(m):
        xAx += A[i, i] * x[i] * x[i]
        for j in range(i + 1, m):
            xAx += 2. * A[i, j] * x[i] * x[j]
    return xAx


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _mat_vec(const double[:, ::1] A, const double* x, double* Ax,
                   const int m, double alpha=1.) nogil:
    cdef size_t i, j
    for i in range(m):
        Ax[i] = 0.
        for j in range(m):
            Ax[i] += A[i, j] * x[j]
        Ax[i] *= alpha


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _cubic_01(const double xval, const double* x_s, const double* y_s,
                   const double* y2_s, double* y0, double* y1,
                   const int n_b) nogil:
    cdef int i
    cdef double a, b, h
    i = _find_interval(x_s, n_b, xval)
    if i < 0:
        return i
    # TODO: we can probably rewrite the cubic spline as a more efficient form
    h = x_s[i + 1] - x_s[i]
    a = (x_s[i + 1] - xval) / h
    b = (xval - x_s[i]) / h
    y0[0] = (a * y_s[i] + b * y_s[i + 1] + h * h / 6. * (a * (a * a - 1.) * 
             y2_s[i] + b * (b * b - 1.) * y2_s[i + 1]))
    y1[0] = ((y_s[i + 1] - y_s[i]) / h - h * y2_s[i] / 6. * (3. * a * a - 1.) + 
             h * y2_s[i + 1] / 6. * (3. * b * b - 1.))
    return i


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _cubic_012(const double xval, const double* x_s, const double* y_s,
                    const double* y2_s, double* y0, double* y1, double* y2,
                    const int n_b) nogil:
    cdef int i
    cdef double a, b, h
    i = _find_interval(x_s, n_b, xval)
    if i < 0:
        return i
    # TODO: we can probably rewrite the cubic spline as a more efficient form
    h = x_s[i + 1] - x_s[i]
    a = (x_s[i + 1] - xval) / h
    b = (xval - x_s[i]) / h
    y0[0] = (a * y_s[i] + b * y_s[i + 1] + h * h / 6. * (a * (a * a - 1.) * 
             y2_s[i] + b * (b * b - 1.) * y2_s[i + 1]))
    y1[0] = ((y_s[i + 1] - y_s[i]) / h - h * y2_s[i] / 6. * (3. * a * a - 1.) + 
             h * y2_s[i + 1] / 6. * (3. * b * b - 1.))
    y2[0] = y2_s[i] * a + y2_s[i + 1] * b
    return i


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _commander_f(const double[::1] cl, const double ap, double[::1] out_f,
                 const double[:, :, ::1] cl2x, const double[::1] mu,
                 const double[:, ::1] cov_inv, size_t l_min=2, size_t l_max=29,
                 size_t n_b=1000):
    # cl2x : (n_l, 3, n_b)
    # 2 <= l_min < l_max <= 250
    cdef size_t n_l = l_max - l_min + 1
    cdef double *y0 = <double *> malloc(n_l * sizeof(double))
    cdef double *y1 = <double *> malloc(n_l * sizeof(double))
    cdef double power, tmp
    cdef double ap2 = ap * ap
    cdef size_t i, j
    cdef int interval
    if not (y0 and y1):
        raise MemoryError('cannot malloc required array in _commander_f.')
    try:
        for i in range(n_l):
            j = i + l_min
            tmp = j * (j + 1) / 2. / pi / ap2
            power = cl[i] * tmp
            interval = _cubic_01(
                power, &cl2x[i, 0, 0], &cl2x[i, 1, 0], &cl2x[i, 2, 0], &y0[i], 
                &y1[i], n_b)
            if interval < 0:
                out_f[0] = nan
                return
            y0[i] -= mu[i]
        out_f[0] = -0.5 * _quad_sym(cov_inv, y0, n_l)
        for i in range(n_l):
            out_f[0] += log(y1[i])
    finally:
        free(y0)
        free(y1)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _commander_j(const double[::1] cl, const double ap, double[:, ::1] out_j,
                 const double[:, :, ::1] cl2x, const double[::1] mu,
                 const double[:, ::1] cov_inv, size_t l_min=2, size_t l_max=29,
                 size_t n_b=1000):
    cdef size_t n_l = l_max - l_min + 1
    cdef double *y0 = <double *> malloc(n_l * sizeof(double))
    cdef double *y1 = <double *> malloc(n_l * sizeof(double))
    cdef double *y2 = <double *> malloc(n_l * sizeof(double))
    cdef double *tmp = <double *> malloc(n_l * sizeof(double))
    cdef double power
    cdef double ap2 = ap * ap
    cdef size_t i, j
    cdef int interval
    if not (y0 and y1 and y2):
        raise MemoryError('cannot malloc required array in _commander_j.')
    try:
        for i in range(n_l):
            j = i + l_min
            tmp[i] = j * (j + 1) / 2. / pi / ap2
            power = cl[i] * tmp[i]
            interval = _cubic_012(
                power, &cl2x[i, 0, 0], &cl2x[i, 1, 0], &cl2x[i, 2, 0], &y0[i], 
                &y1[i], &y2[i], n_b)
            if interval < 0:
                for i in range(n_l + 1):
                    out_j[0, i] = nan
                return
            y0[i] -= mu[i]
        _mat_vec(cov_inv, y0, &out_j[0, 0], n_l, -1.)
        for i in range(n_l):
            out_j[0, i] *= y1[i]
            out_j[0, i] += y2[i] / y1[i]
            out_j[0, i] *= tmp[i]
        out_j[0, n_l] = 0
        for i in range(n_l):
            out_j[0, n_l] -= 2. * cl[i] / ap * out_j[0, i]
    finally:
        free(y0)
        free(y1)
        free(y2)
        free(tmp)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _commander_fj(const double[::1] cl, const double ap, double[::1] out_f,
                  double[:, ::1] out_j, const double[:, :, ::1] cl2x,
                  const double[::1] mu, const double[:, ::1] cov_inv,
                  size_t l_min=2, size_t l_max=29, size_t n_b=1000):
    cdef size_t n_l = l_max - l_min + 1
    cdef double *y0 = <double *> malloc(n_l * sizeof(double))
    cdef double *y1 = <double *> malloc(n_l * sizeof(double))
    cdef double *y2 = <double *> malloc(n_l * sizeof(double))
    cdef double *tmp = <double *> malloc(n_l * sizeof(double))
    cdef double power
    cdef double ap2 = ap * ap
    cdef size_t i, j
    cdef int interval
    if not (y0 and y1 and y2 and tmp):
        raise MemoryError('cannot malloc required array in _commander_fj.')
    try:
        for i in range(n_l):
            j = i + l_min
            tmp[i] = j * (j + 1) / 2. / pi / ap2
            power = cl[i] * tmp[i]
            interval = _cubic_012(
                power, &cl2x[i, 0, 0], &cl2x[i, 1, 0], &cl2x[i, 2, 0], &y0[i], 
                &y1[i], &y2[i], n_b)
            if interval < 0:
                out_f[0] = nan
                for i in range(n_l + 1):
                    out_j[0, i] = nan
                return
            y0[i] -= mu[i]
        out_f[0] = -0.5 * _quad_sym(cov_inv, y0, n_l)
        _mat_vec(cov_inv, y0, &out_j[0, 0], n_l, -1.)
        for i in range(n_l):
            out_f[0] += log(y1[i])
            out_j[0, i] *= y1[i]
            out_j[0, i] += y2[i] / y1[i]
            out_j[0, i] *= tmp[i]
        out_j[0, n_l] = 0
        for i in range(n_l):
            out_j[0, n_l] -= 2. * cl[i] / ap * out_j[0, i]
    finally:
        free(y0)
        free(y1)
        free(y2)
        free(tmp)
