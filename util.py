import numpy as np
import numba as nb
import scipy.optimize as so


@nb.jit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :]), nopython=True, nogil=True)
def fast_symmetric_convolve(input, kernel):
    result = np.empty_like(input, dtype=np.float64)*np.nan
    sf_y, sf_x = (kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2
    for i in range(sf_y, input.shape[0]-sf_y):
        for j in range(sf_x, input.shape[1]-sf_x):
            result[i, j] = 0
            for m in range(-sf_y, sf_y+1):
                for n in range(-sf_x, sf_x+1):
                    result[i, j] += input[i+m, j+n]*kernel[m+sf_y, n+sf_x]

    return result


# @nb.jit(nb.float64(nb.float64, nb.float64, nb.float64), nopython=True, nogil=True)
@nb.jit(nopython=True, nogil=True)
def line(x, a, b):
    return a*x+b


# @nb.jit(nb.int64(nb.float64[:]), nopython=True, nogil=True)
# @nb.jit(nopython=True, nogil=True)
def arg_first_not_nan(arr):
    i = np.argwhere(np.logical_not(np.isnan(arr)))
    if i.shape[0] == 0:
        raise ValueError("Array is only filled with nan values")
    else:
        return i[0][0]


# @nb.jit(nb.int64(nb.float64[:]), nopython=True, nogil=True)
# @nb.jit(nopython=True, nogil=True)
def arg_last_non_nan(arr):
    return arr.shape[0] - 1 - arg_first_not_nan(np.flip(arr, 0))


# nb_argwhere and nb_where are potentially faster than numpy's equivalent functions, but have not been validated.
# Use with caution.
@nb.jit(nopython=True, nogil=True)
def nb_argwhere(arr):
    is_nonzero = nb_where(arr)
    ret = np.empty((1, np.sum(is_nonzero)))
    j = 0
    for i in range(arr.shape[0]):
        if is_nonzero[i]:
            ret[j] = i
            j += 1
    return ret


@nb.jit(nopython=True, nogil=True)
def nb_where(arr):
    ret = np.empty_like(arr)
    for i in range(arr.shape[0]):
        ret[i] = True if arr[i] != 0 else False
    return ret


@nb.jit(nopython=True, nogil=True)
def hhr_to_hchb(h, hr):
    return 0.5*(h-hr), 0.5*(h+hr)


@nb.jit(nopython=True, nogil=True)
def compute_forc_sg(m, sf, step_x, step_y):
    kernel = sg_kernel(sf, step_x, step_y)
    # return snf.convolve(m, kernel, mode='constant', cval=np.nan)
    return -0.5*fast_symmetric_convolve(m, kernel)


@nb.jit(nopython=True, nogil=True)
def sg_kernel(sf, step_x, step_y):

    xx, yy = np.meshgrid(np.linspace(sf*step_x, -sf*step_x, 2*sf+1),
                         np.linspace(sf*step_y, -sf*step_y, 2*sf+1))

    xx = np.reshape(xx, (-1, 1))
    yy = np.reshape(yy, (-1, 1))

    coefficients = np.linalg.pinv(np.hstack((np.ones_like(xx), xx, xx**2, yy, yy**2, xx*yy)))

    kernel = np.reshape(coefficients[5, :], (2*sf+1, 2*sf+1))

    return kernel


@nb.jit(nopython=True, nogil=True)
def extend_flat(h, m):
    for i in range(m.shape[0]):
        first_data_index = arg_first_not_nan(m[i])
        m[i, 0:first_data_index] = m[i, first_data_index]
    return


@nb.jit(nopython=True, nogil=True)
def extend_slope(h, m, n_fit_points=10):
    for i in range(m.shape[0]):
        j = arg_first_not_nan(m[i])
        popt, _ = so.curve_fit(line, h[i, j:j+n_fit_points], m[i, j:j+n_fit_points])
        m[i, 0:j] = line(h[i, 0:j], *popt)

    return
