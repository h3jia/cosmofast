import numpy as np
import jax.numpy as jnp
from jax import jit
from ._plik_lite import _get_binned_cls
from functools import partial
import os

__all__ = ['plik_lite_get_binned', 'plik_lite']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
f_tt = np.load(os.path.join(CURRENT_PATH, 'data/plik_lite_tt.npz'))
f_ttteee = np.load(os.path.join(CURRENT_PATH, 'data/plik_lite_ttteee.npz'))


def plik_lite_get_binned(cl, kind='TT'):
    cl = np.ascontiguousarray(cl)
    if kind == 'TT':
        f_now = f_tt
        cl_b = np.empty(f_now['n_bin'])
        _get_binned_cls(cl[30:2509], cl_b, f_now['weight_t'], f_now['low_t'], f_now['width_t'],
                        f_now['n_bin_t'])
    elif kind == 'TTTEEE':
        f_now = f_ttteee
        cl_b = np.empty(f_now['n_bin'])
        _get_binned_cls(cl[30:2509], cl_b[f_now['i_bin'][0]:f_now['i_bin'][1]], f_now['weight_t'],
                        f_now['low_t'], f_now['width_t'], f_now['n_bin_t'])
        _get_binned_cls(cl[(2509+2509+30):(2509+2509+2509)],
                        cl_b[f_now['i_bin'][1]:f_now['i_bin'][2]], f_now['weight_e'],
                        f_now['low_e'], f_now['width_e'], f_now['n_bin_e'])
        _get_binned_cls(cl[(2509+30):(2509+2509)], cl_b[f_now['i_bin'][2]:f_now['i_bin'][3]],
                        f_now['weight_e'], f_now['low_e'], f_now['width_e'], f_now['n_bin_e'])
    else:
        raise ValueError('unexpected value kind = {}, which should be "TT" or '
                         '"TTTEEE".'.format(kind))
    cl_b = jnp.asarray(cl_b @ f_now['cov_inv_low'])
    return cl_b


@partial(jit, static_argnums=(2,))
def plik_lite(cl, ap, kind='TT'):
    if kind == 'TT':
        return -0.5 * jnp.sum((cl / ap**2 - f_tt['mu'])**2)
    elif kind == 'TTTEEE':
        return -0.5 * jnp.sum((cl / ap**2 - f_ttteee['mu'])**2)
    else:
        raise ValueError('unexpected value kind = {}, which should be "TT" or '
                         '"TTTEEE".'.format(kind))
