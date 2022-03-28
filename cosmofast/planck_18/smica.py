import numpy as np
import jax.numpy as jnp
from jax import jit
from ._smica import _get_binned_cls
import os

__all__ = ['smica_get_binned', 'smica', 'smica_cmb_marged']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
f = np.load(os.path.join(CURRENT_PATH, 'data/smica.npz'))
f_cm = np.load(os.path.join(CURRENT_PATH, 'data/smica_cmb_marged.npz'))

siginv_low = np.linalg.cholesky(f['siginv'])
siginv_low_cm = np.linalg.cholesky(f_cm['siginv'])
mu = jnp.asarray(f['mu'] @ siginv_low)
mu_cm = jnp.asarray(f_cm['mu'] @ siginv_low_cm)


def smica_get_binned(cl, cmb_marged=False):
    cl = np.ascontiguousarray(cl)
    f_now = f_cm if cmb_marged else f
    siginv_low_now = siginv_low_cm if cmb_marged else siginv_low
    cl_l = np.empty(f_now['n_bin'])
    cl_c = np.empty(f_now['n_bin'])
    _get_binned_cls(cl, cl_l, cl_c, f_now['F'], f_now['n_cmb'], f_now['l_min'], f_now['l_max'],
                    f_now['n_bin'])
    if cmb_marged:
        return jnp.asarray(cl_l @ siginv_low_now)
    else:
        return jnp.asarray(cl_l @ siginv_low_now), jnp.asarray(cl_c @ siginv_low_now)


@jit
def smica(cl_l, cl_c, ap):
    return -0.5 * jnp.sum((cl_l + cl_c / ap**2 - mu)**2)


@jit
def smica_cmb_marged(cl_l):
    return -0.5 * jnp.sum((cl_l - mu_cm)**2)
