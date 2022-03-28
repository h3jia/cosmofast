import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from functools import partial
import os

__all__ = ['simall']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
f_ee = jnp.load(os.path.join(CURRENT_PATH, 'data/simall_ee.npz'))
c_ee = jnp.asarray(f_ee['c'])
step_ee = jnp.asarray(f_ee['step'])
n_step_ee = jnp.asarray(f_ee['n_step'])

f_bb = jnp.load(os.path.join(CURRENT_PATH, 'data/simall_bb.npz'))
c_bb = jnp.asarray(f_bb['c'])
step_bb = jnp.asarray(f_bb['step'])
n_step_bb = jnp.asarray(f_bb['n_step'])

l_min = 2
l_max = 29


@jit
def _evaluate_cubic(x, c):
    return c[0] * x**3 + c[1] * x**2 + c[2] * x + c[3]


@partial(jit, static_argnums=(2,))
def simall(cl, ap, kind='EE'):
    if kind == 'EE':
        f = f_ee
        c = c_ee
        step = step_ee
        n_step = n_step_ee
    elif kind == 'BB':
        f = f_bb
        c = c_bb
        step = step_bb
        n_step = n_step_bb
    else:
        raise ValueError('unexpected value kind = {}, which should be "EE" or "BB".'.format(kind))

    ap2 = ap**2
    out = 0.
    for i in range(l_min, l_max + 1):
        power = cl[i - l_min] * i * (i + 1) / 2. / jnp.pi / ap2
        k = (power / step).astype(int)
        out += cond(jnp.logical_and(cl[i] >= 0, k < n_step - 1), _evaluate_cubic,
                    lambda x, c: jnp.nan, power - k * step, c[i - l_min, k])
    return out
