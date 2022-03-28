import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
import os

__all__ = ['commander']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
f = jnp.load(os.path.join(CURRENT_PATH, 'data/commander.npz'))
cl2x = jnp.asarray(f['cl2x'])
mu = jnp.asarray(f['mu'])
cov_inv = jnp.asarray(np.linalg.inv(f['cov']))
offset = jnp.asarray(f['offset'])

l_min = 2
l_max = 29
n_b = 1000


@jit
def commander(cl, ap):
    ap2 = ap**2
    out = 0.
    y0 = -mu
    for i in range(l_min, l_max + 1):
        power = cl[i - l_min] * i * (i + 1) / 2. / jnp.pi / ap2
        j = jnp.searchsorted(cl2x[i - l_min, 0], power, side='right').astype(int) - 1
        h = cl2x[i - l_min, 0, j + 1] - cl2x[i - l_min, 0, j]
        a = (cl2x[i - l_min, 0, j + 1] - power) / h
        b = (power - cl2x[i - l_min, 0, j]) / h
        ya = cl2x[i - l_min, 1, j]
        yb = cl2x[i - l_min, 1, j + 1]
        y2a = cl2x[i - l_min, 2, j]
        y2b = cl2x[i - l_min, 2, j + 1]
        y0 = y0.at[i - l_min].add(a * ya + b * yb +
                                  h**2 / 6. * ((a**3 - a) * y2a + (b**3 - b) * y2b))
        y1 = (yb - ya) / h + (y2b * (3 * b**2 - 1) - y2a * (3 * a**2 - 1)) * h / 6.
        out += cond(jnp.logical_and(j >= 0, j < n_b - 1), lambda: jnp.log(y1), lambda: jnp.nan)
    out -= 0.5 * y0 @ cov_inv @ y0
    return out - offset
