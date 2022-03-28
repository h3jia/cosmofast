import numpy as np
import jax.numpy as jnp
import cosmofast as cf
import os


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
ft = np.load(os.path.join(CURRENT_PATH, 'test_commander.npz'))


def test_commander():
    cl = jnp.asarray(ft['test_param'][2:-1])
    ap = jnp.asarray(ft['test_param'][-1])
    f = cf.planck_18.commander(cl, ap)
    assert np.isclose(float(f), float(ft['test_value']), atol=1e-3)
