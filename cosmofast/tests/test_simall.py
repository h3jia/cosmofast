import numpy as np
import jax.numpy as jnp
import cosmofast as cf
import os


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
ft_ee = np.load(os.path.join(CURRENT_PATH, 'test_simall_ee.npz'))
ft_bb = np.load(os.path.join(CURRENT_PATH, 'test_simall_bb.npz'))


def test_simall_ee():
    cl = jnp.asarray(ft_ee['test_param'][2:-1])
    ap = jnp.asarray(ft_ee['test_param'][-1])
    f = cf.planck_18.simall(cl, ap, 'EE')
    assert np.isclose(float(f), float(ft_ee['test_value']), atol=5e-2)


def test_simall_bb():
    cl = jnp.asarray(ft_bb['test_param'][2:-1])
    ap = jnp.asarray(ft_bb['test_param'][-1])
    f = cf.planck_18.simall(cl, ap, 'BB')
    assert np.isclose(float(f), float(ft_bb['test_value']), atol=5e-2)
