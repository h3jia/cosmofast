import numpy as np
import jax.numpy as jnp
import cosmofast as cf
import os


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
ft = np.load(os.path.join(CURRENT_PATH, 'test_smica.npz'))
ft_cm = np.load(os.path.join(CURRENT_PATH, 'test_smica_cmb_marged.npz'))


def test_smica():
    cl_l, cl_c = cf.planck_18.smica_get_binned(ft['test_param'][:-1], False)
    ap = jnp.asarray(ft['test_param'][-1])
    f = cf.planck_18.smica(cl_l, cl_c, ap)
    assert np.isclose(float(f), float(ft['test_value']), atol=1e-3)


def test_smica_cmb_marged():
    cl_l = cf.planck_18.smica_get_binned(ft_cm['test_param'][:-1], True)
    f = cf.planck_18.smica_cmb_marged(cl_l)
    assert np.isclose(float(f), float(ft_cm['test_value']), atol=1e-3)
