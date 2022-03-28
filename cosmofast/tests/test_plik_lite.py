import numpy as np
import jax.numpy as jnp
import cosmofast as cf
import os


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
ft_tt = np.load(os.path.join(CURRENT_PATH, 'test_plik_lite_tt.npz'))
ft_ttteee = np.load(os.path.join(CURRENT_PATH, 'test_plik_lite_ttteee.npz'))


def test_plik_tt():
    cl = cf.planck_18.plik_lite_get_binned(ft_tt['test_param'][:-1], 'TT')
    ap = jnp.asarray(ft_tt['test_param'][-1])
    f = cf.planck_18.plik_lite(cl, ap, 'TT')
    assert np.isclose(float(f), float(ft_tt['test_value']), atol=1e-3)


def test_plik_ttteee():
    cl = cf.planck_18.plik_lite_get_binned(ft_ttteee['test_param'][:-1], 'TTTEEE')
    ap = jnp.asarray(ft_ttteee['test_param'][-1])
    f = cf.planck_18.plik_lite(cl, ap, 'TTTEEE')
    assert np.isclose(float(f), float(ft_ttteee['test_value']), atol=1e-3)
