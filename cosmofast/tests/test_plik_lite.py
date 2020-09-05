import numpy as np
import cosmofast as cf
import numdifftools as nd
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo_tt = np.load(
    os.path.join(CURRENT_PATH,'../planck_18/data/plik_lite_tt.npz'))
test_tt = np.load(os.path.join(CURRENT_PATH, 'test_plik_lite_tt.npz'))
foo_ttteee = np.load(
    os.path.join(CURRENT_PATH, '../planck_18/data/plik_lite_ttteee.npz'))
test_ttteee = np.load(os.path.join(CURRENT_PATH, 'test_plik_lite_ttteee.npz'))

# test plik_lite tt

cls_b_tt = np.empty(foo_tt['n_bin'])
cf.planck_18._plik_lite._get_binned_cls(
    test_tt['test_param'][30:2509], cls_b_tt, foo_tt['weight_t'],
    foo_tt['low_t'], foo_tt['width_t'], foo_tt['n_bin_t'])
cls_b_tt = cls_b_tt @ foo_tt['cov_inv_low']
x_tt = np.concatenate((cls_b_tt, [test_tt['test_param'][-1]]))

def bf_f_tt(x):
    test_f = np.empty(1)
    cf.planck_18._plik_lite._plik_lite_f(
        x[:-1], x[-1], test_f, foo_tt['mu'], foo_tt['n_bin'])
    return test_f

def bf_j_tt(x):
    test_j = np.empty((1, foo_tt['n_bin'] + 1))
    cf.planck_18._plik_lite._plik_lite_j(
        x[:-1], x[-1], test_j, foo_tt['mu'], foo_tt['n_bin'])
    return test_j

def bf_fj_tt(x):
    test_fjf = np.empty(1)
    test_fjj = np.empty((1, foo_tt['n_bin'] + 1))
    cf.planck_18._plik_lite._plik_lite_fj(
        x[:-1], x[-1], test_fjf, test_fjj, foo_tt['mu'], foo_tt['n_bin'])
    return test_fjf, test_fjj

f_tt = bf_f_tt(x_tt)
j_tt = bf_j_tt(x_tt)
f_fj_tt, j_fj_tt = bf_fj_tt(x_tt)

def test_fj_tt():
    assert np.isclose(f_tt, f_fj_tt).all()
    assert np.isclose(j_tt, j_fj_tt).all()

def test_f_tt():
    assert np.isclose(f_tt, test_tt['test_value'], rtol=0, atol=1e-5).all()

def test_j_tt():
    assert np.isclose(
        j_tt[0], nd.Gradient(bf_f_tt, step=1e-8)(x_tt), rtol=1e-3, atol=0).all()

# test plik_lite ttteee

cls_b_ttteee = np.empty(foo_ttteee['n_bin'])
cf.planck_18._plik_lite._get_binned_cls(
    test_ttteee['test_param'][30:2509],
    cls_b_ttteee[foo_ttteee['i_bin'][0]:foo_ttteee['i_bin'][1]],
    foo_ttteee['weight_t'], foo_ttteee['low_t'], foo_ttteee['width_t'],
    foo_ttteee['n_bin_t'])
cf.planck_18._plik_lite._get_binned_cls(
    test_ttteee['test_param'][(2509+2509+30):(2509+2509+2509)],
    cls_b_ttteee[foo_ttteee['i_bin'][1]:foo_ttteee['i_bin'][2]],
    foo_ttteee['weight_e'], foo_ttteee['low_e'], foo_ttteee['width_e'],
    foo_ttteee['n_bin_e'])
cf.planck_18._plik_lite._get_binned_cls(
    test_ttteee['test_param'][(2509+30):(2509+2509)],
    cls_b_ttteee[foo_ttteee['i_bin'][2]:foo_ttteee['i_bin'][3]],
    foo_ttteee['weight_e'], foo_ttteee['low_e'], foo_ttteee['width_e'],
    foo_ttteee['n_bin_e'])
cls_b_ttteee = cls_b_ttteee @ foo_ttteee['cov_inv_low']
x_ttteee = np.concatenate((cls_b_ttteee, [test_ttteee['test_param'][-1]]))

def bf_f_ttteee(x):
    test_f = np.empty(1)
    cf.planck_18._plik_lite._plik_lite_f(
        x[:-1], x[-1], test_f, foo_ttteee['mu'], foo_ttteee['n_bin'])
    return test_f

def bf_j_ttteee(x):
    test_j = np.empty((1, foo_ttteee['n_bin'] + 1))
    cf.planck_18._plik_lite._plik_lite_j(
        x[:-1], x[-1], test_j, foo_ttteee['mu'], foo_ttteee['n_bin'])
    return test_j

def bf_fj_ttteee(x):
    test_fjf = np.empty(1)
    test_fjj = np.empty((1, foo_ttteee['n_bin'] + 1))
    cf.planck_18._plik_lite._plik_lite_fj(
        x[:-1], x[-1], test_fjf, test_fjj, foo_ttteee['mu'],
        foo_ttteee['n_bin'])
    return test_fjf, test_fjj

f_ttteee = bf_f_ttteee(x_ttteee)
j_ttteee = bf_j_ttteee(x_ttteee)
f_fj_ttteee, j_fj_ttteee = bf_fj_ttteee(x_ttteee)

def test_fj_ttteee():
    assert np.isclose(f_ttteee, f_fj_ttteee).all()
    assert np.isclose(j_ttteee, j_fj_ttteee).all()

def test_f_ttteee():
    assert np.isclose(
        f_ttteee, test_ttteee['test_value'], rtol=0, atol=1e-5).all()

def test_j_ttteee():
    assert np.isclose(
        j_ttteee[0], nd.Gradient(bf_f_ttteee, step=1e-8)(x_ttteee), rtol=1e-2,
        atol=0).all()
