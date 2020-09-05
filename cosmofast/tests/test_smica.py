import numpy as np
import cosmofast as cf
import numdifftools as nd
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo = np.load(os.path.join(CURRENT_PATH, '../planck_18/data/smica.npz'))
test = np.load(os.path.join(CURRENT_PATH, 'test_smica.npz'))
foo_cmb_marged = np.load(
    os.path.join(CURRENT_PATH, '../planck_18/data/smica_cmb_marged.npz'))
test_cmb_marged = np.load(
    os.path.join(CURRENT_PATH, '../tests/test_smica_cmb_marged.npz'))

# test smica

lens_b = np.empty(foo['n_bin'])
cmb_b = np.empty(foo['n_bin'])
cf.planck_18._smica._get_binned_cls(
    test['test_param'][:-1], lens_b, cmb_b, foo['F'], foo['n_cmb'],
    foo['l_min'], foo['l_max'], foo['n_bin'])
xx = np.concatenate((lens_b, cmb_b, test['test_param'][-1:]))

def bf_f(x):
    test_f = np.empty(1)
    cf.planck_18._smica._smica_f(
        x[:foo['n_bin']], x[foo['n_bin']:(2 * foo['n_bin'])], x[-1], test_f,
        foo['mu'], foo['siginv'], foo['n_cmb'], foo['n_bin'])
    return test_f

def bf_j(x):
    test_j = np.empty((1, (2 * foo['n_bin'] + 1)))
    cf.planck_18._smica._smica_j(
        x[:foo['n_bin']], x[foo['n_bin']:(2 * foo['n_bin'])], x[-1], test_j,
        foo['mu'], foo['siginv'], foo['n_cmb'], foo['n_bin'])
    return test_j

def bf_fj(x):
    test_fjf = np.empty(1)
    test_fjj = np.empty((1, (2 * foo['n_bin'] + 1)))
    cf.planck_18._smica._smica_fj(
        x[:foo['n_bin']], x[foo['n_bin']:(2 * foo['n_bin'])], x[-1], test_fjf,
        test_fjj, foo['mu'], foo['siginv'], foo['n_cmb'], foo['n_bin'])
    return test_fjf, test_fjj

f = bf_f(xx)
j = bf_j(xx)
f_fj, j_fj = bf_fj(xx)

def test_fj():
    assert np.isclose(f, f_fj).all()
    assert np.isclose(j, j_fj).all()

def test_f():
    assert np.isclose(f, test['test_value'], rtol=0, atol=1e-5).all()

def test_j():
    assert np.isclose(
        j[0], nd.Gradient(bf_f, step=1e-8)(xx), rtol=1e-5, atol=0).all()

# test smica cmb marged

lens_b_cmb_marged = np.empty(foo_cmb_marged['n_bin'])
cmb_b_cmb_marged = np.empty(1)
cf.planck_18._smica._get_binned_cls(
    test_cmb_marged['test_param'][:-1], lens_b_cmb_marged, cmb_b_cmb_marged,
    foo_cmb_marged['F'], foo_cmb_marged['n_cmb'], foo_cmb_marged['l_min'],
    foo_cmb_marged['l_max'], foo_cmb_marged['n_bin'])
xx_cmb_marged = lens_b_cmb_marged

def bf_f_cmb_marged(x):
    test_f = np.empty(1)
    cf.planck_18._smica._smica_f(
        x, cmb_b_cmb_marged, 1., test_f, foo_cmb_marged['mu'],
        foo_cmb_marged['siginv'], foo_cmb_marged['n_cmb'],
        foo_cmb_marged['n_bin'])
    return test_f

def bf_j_cmb_marged(x):
    test_j = np.empty((1, foo_cmb_marged['n_bin']))
    cf.planck_18._smica._smica_j(
        x, cmb_b_cmb_marged, 1., test_j, foo_cmb_marged['mu'],
        foo_cmb_marged['siginv'], foo_cmb_marged['n_cmb'],
        foo_cmb_marged['n_bin'])
    return test_j

def bf_fj_cmb_marged(x):
    test_fjf = np.empty(1)
    test_fjj = np.empty((1, foo_cmb_marged['n_bin']))
    cf.planck_18._smica._smica_fj(
        x, cmb_b_cmb_marged, 1., test_fjf, test_fjj, foo_cmb_marged['mu'],
        foo_cmb_marged['siginv'], foo_cmb_marged['n_cmb'],
        foo_cmb_marged['n_bin'])
    return test_fjf, test_fjj

f_cmb_marged = bf_f_cmb_marged(xx_cmb_marged)
j_cmb_marged = bf_j_cmb_marged(xx_cmb_marged)
f_fj_cmb_marged, j_fj_cmb_marged = bf_fj_cmb_marged(xx_cmb_marged)

def test_fj_cmb_marged():
    assert np.isclose(f_cmb_marged, f_fj_cmb_marged).all()
    assert np.isclose(j_cmb_marged, j_fj_cmb_marged).all()

def test_f_cmb_marged():
    assert np.isclose(
        f_cmb_marged, test_cmb_marged['test_value'], rtol=0, atol=1e-5).all()

def test_j_cmb_marged():
    assert np.isclose(
        j_cmb_marged[0], nd.Gradient(bf_f_cmb_marged, step=1e-8)(xx_cmb_marged),
        rtol=1e-5, atol=0).all()
