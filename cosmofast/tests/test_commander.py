import numpy as np
import cosmofast as cf
import numdifftools as nd
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo = np.load(os.path.join(CURRENT_PATH, '../planck_18/data/commander.npz'))
test = np.load(os.path.join(CURRENT_PATH, 'test_commander.npz'))
cl2x = foo['cl2x']
mu = foo['mu']
cov = foo['cov']
cov_inv = np.linalg.inv(cov)
offset = foo['offset']

# test commander

def bf_f(x):
    test_f = np.empty(1)
    cf.planck_18._commander._commander_f(
        x[:-1], x[-1], test_f, cl2x, mu, cov_inv)
    return test_f - offset

def bf_j(x):
    test_j = np.empty((1, 29))
    cf.planck_18._commander._commander_j(
        x[:-1], x[-1], test_j, cl2x, mu, cov_inv)
    return test_j

def bf_fj(x):
    test_fjf = np.empty(1)
    test_fjj = np.empty((1, 29))
    cf.planck_18._commander._commander_fj(
        x[:-1], x[-1], test_fjf, test_fjj, cl2x, mu, cov_inv)
    return test_fjf - offset, test_fjj

f = bf_f(test['test_param'][2:])
j = bf_j(test['test_param'][2:])
f_fj, j_fj = bf_fj(test['test_param'][2:])

def test_fj():
    assert np.isclose(f, f_fj).all()
    assert np.isclose(j, j_fj).all()

def test_f():
    assert np.isclose(f, test['test_value'], rtol=0, atol=1e-5).all()

def test_j():
    assert np.isclose(
        j[0], nd.Gradient(bf_f, step=1e-8)(test['test_param'][2:]), rtol=1e-2,
        atol=0).all()
