import numpy as np
import cosmofast as cf
import numdifftools as nd
import os

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo_ee = np.load(os.path.join(CURRENT_PATH, '../planck_18/data/simall_ee.npz'))
foo_bb = np.load(os.path.join(CURRENT_PATH, '../planck_18/data/simall_bb.npz'))
test_ee = np.load(os.path.join(CURRENT_PATH, 'test_simall_ee.npz'))
test_bb = np.load(os.path.join(CURRENT_PATH, 'test_simall_bb.npz'))

# test simall_ee

def bf_f_ee(x):
    test_f = np.empty(1)
    cf.planck_18._simall._simall_f(
        x[:-1], x[-1], test_f, foo_ee['c'], foo_ee['step'], foo_ee['n_step'])
    return test_f

def bf_j_ee(x):
    test_j = np.empty((1, 29))
    cf.planck_18._simall._simall_j(
        x[:-1], x[-1], test_j, foo_ee['c'], foo_ee['step'], foo_ee['n_step'])
    return test_j

def bf_fj_ee(x):
    test_fjf = np.empty(1)
    test_fjj = np.empty((1, 29))
    cf.planck_18._simall._simall_fj(
        x[:-1], x[-1], test_fjf, test_fjj, foo_ee['c'], foo_ee['step'],
        foo_ee['n_step'])
    return test_fjf, test_fjj

f_ee = bf_f_ee(test_ee['test_param'][2:])
j_ee = bf_j_ee(test_ee['test_param'][2:])
f_fj_ee, j_fj_ee = bf_fj_ee(test_ee['test_param'][2:])

def test_fj_ee():
    assert np.isclose(f_ee, f_fj_ee).all()
    assert np.isclose(j_ee, j_fj_ee).all()

def test_f_ee():
    assert np.isclose(f_ee, test_ee['test_value'], rtol=0, atol=5e-2).all()

def test_j_ee():
    assert np.isclose(
        j_ee[0], nd.Gradient(bf_f_ee, step=1e-8)(test_ee['test_param'][2:]),
        rtol=1e-2, atol=0).all()

# test simall_bb

def bf_f_bb(x):
    test_f = np.empty(1)
    cf.planck_18._simall._simall_f(
        x[:-1], x[-1], test_f, foo_bb['c'], foo_bb['step'], foo_bb['n_step'])
    return test_f

def bf_j_bb(x):
    test_j = np.empty((1, 29))
    cf.planck_18._simall._simall_j(
        x[:-1], x[-1], test_j, foo_bb['c'], foo_bb['step'], foo_bb['n_step'])
    return test_j

def bf_fj_bb(x):
    test_fjf = np.empty(1)
    test_fjj = np.empty((1, 29))
    cf.planck_18._simall._simall_fj(
        x[:-1], x[-1], test_fjf, test_fjj, foo_bb['c'], foo_bb['step'],
        foo_bb['n_step'])
    return test_fjf, test_fjj

f_bb = bf_f_bb(test_bb['test_param'][2:])
j_bb = bf_j_bb(test_bb['test_param'][2:])
f_fj_bb, j_fj_bb = bf_fj_bb(test_bb['test_param'][2:])

def test_fj_bb():
    assert np.isclose(f_bb, f_fj_bb).all()
    assert np.isclose(j_bb, j_fj_bb).all()

def test_f_bb():
    assert np.isclose(f_bb, test_bb['test_value'], rtol=0, atol=5e-2).all()

def test_j_bb():
    assert np.isclose(
        j_bb[0], nd.Gradient(bf_f_bb, step=1e-8)(test_bb['test_param'][2:]),
        rtol=1e-2, atol=0).all()
