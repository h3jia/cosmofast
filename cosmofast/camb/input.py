import numpy as np
import bayesfast as bf
from bayesfast.utils.collections import PropertyList
from collections import OrderedDict
import camb

__all__ = ['CInputBase', 'CInput']

# TODO: add consistency module to allow different parameterization
# TODO: implement convenience utilities for initial points and priors


_default_input = OrderedDict(
    H0=67.0,
    ombh2=0.022,
    omch2=0.12,
    omk=0.0,
    mnu=0.06,
    nnu=3.046,
    YHe=None,
    meffsterile=0.0,
    tau=None,
    Alens=1.0,
    w=-1.0,
    wa=0.0,
    As=2e-9,
    ns=0.96,
    nrun=0.0,
    nrunrun=0.0,
    r=0.0,
)


_default_init = OrderedDict(
    H0=[67.0, 20.0, 100.0],
    ombh2=[0.022, 0.005, 0.1],
    omch2=[0.12, 0.001, 0.99],
    omk=[0.0, -0.3, 0.3],
    mnu=[0.06, 0.0, 5.0],
    nnu=[3.046, 0.05, 10.0], # [3.1, 3.046, 10.0] with meffsterile
    YHe=[0.25, 0.1, 0.5],
    meffsterile=[0.0, 0.1, 3.0],
    tau=[0.06, 0.01, 0.8],
    Alens=[1.0, 0.0, 10.0],
    w=[-1.0, -3.0, 1.0],
    wa=[0.0, -3.0, 2.0],
    As=[2e-9, 5e-10, 5e-9],
    ns=[0.96, 0.8, 1.2],
    nrun=[0.0, -1.0, 1.0],
    nrunrun=[0.0, -1.0, 1.0],
    r=[0.01, 0.0, 3.0],
)


_supported_keys = list(_default_input.keys())


_set_cosmology_keys = ['H0', 'ombh2', 'omch2', 'omk', 'mnu', 'nnu', 'YHe',
                       'meffsterile', 'tau', 'Alens']


_set_dark_energy_keys = ['w', 'wa']


_set_init_power_keys = ['As', 'ns', 'nrun', 'nrunrun', 'r']


_get_subdict = lambda dic, ks: OrderedDict([(k, dic[k]) for k in ks])


class CInputBase:
    '''Base class for CAMB input model.'''
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('abstract method.')

    def get(self, x):
        raise NotImplementedError('abstract method.')

    __call__ = get


class CInput(CInputBase):
    '''Default CAMB input model.'''
    def __init__(self, input_vars, fixed_vars={}, post=None):
        self.input_vars = input_vars
        self.fixed_vars = fixed_vars
        self.post = post

    @property
    def input_vars(self):
        return self._input_vars

    @input_vars.setter
    def input_vars(self, iv):
        if isinstance(iv, str):
            iv = [iv]
        self._input_vars = PropertyList(iv, self._input_check)

    @staticmethod
    def _input_check(input_vars):
        input_vars = list(input_vars)
        for iv in input_vars:
            if not (iv in _supported_keys):
                raise ValueError('unsupported input variable: {}.'.format(iv))
        return input_vars

    @property
    def fixed_vars(self):
        return self._fixed_vars

    @fixed_vars.setter
    def fixed_vars(self, fv):
        try:
            self._fixed_vars = dict(fv)
        except Exception:
            raise ValueError('invalid value for fixed_vars.')

    @property
    def post(self):
        return self._post

    @post.setter
    def post(self, p):
        self._post = p if callable(p) else (lambda params, x: params)

    def get(self, x):
        try:
            x = np.atleast_1d(x)
            assert x.shape == (len(self.input_vars),)
        except Exception:
            raise ValueError('invalid input x.')
        input_dict = _default_input.copy()
        input_dict.update(self.fixed_vars)
        for i, iv in enumerate(self.input_vars):
            input_dict[iv] = x[i]

        params = camb.CAMBparams()
        params.set_cosmology(**_get_subdict(input_dict, _set_cosmology_keys))
        _dark_energy_model = 'fluid' if input_dict['wa'] == 0. else 'ppf'
        params.set_dark_energy(dark_energy_model=_dark_energy_model,
                             **_get_subdict(input_dict, _set_dark_energy_keys))
        params.InitPower.set_params(**_get_subdict(input_dict,
                                                 _set_init_power_keys))
        return self.post(params, x)

    __call__ = get
