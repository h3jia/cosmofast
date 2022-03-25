from .input import CInputBase
from .calc import CCalcBase
from bayesfast import ModuleBase
from bayesfast.utils.collections import PropertyList
from bayesfast.utils.sobol import multivariate_normal
from collections import OrderedDict
from itertools import chain
import numpy as np
import camb
import warnings

__all__ = ['CAMB']


_default_init = OrderedDict(
    H0=[67.0, 20.0, 100.0],
    ombh2=[0.022, 0.005, 0.1],
    omch2=[0.12, 0.001, 0.99],
    omk=[0.0, -0.3, 0.3],
    mnu=[0.06, 0.0, 5.0],
    nnu=[3.046, 0.05, 10.0],
    nnusterile=[3.1, 3.046, 10.0], # when meffsterile is used
    YHe=[0.25, 0.1, 0.5],
    meffsterile=[0.1, 0.0, 3.0],
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


def _in_bound(x, bound):
    xt = np.atleast_2d(x).T
    return np.product([np.where(xi>bound[i,0], True, False) * 
                       np.where(xi<bound[i,1], True, False) for i, xi in 
                       enumerate(xt)], axis=0).astype(bool)


class CAMB(ModuleBase):
    '''CAMB cosmology model.'''
    def __init__(self, c_input, c_calc, output_vars=None, get_output=None,
                 delete_vars=[], label=None):
        super().__init__(output_vars=output_vars, delete_vars=delete_vars,
                         input_shapes=-1, output_shapes=None, label=label)
        self.c_input = c_input
        self.c_calc = c_calc
        self.get_output = get_output

    @property
    def c_input(self):
        return self._c_input

    @c_input.setter
    def c_input(self, ci):
        if isinstance(ci, CInputBase):
            self._c_input = ci
        else:
            self._c_input = None
            raise ValueError('c_input should be a subclass of CInputBase.')

    @property
    def c_calc(self):
        return self._c_calc

    @c_calc.setter
    def c_calc(self, cc):
        if isinstance(cc, CCalcBase):
            cc = [cc]
        self._c_calc = PropertyList(cc, self._calc_check)

    @staticmethod
    def _calc_check(c_calc):
        if not all(isinstance(cc, CCalcBase) for cc in c_calc):
            raise ValueError('some element of c_calc is not a subclass of '
                             'CCalcBase.')
        return c_calc

    @property
    def input_vars(self):
        return self.c_input.input_vars

    @property
    def output_vars(self):
        if self._output_vars is None:
            try:
                ov = [v for c in self.c_calc for v in c.output_vars]
                assert all(isinstance(v, str) for v in ov)
                ov = sorted(set(ov), key=ov.index) # remove redundant elements
                return ov
            except Exception:
                raise RuntimeError('failed to get output_vars from c_calc.')
        else:
            return self._output_vars

    @output_vars.setter
    def output_vars(self, ov):
        if ov is None:
            self._output_vars = None
        else:
            self._output_vars = PropertyList(
                ov, lambda x: ModuleBase._var_check(x, 'output', 'raise', 1,
                np.inf))

    @staticmethod
    def _dict_output(tmp_dict):
        return list(tmp_dict.values())

    @property
    def get_output(self):
        if self._get_output is None:
            return self._dict_output
        else:
            return self._get_output

    @get_output.setter
    def get_output(self, go):
        if go is None or callable(go):
            self._get_output = go
        else:
            raise ValueError('invalid value for get_output.')

    def set_output(self, modules):
        try:
            if hasattr(modules, 'camb_output_vars'):
                modules = (modules,)
            ov = list(chain(*[list(m.camb_output_vars) for m in modules]))
            assert all([callable(m.camb_get_output) for m in modules])
            go = self.multi_output([m.camb_get_output for m in modules])
            os = list(chain(*[list(m.camb_output_shapes) for m in modules]))
            self.output_vars = ov
            self.get_output = go
            self.output_shapes = os
        except Exception:
            raise ValueError('failed to set output for the modules you give.')

    @staticmethod
    def _check_output(output):
        if isinstance(output, np.ndarray):
            return [output]
        elif isinstance(output, (list, tuple)):
            return output
        else:
            raise RuntimeError('invalid value in _check_output.')

    @classmethod
    def multi_output(cls, fun_list):
        return lambda x: list(
            chain(*[cls._check_output(fun(x)) for fun in fun_list]))

    def _fun(self, x):
        params = self.c_input(x)
        data = camb.CAMBdata()

        min_calc = max(calc.min_calc for calc in self.c_calc)
        if min_calc == 0:
            data.set_params(params)
        elif min_calc == 1:
            data.calc_background_no_thermo(params)
        elif min_calc == 2:
            data.calc_background(params)
        elif min_calc == 3:
            data.calc_power_spectra(params)
        else:
            raise RuntimeError('unexpected value for min_calc.')

        tmp_dict = OrderedDict()
        for calc in self.c_calc:
            calc(data, tmp_dict)
        return self.get_output(tmp_dict)

    def default_input_scales(self, warn_not_found=True, a_planck=True):
        input_scales = []
        for i in self.input_vars:
            if i == 'nnu' and 'meffsterile' in self.input_vars:
                input_scales.append(_default_init['nnusterile'][1:3])
            elif i in _default_init:
                input_scales.append(_default_init[i][1:3])
            else:
                if warn_not_found:
                    warnings.warn('does not find default input info for '
                                  '{}.'.format(i), RuntimeWarning)
                input_scales.append([0, 1])
        if a_planck:
            input_scales.append([1, 1 + 0.0025])
        return np.array(input_scales)

    def default_hard_bounds(self, warn_not_found=True, a_planck=True):
        hard_bounds = []
        for i in self.input_vars:
            if i in _default_init:
                hard_bounds.append(True)
            else:
                if warn_not_found:
                    warnings.warn('does not find default input info for '
                                  '{}.'.format(i), RuntimeWarning)
                hard_bounds.append(False)
        if a_planck:
            hard_bounds.append(False)
        return np.array(hard_bounds)

    def default_x_0(self, n=100, warn_not_found=True, a_planck=True,
                    scale_factor=1000.):
        try:
            n = int(n)
            assert n > 0
        except Exception:
            raise ValueError('invalid value for n.')
        try:
            scale_factor = float(scale_factor)
            assert scale_factor > 0
        except Exception:
            raise ValueError('invalid value for scale_factor.')

        dis = self.default_input_scales(False, False)
        dhb = self.default_hard_bounds(False, False)
        sig = (dis[:, 1] - dis[:, 0]) / scale_factor
        x_center = np.empty(0)

        for i in self.input_vars:
            if i == 'nnu' and 'meffsterile' in self.input_vars:
                x_center = np.append(x_center, _default_init['nnusterile'][0])
            elif i in _default_init:
                x_center = np.append(x_center, _default_init[i][0])
            else:
                if warn_not_found:
                    warnings.warn('does not find default input info for '
                                  '{}.'.format(i), RuntimeWarning)
                x_center = np.append(x_center, 0.5)

        x_0_all = np.empty((0, dhb.shape[0]))
        n_mvn = int(1.5 * n)
        skip_mvn = 1
        while True:
            x_0 = multivariate_normal(x_center, np.diag(sig**2), n_mvn,
                                      skip_mvn)
            skip_mvn = skip_mvn + n_mvn
            x_0 = x_0[_in_bound(x_0, dis)]
            if x_0.shape[0] == 0:
                raise RuntimeError('no x_0 in this batch is inside the '
                                   'boundary. Plese check your settings.')
            x_0_all = np.concatenate((x_0_all, x_0), axis=0)
            if x_0_all.shape[0] >= n:
                break

        x_0_all = x_0_all[:n]
        if a_planck:
            x_0_all = np.hstack((x_0_all, np.ones((n, 1))))
        return x_0_all
