from .input import CInputBase
from .calc import CCalcBase
from bayesfast import ModuleBase
from bayesfast.utils.collections import PropertyList
from collections import OrderedDict
from itertools import chain
import numpy as np
import camb

__all__ = ['CAMB']


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

    @staticmethod
    def multi_output(fun_list):
        return lambda x: list(chain([list(fun(x)) for fun in fun_list]))

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
