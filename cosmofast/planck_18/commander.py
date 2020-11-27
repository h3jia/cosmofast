import numpy as np
from bayesfast import Module
from ._commander import _commander_f, _commander_j, _commander_fj
import os

__all__ = ['Commander']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo = np.load(os.path.join(CURRENT_PATH, 'data/commander.npz'))
cl2x = foo['cl2x']
mu = foo['mu']
cov = foo['cov']
cov_inv = np.linalg.inv(cov)
offset = foo['offset']


class Commander(Module):
    """Planck 2018 Commander low-l TT likelihood."""
    def __init__(self, tt_name='TT', m_name='TT-Commander', ap_name='a_planck',
                 logp_name='logp-Commander', delete_vars=[], label=None):
        super().__init__(delete_vars=delete_vars, concat_input=False,
                         concat_output=False, label=label)
        self.tt_name = tt_name
        self.m_name = m_name
        self.ap_name = ap_name
        self.logp_name = logp_name

    def _fun_jac_init(self, fun, jac, fun_and_jac):
        pass

    def _input_output_init(self, input_vars, output_vars):
        pass

    @property
    def tt_name(self):
        return self._tt_name

    @tt_name.setter
    def tt_name(self, ttn):
        if isinstance(ttn, str):
            self._tt_name = ttn
        else:
            raise ValueError('invalid value for tt_name.')

    @property
    def m_name(self):
        return self._m_name

    @m_name.setter
    def m_name(self, mn):
        if isinstance(mn, str):
            self._m_name = mn
        else:
            raise ValueError('invalid value for m_name.')

    @property
    def ap_name(self):
        return self._ap_name

    @ap_name.setter
    def ap_name(self, apn):
        if isinstance(apn, str):
            self._ap_name = apn
        else:
            raise ValueError('invalid value for ap_name.')

    @property
    def logp_name(self):
        return self._logp_name

    @logp_name.setter
    def logp_name(self, ln):
        if isinstance(ln, str):
            self._logp_name = ln
        else:
            raise ValueError('invalid value for logp_name.')

    @property
    def input_vars(self):
        return [self.m_name, self.ap_name]

    @property
    def output_vars(self):
        return [self.logp_name]

    def pre_cl(self, tmp_dict):
        raw_cl = tmp_dict[self.tt_name]
        try:
            assert raw_cl.ndim == 1
            assert raw_cl.size >= 30
        except Exception:
            raise ValueError('invalid shapr for raw_cl.')
        return raw_cl[2:30]

    def _fun(self, cl, ap):
        out_f = np.empty(1)
        _commander_f(cl, ap, out_f, cl2x, mu, cov_inv)
        out_f -= offset
        return out_f

    def _jac(self, cl, ap):
        out_j = np.empty((1, 29))
        _commander_j(cl, ap, out_j, cl2x, mu, cov_inv)
        return out_j

    def _fun_and_jac(self, cl, ap):
        out_f = np.empty(1)
        out_j = np.empty((1, 29))
        _commander_fj(cl, ap, out_f, out_j, cl2x, mu, cov_inv)
        out_f -= offset
        return out_f, out_j
