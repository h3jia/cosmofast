import numpy as np
from bayesfast import ModuleBase
from ._simall import _simall_f, _simall_j, _simall_fj
import os

__all__ = ['Simall']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo_ee = dict(np.load(os.path.join(CURRENT_PATH, 'data/simall_ee.npz')))
foo_bb = dict(np.load(os.path.join(CURRENT_PATH, 'data/simall_bb.npz')))


class Simall(ModuleBase):
    """Planck 2018 Simall low-l EE and BB likelihoods."""
    def __init__(self, kind='EE', ee_name='EE', bb_name='BB', m_name=None,
                 ap_name='a_planck', logp_name=None, delete_vars=[],
                 label=None):
        super().__init__(delete_vars=delete_vars, input_shapes=None,
                         output_shapes=None, label=label)
        self.kind = kind
        self.ee_name = ee_name
        self.bb_name = bb_name
        self.m_name = m_name
        self.ap_name = ap_name
        self.logp_name = logp_name

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, k):
        if k == 'EE' or k == 'BB':
            self._kind = k
        elif k == 'EEBB':
            raise NotImplementedError('for EEBB, please just use one EE plus '
                                      'one BB for now.')
        else:
            raise ValueError('invalid value for kind.')

    @property
    def ee_name(self):
        return self._ee_name

    @ee_name.setter
    def ee_name(self, een):
        if isinstance(een, str):
            self._ee_name = een
        else:
            raise ValueError('invalid value for ee_name.')

    @property
    def bb_name(self):
        return self._bb_name

    @bb_name.setter
    def bb_name(self, bbn):
        if isinstance(bbn, str):
            self._bb_name = bbn
        else:
            raise ValueError('invalid value for bb_name.')

    @property
    def m_name(self):
        if self._m_name is None:
            return self.kind.lower() + '_simall'
        else:
            return self._m_name

    @m_name.setter
    def m_name(self, mn):
        if mn is None or isinstance(mn, str):
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
        if self._logp_name is None:
            return 'logp_simall_' + self.kind.lower()
        else:
            return self._logp_name

    @logp_name.setter
    def logp_name(self, ln):
        if ln is None or isinstance(ln, str):
            self._logp_name = ln
        else:
            raise ValueError('invalid value for logp_name.')

    @property
    def input_vars(self):
        return [self.m_name, self.ap_name]

    @property
    def output_vars(self):
        return [self.logp_name]

    @property
    def camb_output_vars(self):
        return [self.m_name]

    def camb_get_output(self, tmp_dict):
        if self.kind == 'EE':
            raw_cl = tmp_dict[self.ee_name]
        elif self.kind == 'BB':
            raw_cl = tmp_dict[self.bb_name]
        else:
            raise RuntimeError('unexpected value for self.kind.')
        try:
            assert raw_cl.ndim == 1
            assert raw_cl.size >= 30
        except Exception:
            raise ValueError('invalid shape for raw_cl.')
        return raw_cl[2:30]

    @property
    def camb_output_shapes(self):
        return [28]

    def _fun(self, m, ap):
        out_f = np.empty(1)
        if self.kind == 'EE':
            _simall_f(m, ap, out_f, foo_ee['c'], foo_ee['step'],
                      foo_ee['n_step'])
        elif self.kind == 'BB':
            _simall_f(m, ap, out_f, foo_bb['c'], foo_bb['step'],
                      foo_bb['n_step'])
        else:
            raise RuntimeError('unexpected value for self.kind.')
        return out_f

    def _jac(self, m, ap):
        out_j = np.empty((1, 29))
        if self.kind == 'EE':
            _simall_j(m, ap, out_j, foo_ee['c'], foo_ee['step'],
                      foo_ee['n_step'])
        elif self.kind == 'BB':
            _simall_j(m, ap, out_j, foo_bb['c'], foo_bb['step'],
                      foo_bb['n_step'])
        else:
            raise RuntimeError('unexpected value for self.kind.')
        return out_j

    def _fun_and_jac(self, m, ap):
        out_f = np.empty(1)
        out_j = np.empty((1, 29))
        if self.kind == 'EE':
            _simall_fj(m, ap, out_f, out_j, foo_ee['c'], foo_ee['step'],
                       foo_ee['n_step'])
        elif self.kind == 'BB':
            _simall_fj(m, ap, out_f, out_j, foo_bb['c'], foo_bb['step'],
                      foo_bb['n_step'])
        else:
            raise RuntimeError('unexpected value for self.kind.')
        return out_f, out_j
