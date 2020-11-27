import numpy as np
from bayesfast import Module
from ._plik_lite import _get_binned_cls, _plik_lite_f
from ._plik_lite import _plik_lite_j, _plik_lite_fj
import os

__all__ = ['PlikLite']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo_tt = dict(np.load(os.path.join(CURRENT_PATH,'data/plik_lite_tt.npz')))
foo_ttteee = dict(
    np.load(os.path.join(CURRENT_PATH, 'data/plik_lite_ttteee.npz')))


class PlikLite(Module):
    """Planck 2018 PlikLite high-l TT and TTTEEE likelihoods."""
    def __init__(self, kind='TT', tt_name='TT', te_name='TE', ee_name='EE',
                 m_name=None, ap_name='a_planck', logp_name=None,
                 delete_vars=[], label=None):
        super().__init__(delete_vars=delete_vars, concat_input=False,
                         concat_output=False, label=label)
        self.kind = kind
        self.tt_name = tt_name
        self.te_name = te_name
        self.ee_name = ee_name
        self.m_name = m_name
        self.ap_name = ap_name
        self.logp_name = logp_name

    def _fun_jac_init(self, fun, jac, fun_and_jac):
        pass

    def _input_output_init(self, input_vars, output_vars):
        pass

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, k):
        if k == 'TT' or k == 'TTTEEE':
            self._kind = k
        else:
            raise ValueError('invalid value for kind.')

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
    def te_name(self):
        return self._te_name

    @te_name.setter
    def te_name(self, ten):
        if isinstance(ten, str):
            self._te_name = ten
        else:
            raise ValueError('invalid value for te_name.')

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
    def m_name(self):
        if self._m_name is None:
            return self.kind + '-PlikLite'
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
            return 'logp-PlikLite-' + self.kind
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

    def pre_cl(self, tmp_dict):
        raw_tt = tmp_dict[self.tt_name]
        try:
            assert raw_tt.ndim == 1
            assert raw_tt.size >= 2509
        except Exception:
            raise ValueError('invalid shape for raw_tt.')

        if self.kind == 'TT':
            b_tt = np.empty(foo_tt['n_bin'])
            _get_binned_cls(
                raw_tt[30:2509], b_tt, foo_tt['weight_t'], foo_tt['low_t'],
                foo_tt['width_t'], foo_tt['n_bin_t'])
            b_tt = b_tt @ foo_tt['cov_inv_low']
            return b_tt

        elif self.kind == 'TTTEEE':
            raw_te = tmp_dict[self.te_name]
            raw_ee = tmp_dict[self.ee_name]
            try:
                assert raw_te.ndim == 1
                assert raw_te.size >= 2509
            except Exception:
                raise ValueError('invalid shape for raw_te.')
            try:
                assert raw_ee.ndim == 1
                assert raw_ee.size >= 2509
            except Exception:
                raise ValueError('invalid shape for raw_ee.')

            b_ttteee = np.empty(foo_ttteee['n_bin'])
            _get_binned_cls(
                raw_tt[30:2509],
                b_ttteee[foo_ttteee['i_bin'][0]:foo_ttteee['i_bin'][1]],
                foo_ttteee['weight_t'], foo_ttteee['low_t'],
                foo_ttteee['width_t'], foo_ttteee['n_bin_t'])
            _get_binned_cls(
                raw_te[30:2509],
                b_ttteee[foo_ttteee['i_bin'][1]:foo_ttteee['i_bin'][2]],
                foo_ttteee['weight_e'], foo_ttteee['low_e'],
                foo_ttteee['width_e'], foo_ttteee['n_bin_e'])
            _get_binned_cls(
                raw_ee[30:2509],
                b_ttteee[foo_ttteee['i_bin'][2]:foo_ttteee['i_bin'][3]],
                foo_ttteee['weight_e'], foo_ttteee['low_e'],
                foo_ttteee['width_e'], foo_ttteee['n_bin_e'])
            b_ttteee = b_ttteee @ foo_ttteee['cov_inv_low']
            return b_ttteee

    def _fun(self, m, ap):
        out_f = np.empty(1)
        if self.kind == 'TT':
            _plik_lite_f(m, ap, out_f, foo_tt['mu'], foo_tt['n_bin'])
        elif self.kind == 'TTTEEE':
            _plik_lite_f(m, ap, out_f, foo_ttteee['mu'], foo_ttteee['n_bin'])
        else:
            raise RuntimeError('unexpected value for self.kind.')
        return out_f

    def _jac(self, m, ap):
        if self.kind == 'TT':
            out_j = np.empty((1, foo_tt['n_bin'] + 1))
            _plik_lite_j(m, ap, out_j, foo_tt['mu'], foo_tt['n_bin'])
        elif self.kind == 'TTTEEE':
            out_j = np.empty((1, foo_ttteee['n_bin'] + 1))
            _plik_lite_j(m, ap, out_j, foo_ttteee['mu'], foo_ttteee['n_bin'])
        else:
            raise RuntimeError('unexpected value for self.kind.')
        return out_j

    def _fun_and_jac(self, m, ap):
        out_f = np.empty(1)
        if self.kind == 'TT':
            out_j = np.empty((1, foo_tt['n_bin'] + 1))
            _plik_lite_fj(m, ap, out_f, out_j, foo_tt['mu'], foo_tt['n_bin'])
        elif self.kind == 'TTTEEE':
            out_j = np.empty((1, foo_ttteee['n_bin'] + 1))
            _plik_lite_fj(m, ap, out_f, out_j, foo_ttteee['mu'],
                          foo_ttteee['n_bin'])
        else:
            raise RuntimeError('unexpected value for self.kind.')
        return out_f, out_j
