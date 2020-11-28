import numpy as np
from bayesfast import Module
from ._smica import _get_binned_cls, _smica_f, _smica_j, _smica_fj
import os

__all__ = ['Smica']


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
foo = dict(np.load(os.path.join(CURRENT_PATH, 'data/smica.npz')))
foo_cmb_marged = dict(
    np.load(os.path.join(CURRENT_PATH, 'data/smica_cmb_marged.npz')))


class Smica(Module):
    """Planck 2018 Smica lensing likelihoods."""
    def __init__(self, cmb_marged=False, pp_name='PP', tt_name='TT',
                 te_name='TE', ee_name='EE', m_name=None, ap_name='a_planck',
                 logp_name=None, delete_vars=[], label=None):
        super().__init__(delete_vars=delete_vars, concat_input=False,
                         concat_output=False, label=label)
        self.cmb_marged = cmb_marged
        self.pp_name = pp_name
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
    def cmb_marged(self):
        return self._cmb_marged

    @cmb_marged.setter
    def cmb_marged(self, cm):
        self._cmb_marged = bool(cm)

    @property
    def pp_name(self):
        return self._pp_name

    @pp_name.setter
    def pp_name(self, ppn):
        if isinstance(ppn, str):
            self._pp_name = ppn
        else:
            raise ValueError('invalid value for pp_name.')

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
            return 'm-Smica-CMB-Marged' if self.cmb_marged else 'm-Smica'
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
            return 'logp-Smica-CMB-Marged' if self.cmb_marged else 'logp-Smica'
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
        if self.cmb_marged:
            return [self.m_name]
        else:
            return [self.m_name, self.ap_name]

    @property
    def output_vars(self):
        return [self.logp_name]

    def pre_cl(self, tmp_dict):
        raw_pp = tmp_dict[self.pp_name]
        try:
            assert raw_pp.ndim == 1
            assert raw_pp.size >= 2501
        except Exception:
            raise ValueError('invalid shape for raw_pp.')

        if self.cmb_marged:
            lens = np.empty(foo_cmb_marged['n_bin'])
            cmb = np.empty(1)
            _get_binned_cls(raw_pp[:2501], lens, cmb, foo_cmb_marged['F'],
                            foo_cmb_marged['n_cmb'], foo_cmb_marged['l_min'],
                            foo_cmb_marged['l_max'], foo_cmb_marged['n_bin'])
            return lens

        else:
            raw_tt = tmp_dict[self.tt_name]
            raw_te = tmp_dict[self.te_name]
            raw_ee = tmp_dict[self.ee_name]
            try:
                assert raw_tt.ndim == 1
                assert raw_tt.size >= 2501
            except Exception:
                raise ValueError('invalid shape for raw_tt.')
            try:
                assert raw_te.ndim == 1
                assert raw_te.size >= 2501
            except Exception:
                raise ValueError('invalid shape for raw_te.')
            try:
                assert raw_ee.ndim == 1
                assert raw_ee.size >= 2501
            except Exception:
                raise ValueError('invalid shape for raw_ee.')

            lens = np.empty(foo['n_bin'])
            cmb = np.empty(foo['n_bin'])
            raw_cl = np.concatenate((raw_pp[:2501], raw_tt[:2501],
                                     raw_ee[:2501], raw_te[:2501]))
            _get_binned_cls(raw_cl, lens, cmb, foo['F'], foo['n_cmb'],
                            foo['l_min'], foo['l_max'], foo['n_bin'])
            return np.concatenate((lens, cmb))

    def _fun(self, m, ap=1.):
        out_f = np.empty(1)
        if self.cmb_marged:
            _smica_f(m, np.empty(1), 1., out_f, foo_cmb_marged['mu'],
                     foo_cmb_marged['siginv'], foo_cmb_marged['n_cmb'],
                     foo_cmb_marged['n_bin'])
        else:
            _smica_f(m[:foo['n_bin']], m[foo['n_bin']:(2 * foo['n_bin'])], ap,
                     out_f, foo['mu'], foo['siginv'], foo['n_cmb'],
                     foo['n_bin'])
        return out_f

    def _jac(self, m, ap=1.):
        if self.cmb_marged:
            out_j = np.empty((1, foo_cmb_marged['n_bin']))
            _smica_j(m, np.empty(1), 1., out_j, foo_cmb_marged['mu'],
                     foo_cmb_marged['siginv'], foo_cmb_marged['n_cmb'],
                     foo_cmb_marged['n_bin'])
        else:
            out_j = np.empty((1, (2 * foo['n_bin'] + 1)))
            _smica_j(m[:foo['n_bin']], m[foo['n_bin']:(2 * foo['n_bin'])], ap,
                     out_j, foo['mu'], foo['siginv'], foo['n_cmb'],
                     foo['n_bin'])
        return out_j

    def _fun_and_jac(self, m, ap=1.):
        out_f = np.empty(1)
        if self.cmb_marged:
            out_j = np.empty((1, foo_cmb_marged['n_bin']))
            _smica_fj(m, np.empty(1), 1., out_f, out_j, foo_cmb_marged['mu'],
                      foo_cmb_marged['siginv'], foo_cmb_marged['n_cmb'],
                      foo_cmb_marged['n_bin'])
        else:
            out_j = np.empty((1, (2 * foo['n_bin'] + 1)))
            _smica_fj(m[:foo['n_bin']], m[foo['n_bin']:(2 * foo['n_bin'])], ap,
                      out_f, out_j, foo['mu'], foo['siginv'], foo['n_cmb'],
                      foo['n_bin'])
        return out_f, out_j
