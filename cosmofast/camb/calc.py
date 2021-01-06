import numpy as np
from collections import OrderedDict

__all__ = ['CCalcBase', 'CCalc']

# min_calc: 0 -> None
#           1 -> calc_background_no_thermo
#           2 -> calc_background
#           3 -> calc_power_spectra


_supported_keys = []

_put_in_list = lambda x, output_vars: [np.asarray(x).flatten()]

_to_list = lambda x, output_vars: [np.asarray(xi).flatten() for xi in x]

_split_along_first = lambda x, output_vars: [x[i].flatten() for i in
                                             range(x.shape[0])]

_split_along_last = lambda x, output_vars: [x[..., i].flatten() for i in
                                            range(x.shape[-1])]

_from_dict = lambda x, output_vars: [np.asarray(x[var]).flatten() for var in
                                     output_vars]


# angular_diameter_distance

_angular_diameter_distance = OrderedDict(
    name='angular_diameter_distance',
    min_calc=1,
    output_vars=['angular_diameter_distance'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('angular_diameter_distance')


# angular_diameter_distance2

_angular_diameter_distance2 = OrderedDict(
    name='angular_diameter_distance2',
    min_calc=1,
    output_vars=['angular_diameter_distance2'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('angular_diameter_distance2')


# comoving_radial_distance

_comoving_radial_distance = OrderedDict(
    name='comoving_radial_distance',
    min_calc=1,
    output_vars=['comoving_radial_distance'],
    kwargs={'tol': 0.0001},
    get_output=_put_in_list,
)

_supported_keys.append('comoving_radial_distance')


# comoving_radial_distance

_conformal_time = OrderedDict(
    name='conformal_time',
    min_calc=1,
    output_vars=['conformal_time'],
    kwargs={'presorted': None, 'tol': None},
    get_output=_put_in_list,
)

_supported_keys.append('conformal_time')


# conformal_time_a1_a2

_conformal_time_a1_a2 = OrderedDict(
    name='conformal_time_a1_a2',
    min_calc=1,
    output_vars=['conformal_time_a1_a2'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('conformal_time_a1_a2')


# cosmomc_theta

_cosmomc_theta = OrderedDict(
    name='cosmomc_theta',
    min_calc=1,
    output_vars=['cosmomc_theta'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('cosmomc_theta')


# get_BAO: NotImplemented


# get_Omega

def _get_Omega_vars(kwargs):
    return ['Omega_' + kwargs['var']]

_get_Omega = OrderedDict(
    name='get_Omega',
    min_calc=1,
    output_vars=_get_Omega_vars,
    kwargs={'z': 0.},
    get_output=_put_in_list,
)

_supported_keys.append('get_Omega')


# get_background_densities

def _get_background_densities_vars(kwargs):
    return ['density_' + v for v in kwargs['vars']]

_get_background_densities = OrderedDict(
    name='get_background_densities',
    min_calc=1,
    output_vars=_get_background_densities_vars,
    kwargs={'vars': ['tot', 'K', 'cdm', 'baryon', 'photon', 'neutrino', 'nu',
            'de'], 'format': 'array'},
    get_output=_split_along_last,
)

_supported_keys.append('get_background_densities')


# get_background_outputs

_get_background_outputs = OrderedDict(
    name='get_background_outputs',
    min_calc=2,
    output_vars=['rs/DV', 'H', 'DA', 'F_AP'],
    kwargs={},
    get_output=_split_along_last,
)

_supported_keys.append('get_background_outputs')


# get_background_redshift_evolution

def _get_background_redshift_evolution_vars(kwargs):
    return list(kwargs['vars'])

_get_background_redshift_evolution = OrderedDict(
    name='get_background_redshift_evolution',
    min_calc=2,
    output_vars=_get_background_redshift_evolution_vars,
    kwargs={'vars': ['x_e', 'opacity', 'visibility', 'cs2b', 'T_b', 'dopacity',
            'ddopacity', 'dvisibility', 'ddvisibility'], 'format': 'array'},
    get_output=_split_along_last,
)

_supported_keys.append('get_background_redshift_evolution')


# get_background_time_evolution

def _get_background_time_evolution_vars(kwargs):
    return list(kwargs['vars'])

_get_background_time_evolution = OrderedDict(
    name='get_background_time_evolution',
    min_calc=2,
    output_vars=_get_background_time_evolution_vars,
    kwargs={'vars': ['x_e', 'opacity', 'visibility', 'cs2b', 'T_b', 'dopacity',
            'ddopacity', 'dvisibility', 'ddvisibility'], 'format': 'array'},
    get_output=_split_along_last,
)

_supported_keys.append('get_background_time_evolution')


# get_cmb_correlation_functions: NotImplemented


# get_cmb_power_spectra: NotImplemented


# get_cmb_transfer_data: NotImplemented


# get_cmb_unlensed_scalar_array_dict

_get_cmb_unlensed_scalar_array_dict = OrderedDict(
    name='get_cmb_unlensed_scalar_array_dict',
    min_calc=3,
    output_vars=['TxT', 'TxE', 'TxP', 'ExT', 'ExE', 'ExP', 'PxT', 'PxE', 'PxP'],
    kwargs={'params': None, 'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_from_dict,
)

_supported_keys.append('get_cmb_unlensed_scalar_array_dict')


# get_dark_energy_rho_w

_get_dark_energy_rho_w = OrderedDict(
    name='get_dark_energy_rho_w',
    min_calc=1,
    output_vars=['rho_de_at_a', 'w_at_a'],
    kwargs={},
    get_output=_to_list,
)

_supported_keys.append('get_dark_energy_rho_w')


# get_derived_params

_get_derived_params = OrderedDict(
    name='get_derived_params',
    min_calc=2,
    output_vars=['age', 'zstar', 'rstar', 'thetastar', 'DAstar', 'zdrag',
                 'rdrag', 'kd', 'thetad', 'zeq', 'keq', 'thetaeq', 'thetarseq'],
    kwargs={},
    get_output=_from_dict,
)

_supported_keys.append('get_derived_params')


# get_fsigma8

_get_fsigma8 = OrderedDict(
    name='get_fsigma8',
    min_calc=3,
    output_vars=['fsigma8'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('get_fsigma8')


# get_lens_potential_cls

_get_lens_potential_cls = OrderedDict(
    name='get_lens_potential_cls',
    min_calc=3,
    output_vars=['PP', 'PT', 'PE'],
    kwargs={'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_split_along_last,
)

_supported_keys.append('get_lens_potential_cls')


# get_lensed_gradient_cls

_get_lensed_gradient_cls = OrderedDict(
    name='get_lensed_gradient_cls',
    min_calc=3,
    output_vars=['TgradT', 'EgradE', 'BgradB', 'PPperp', 'TgradE', 'TPperp',
                 'gradT2', 'gradTgradT'],
    kwargs={'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_split_along_last,
)

_supported_keys.append('get_lensed_gradient_cls')


# get_lensed_scalar_cls

_get_lensed_scalar_cls = OrderedDict(
    name='get_lensed_scalar_cls',
    min_calc=3,
    output_vars=['TT', 'EE', 'BB', 'TE'],
    kwargs={'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_split_along_last,
)

_supported_keys.append('get_lensed_scalar_cls')


# get_linear_matter_power_spectrum: NotImplemented


# get_matter_power_interpolator: NotImplemented


# get_matter_power_spectrum: NotImplemented


# get_matter_transfer_data: NotImplemented


# get_nonlinear_matter_power_spectrum: NotImplemented


# get_redshift_evolution

def _get_redshift_evolution_vars(kwargs):
    return [v + '_z' for v in kwargs['vars']]

_get_redshift_evolution = OrderedDict(
    name='get_redshift_evolution',
    min_calc=2,
    output_vars=_get_redshift_evolution_vars,
    kwargs={'vars': ['k/h', 'delta_cdm', 'delta_baryon', 'delta_photon',
            'delta_neutrino', 'delta_nu', 'delta_tot', 'delta_nonu',
            'delta_tot_de', 'Weyl', 'v_newtonian_cdm', 'v_newtonian_baryon',
            'v_baryon_cdm', 'a', 'etak', 'H', 'growth', 'v_photon', 'pi_photon',
            'E_2', 'v_neutrino', 'T_source', 'E_source',
            'lens_potential_source'], 'lAccuracyBoost': 4},
    get_output=_split_along_last,
)

_supported_keys.append('get_redshift_evolution')


# get_sigma8

_get_sigma8 = OrderedDict(
    name='sigma8',
    min_calc=3,
    output_vars=['sigma8'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('get_sigma8')


# get_sigma8_0

_get_sigma8_0 = OrderedDict(
    name='get_sigma8_0',
    min_calc=3,
    output_vars=['sigma8_0'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('get_sigma8_0')


# get_sigmaR

_get_sigmaR = OrderedDict(
    name='get_sigmaR',
    min_calc=3,
    output_vars=['sigmaR'],
    kwargs={'z_indices': None, 'var1': None, 'var2': None, 'hubble_units': True,
            'return_R_z': False},
    get_output=_put_in_list,
)

_supported_keys.append('get_sigmaR')


# get_source_cls_dict

_get_source_cls_dict = OrderedDict(
    name='get_source_cls_dict',
    min_calc=3,
    output_vars=['PxP'],
    kwargs={'params': None, 'lmax': None, 'raw_cl': False},
    get_output=_from_dict,
)

_supported_keys.append('get_source_cls_dict')


# get_tensor_cls

_get_tensor_cls = OrderedDict(
    name='get_tensor_cls',
    min_calc=3,
    output_vars=['TT_tensor', 'EE_tensor', 'BB_tensor', 'TE_tensor'],
    kwargs={'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_split_along_last,
)

_supported_keys.append('get_tensor_cls')


# get_time_evolution

def _get_time_evolution_vars(kwargs):
    return [v + '_eta' for v in kwargs['vars']]

_get_time_evolution = OrderedDict(
    name='get_time_evolution',
    min_calc=2,
    output_vars=_get_time_evolution_vars,
    kwargs={'vars': ['k/h', 'delta_cdm', 'delta_baryon', 'delta_photon',
            'delta_neutrino', 'delta_nu', 'delta_tot', 'delta_nonu',
            'delta_tot_de', 'Weyl', 'v_newtonian_cdm', 'v_newtonian_baryon',
            'v_baryon_cdm', 'a', 'etak', 'H', 'growth', 'v_photon', 'pi_photon',
            'E_2', 'v_neutrino', 'T_source', 'E_source',
            'lens_potential_source'], 'lAccuracyBoost': 4, 'frame': 'CDM'},
    get_output=_split_along_last,
)

_supported_keys.append('get_time_evolution')


# get_total_cls

_get_total_cls = OrderedDict(
    name='get_total_cls',
    min_calc=3,
    output_vars=['TT_total', 'EE_total', 'BB_total', 'TE_total'],
    kwargs={'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_split_along_last,
)

_supported_keys.append('get_total_cls')


# get_unlensed_scalar_array_cls: NotImplemented


# get_unlensed_scalar_cls

_get_unlensed_scalar_cls = OrderedDict(
    name='get_unlensed_scalar_cls',
    min_calc=3,
    output_vars=['TT_unlensed_scalar', 'EE_unlensed_scalar',
                 'BB_unlensed_scalar', 'TE_unlensed_scalar'],
    kwargs={'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_split_along_last,
)

_supported_keys.append('get_unlensed_scalar_cls')


# get_unlensed_total_cls

_get_unlensed_total_cls = OrderedDict(
    name='get_unlensed_total_cls',
    min_calc=3,
    output_vars=['TT_unlensed_total', 'EE_unlensed_total', 'BB_unlensed_total',
                 'TE_unlensed_total'],
    kwargs={'lmax': None, 'CMB_unit': None, 'raw_cl': False},
    get_output=_split_along_last,
)

_supported_keys.append('get_unlensed_total_cls')


# h_of_z

_h_of_z = OrderedDict(
    name='h_of_z',
    min_calc=1,
    output_vars=['h_of_z'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('h_of_z')


# hubble_parameter

_hubble_parameter = OrderedDict(
    name='hubble_parameter',
    min_calc=1,
    output_vars=['hubble_parameter'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('hubble_parameter')


# luminosity_distance

_luminosity_distance = OrderedDict(
    name='luminosity_distance',
    min_calc=1,
    output_vars=['luminosity_distance'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('luminosity_distance')


# physical_time

_physical_time = OrderedDict(
    name='physical_time',
    min_calc=1,
    output_vars=['physical_time'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('physical_time')


# physical_time_a1_a2

_physical_time_a1_a2 = OrderedDict(
    name='physical_time_a1_a2',
    min_calc=1,
    output_vars=['physical_time_a1_a2'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('physical_time_a1_a2')


# power_spectra_from_transfer: NotImplemented


# redshift_at_comoving_radial_distance

_redshift_at_comoving_radial_distance = OrderedDict(
    name='redshift_at_comoving_radial_distance',
    min_calc=2,
    output_vars=['redshift_at_chi'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('redshift_at_comoving_radial_distance')


# redshift_at_conformal_time

_redshift_at_conformal_time = OrderedDict(
    name='redshift_at_conformal_time',
    min_calc=2,
    output_vars=['redshift_at_eta'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('redshift_at_conformal_time')


# sound_horizon

_sound_horizon = OrderedDict(
    name='sound_horizon',
    min_calc=1,
    output_vars=['r_s'],
    kwargs={},
    get_output=_put_in_list,
)

_supported_keys.append('sound_horizon')


class CCalcBase:
    '''Base class for CAMB calc model.'''
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('abstract method.')

    @property
    def min_calc(self):
        raise NotImplementedError('abstract property.')

    @property
    def output_vars(self):
        raise NotImplementedError('abstract property.')

    def get(self, camb_data, tmp_dict):
        raise NotImplementedError('abstract property.')

    __call__ = get


class CCalc(CCalcBase):
    '''Default CAMB calc model.'''
    def __init__(self, name, min_calc=None, output_vars=None, kwargs=None,
                 get_output=None):
        self.name = name
        if min_calc is None:
            self.min_calc = self._element_dict['min_calc']
        else:
            self.min_calc = min_calc
        if output_vars is None:
            self.output_vars = self._element_dict['output_vars']
        else:
            self.output_vars = output_vars
        k = self._element_dict['kwargs']
        if kwargs is not None:
            k.update(kwargs)
        self.kwargs = k
        if get_output is None:
            self.get_output = self._element_dict['get_output']
        else:
            self.get_output = get_output

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n):
        try:
            exec("assert _" + n + "['name'] == n")
            exec("self._element_dict = _" + n)
            self._name = n
        except Exception:
            self._element_dict = None
            raise ValueError('unsupported calc element: {}.'.format(n))

    @property
    def min_calc(self):
        return self._min_calc

    @min_calc.setter
    def min_calc(self, mc):
        try:
            mc = int(mc)
            assert 0 <= mc <= 3
            self._min_calc = mc
        except Exception:
            self._min_calc = None
            raise ValueError('invalid value for min_calc.')

    @property
    def output_vars(self):
        if callable(self._output_vars):
            return self._output_vars(self.kwargs)
        else:
            return self._output_vars

    @output_vars.setter
    def output_vars(self, ov):
        if callable(ov):
            self._output_vars = ov
        elif isinstance(ov, str):
            self._output_vars = [ov]
        else:
            try:
                self._output_vars = list(ov)
            except Exception:
                self._output_vars = None
                raise ValueError('invalid value for output_vars.')

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k):
        try:
            self._kwargs = dict(k)
        except Exception:
            self._kwargs = None
            raise ValueError('invalid value for kwargs.')

    @property
    def get_output(self):
        return self._get_output

    @get_output.setter
    def get_output(self, p):
        self._get_output = p if callable(p) else (lambda *args: args)

    def get(self, camb_data, tmp_dict):
        try:
            exec('self._f = camb_data.' + self.name)
            r = self.get_output(self._f(**self.kwargs), self.output_vars)
            assert len(r) == len(self.output_vars)
            for i, k in enumerate(self.output_vars):
                tmp_dict[k] = r[i]
        except Exception:
            raise RuntimeError(
                'failed to do CAMB calc for {}.'.format(self.name))

    __call__ = get
