from bayesfast.utils.collections import PropertyList

__all__ = ['COutputBase', 'COutput']


class COutputBase:
    '''Base class for CAMB output model.'''
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('abstract method.')

    @property
    def output_vars(self):
        raise NotImplementedError('abstract property.')

    def get(self, tmp_dict):
        raise NotImplementedError('abstract method.')


class COutput(COutputBase):
    '''Default CAMB output model.'''
    def __init__(self, output_vars=None, get=None):
        self.output_vars = output_vars
        self.get = get

    @property
    def output_vars(self):
        return self._output_vars

    @output_vars.setter
    def output_vars(self, ov):
        if ov is None:
            self._output_vars = None
        elif isinstance(ov, str):
            self._output_vars = [ov]
        else:
            try:
                self._output_vars = list(ov)
            except Exception:
                self._output_vars = None
                raise ValueError('invalid value for output_vars.')

    @property
    def get(self):
        return self._get

    @get.setter
    def get(self, g):
        if g is None:
            self._get = lambda tmp_dict: tmp_dict
        elif callable(g):
            self._get = g
        else:
            raise ValueError('invalid value for get.')
