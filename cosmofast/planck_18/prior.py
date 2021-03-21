from bayesfast.modules import Gaussian

__all__ = ['APlanckPrior']


class APlanckPrior(Gaussian):
    """
    Prior for the a_planck parameter in Planck 2018 likelihoods.
    """
    def __init__(self, loc=1., scale=0.0025, input_vars='a_planck',
                 output_vars='logprior_a_planck', delete_vars=(), label=None):
        super().__init__(mean=loc, cov=scale**2, input_vars=input_vars,
                         output_vars=output_vars, delete_vars=delete_vars,
                         label=label)
