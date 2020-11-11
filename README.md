# CosmoFast

![python package](https://github.com/HerculesJack/cosmofast/workflows/python%20package/badge.svg)
[![codecov](https://codecov.io/gh/HerculesJack/cosmofast/branch/master/graph/badge.svg)](https://codecov.io/gh/HerculesJack/cosmofast)
![PyPI](https://img.shields.io/pypi/v/cosmofast)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/cosmofast)
[![Documentation Status](https://readthedocs.org/projects/cosmofast/badge/?version=latest)](https://cosmofast.readthedocs.io/en/latest/?badge=latest)

CosmoFast is an add-on package for
[BayesFast](https://github.com/HerculesJack/bayesfast)
developed by [He Jia](http://hejia.io) and 
[Uros Seljak](https://physics.berkeley.edu/people/faculty/uros-seljak),
which provides several frequently-used cosmological modules.

## Installation

We plan to add pypi and conda-forge support later. For now, please first install
[BayesFast](https://github.com/HerculesJack/bayesfast),
and then install CosmoFast from source with:

```
git clone https://github.com/HerculesJack/cosmofast
cd cosmofast
pip install -e .
# you can drop the -e option if you don't want to use editable mode
# but note that pytest may not work correctly in this case
```

To check if CosmoFast is built correctly, you can do:

```
pytest # for this you will need to have pytest and numdifftools installed
```

## Dependencies

CosmoFast requires python>=3.6, astropy, bayesfast, camb, cython, numpy and
scipy. Currently, it has been tested on Ubuntu and MacOS, with python 3.6-3.8.

## License

CosmoFast is distributed under the Apache License, Version 2.0.

## Citing CosmoFast

If you find CosmoFast useful for your research,
please consider citing our papers accordingly:

* He Jia and Uros Seljak,
*BayesFast: A Fast and Scalable Method for Cosmological Bayesian Inference*,
in prep (for posterior sampling)
* He Jia and Uros Seljak,
*Normalizing Constant Estimation with Gaussianized Bridge Sampling*,
[AABI 2019 Proceedings, PMLR 118:1-14](http://proceedings.mlr.press/v118/jia20a.html)
(for evidence estimation)
