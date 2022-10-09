# CosmoFast

![python package](https://github.com/h3jia/cosmofast/workflows/python%20package/badge.svg)
[![codecov](https://codecov.io/gh/h3jia/cosmofast/branch/master/graph/badge.svg)](https://codecov.io/gh/h3jia/cosmofast)
![PyPI](https://img.shields.io/pypi/v/cosmofast)
![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/cosmofast)
[![Documentation Status](https://readthedocs.org/projects/cosmofast/badge/?version=latest)](https://cosmofast.readthedocs.io/en/latest/?badge=latest)

CosmoFast is a collection of differentiable cosmological modules, developed by
[He Jia](http://hejia.io) and
[Uros Seljak](https://physics.berkeley.edu/people/faculty/uros-seljak).
It is intended as an add-on package for [BayesFast](https://github.com/h3jia/bayesfast), but can
also be used standalone.
Feel free to contact [He Jia](mailto:he.jia.phy@gmail.com) if you would like to add your own modules
to CosmoFast!

## Links

* BayesFast Website: https://www.bayesfast.org/
* Documentation: https://cosmofast.readthedocs.io/en/latest/
* Source Code: https://github.com/h3jia/cosmofast
* Bug Reports: https://github.com/h3jia/cosmofast/issues

## What's New

We are upgrading BayesFast & CosmoFast to v0.2 with JAX, which would be faster, more accueate, and
much easier to use than the previous version!

## Installation

We plan to add pypi and conda-forge support later.
For now, please install CosmoFast from source with:

```
git clone https://github.com/h3jia/cosmofast
cd cosmofast
pip install -e .
# you can drop the -e option if you don't want to use editable mode
# but note that pytest may not work correctly in this case
```

To check if CosmoFast is built correctly, you can do:

```
pytest # for this you will need to have pytest installed
```

## Dependencies

CosmoFast requires python>=3.7, cython, extension-helpers, jax>=0.3, jaxlib>=0.3 and numpy>=1.17.
Currently, it has been tested on Ubuntu and MacOS, with python 3.7-3.10.

## Available Modules

* Planck 2018 likelihoods `cosmofast.planck_18`: Plik Lite high-l TT & TTTEEE, Commander low-l TT,
Simall low-l EE & BB, Smica lensing full & CMB marginalized. All of these likelihoods are rewritten
using JAX. Some of them are diagonalized for better performance with BayesFast.
* Dark Energy Survey Y1 3x2 likelihood `cosmofast.des_y1`: coming soon.
* Pantheon 2022 likelihood `cosmofast.pantheon_22`: coming soon.

## License

CosmoFast is distributed under the Apache License, Version 2.0.

## Citing CosmoFast

If you find CosmoFast useful for your research, please consider citing our papers accordingly:

* He Jia and Uros Seljak,
*BayesFast: A Fast and Scalable Method for Cosmological Bayesian Inference*,
in prep (for posterior sampling)
* He Jia and Uros Seljak,
*Normalizing Constant Estimation with Gaussianized Bridge Sampling*,
[AABI 2019 Proceedings, PMLR 118:1-14](http://proceedings.mlr.press/v118/jia20a.html)
(for evidence estimation)
