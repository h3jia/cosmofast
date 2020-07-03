# CosmoFast

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
```

## Dependencies

CosmoFast depends on astropy, bayesfast, cython, numpy and scipy.
Currently, it is only tested on Linux with Python 3.6.

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
