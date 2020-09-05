from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
from extension_helpers import add_openmp_flags_if_available
import warnings


ext_modules = [
    Extension(
        "cosmofast.planck_18._commander",
        ["cosmofast/planck_18/_commander.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"],
    ),
    Extension(
        "cosmofast.planck_18._plik_lite",
        ["cosmofast/planck_18/_plik_lite.pyx"],
        # include_dirs=[np.get_include()],
        # libraries=["m"],
    ),
    Extension(
        "cosmofast.planck_18._simall",
        ["cosmofast/planck_18/_simall.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"],
    ),
    Extension(
        "cosmofast.planck_18._smica",
        ["cosmofast/planck_18/_smica.pyx"],
        # include_dirs=[np.get_include()],
        # libraries=["m"],
    ),
]


openmp_added = [add_openmp_flags_if_available(_) for _ in ext_modules]
if not all(openmp_added):
    warnings.warn('OpenMP check failed. Compiling without it for now.',
                  RuntimeWarning)


setup(
    name='cosmofast',
    version='0.1.0.dev2',
    author='He Jia and Uros Seljak',
    maintainer='He Jia',
    maintainer_email='he.jia.phy@gmail.com',
    description='Cosmology add-ons for the BayesFast package.',
    url='https://github.com/HerculesJack/cosmofast',
    license='Apache License, Version 2.0',
    python_requires=">=3.6",
    install_requires=['astropy', 'bayesfast', 'cython', 'extension-helpers',
                      'numpy', 'scipy'],
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, language_level="3"),
)
