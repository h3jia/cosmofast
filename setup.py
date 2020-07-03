from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
        "cosmofast.planck_2018._commander",
        ["cosmofast/planck_2018/_commander.pyx"],
        include_dirs=[np.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        libraries=["m"],
    ),
    Extension(
        "cosmofast.planck_2018._plik_lite",
        ["cosmofast/planck_2018/_plik_lite.pyx"],
        include_dirs=[np.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        # libraries=["m"],
    ),
    Extension(
        "cosmofast.planck_2018._simall",
        ["cosmofast/planck_2018/_simall.pyx"],
        include_dirs=[np.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        libraries=["m"],
    ),
    Extension(
        "cosmofast.planck_2018._smica",
        ["cosmofast/planck_2018/_smica.pyx"],
        include_dirs=[np.get_include()],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        libraries=["m"],
    ),
]

setup(
    name='cosmofast',
    version='0.1.0dev1',
    author='He Jia and Uros Seljak',
    maintainer='He Jia',
    maintainer_email='he.jia.phy@gmail.com',
    description='Cosmology add-ons for the BayesFast package.',
    url='https://github.com/HerculesJack/cosmofast',
    license='Apache License, Version 2.0',
    python_requires=">=3",
    install_requires=['astropy', 'bayesfast', 'cython', 'numpy', 'scipy'],
    ext_modules=cythonize(ext_modules, language_level="3"),
)
