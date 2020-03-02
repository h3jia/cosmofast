from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = []

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
