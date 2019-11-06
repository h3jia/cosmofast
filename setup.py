from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = []

setup(
    name='cosmofast',
    version='0.1.0dev1',
    ext_modules=cythonize(ext_modules, language_level = "3"),
)
