# setup.py
#
# from setuptools import setup
#
# setup()

import os
import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

slash = os.path.sep
extensions = [
    Extension(
        'hisel.lar.lar',
        [f'hisel{slash}lar{slash}lar.pyx'],
        language='c++',
        include_dirs=['hisel/', numpy.get_include()]
    )
]

setup(
    name='hisel',
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    script_args=['build_ext'],
    options={'build_ext': {'inplace': True, 'force': True}},
    packages=setuptools.find_packages(),
)
