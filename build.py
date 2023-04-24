import os
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


def build(setup_kwargs):
    print('Adding extensions:\n{extensions}\n')
    setup_kwargs.update({
        'ext_modules': extensions,
        'cmdclass': {'build_ext': build_ext},
    })
