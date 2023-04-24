# setup.py
#
# from setuptools import setup
#
# setup()

import platform
import os
import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


if platform.system() == "Windows":
    compile_extra_args = []
    link_extra_args = []
elif platform.system() == "Linux":
    compile_extra_args = ["-O3"]
    link_extra_args = ["-O3"]
elif platform.system() == "Darwin":
    compile_extra_args = ["-O3", "-std=c++11", "-mmacosx-version-min=10.9"]
    link_extra_args = ["-O3", "-stdlib=libc++", "-mmacosx-version-min=10.9"]

slash = os.path.sep
extensions = [
    Extension(
        'hisel.lar.lar',
        [f'hisel{slash}lar{slash}lar.pyx'],
        language='c++',
        include_dirs=['hisel/', numpy.get_include()],
        extra_compile_args=compile_extra_args,
        extra_link_args=link_extra_args,
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
