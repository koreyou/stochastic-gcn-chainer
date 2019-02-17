from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "sampling",
    sources=["sampling.pyx", "sampling_c.cpp"],
    include_dirs=['.', np.get_include()],
    language='c++',
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)
setup(name="stochastic-gcn", ext_modules=cythonize([ext]))

