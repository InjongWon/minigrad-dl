"""setup.py — build C++ extensions and install MiniGrad."""
from setuptools import setup, find_packages, Extension
import pybind11
import sys

# C++ CPU extension
cpu_ext = Extension(
    "minigrad.backend_ndarray.ndarray_backend_cpu",
    sources=["minigrad/backend_ndarray/ndarray_backend_cpu.cc"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O2", "-std=c++14"],
    language="c++",
)

# CUDA extension (optional)
extensions = [cpu_ext]
try:
    from torch.utils.cpp_extension import CUDAExtension
except ImportError:
    pass  # CUDA extension built separately via Makefile

setup(
    name="minigrad",
    version="0.1.0",
    description="Deep learning system from scratch (CMU 10-714)",
    packages=find_packages(),
    ext_modules=extensions,
    python_requires=">=3.8",
    install_requires=["numpy", "pybind11"],
    extras_require={"dev": ["pytest", "matplotlib"]},
)