#!/usr/bin/env python3
"""
Build pybind11 extensions for the NDArray CPU/CUDA backends.

Install (editable, recommended):
  pip install -e ".[dev]"

IDE compile flags (clangd):
  cd minigrad/backend_ndarray && mkdir -p build && cd build && cmake ..
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pybind11 is required to build native backends. "
        "Run: pip install pybind11"
    ) from e

ROOT = Path(__file__).resolve().parent


def _extension(name: str, rel_source: str) -> Pybind11Extension:
    src = str(ROOT / rel_source)
    extra: list[str] = []
    if rel_source.endswith(".cu"):
        # File is host C++ today; compile as C++ even when the suffix is .cu.
        extra.append("-x")
        extra.append("c++")
    return Pybind11Extension(
        name,
        [src],
        cxx_std=17,
        extra_compile_args=extra,
    )


ext_modules = [
    _extension(
        "minigrad.backend_ndarray.ndarray_backend_cpu",
        "minigrad/backend_ndarray/ndarray_backend_cpu.cc",
    ),
]

# Optional real CUDA build: set MINIGRAD_NVCC=1 and have nvcc on PATH to compile .cu with nvcc.
# Otherwise the CUDA module is still built with the C++ compiler (host-only bootstrap).
if os.environ.get("MINIGRAD_NVCC") == "1" and shutil.which("nvcc"):
    ext_modules.append(
        Pybind11Extension(
            "minigrad.backend_ndarray.ndarray_backend_cuda",
            [str(ROOT / "minigrad/backend_ndarray/ndarray_backend_cuda.cu")],
            cxx_std=17,
        )
    )
else:
    ext_modules.append(
        _extension(
            "minigrad.backend_ndarray.ndarray_backend_cuda",
            "minigrad/backend_ndarray/ndarray_backend_cuda.cu",
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
