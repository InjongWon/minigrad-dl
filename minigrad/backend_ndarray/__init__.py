"""NDArray backend (Python layer + optional C++/CUDA extensions)."""

from .ndarray import NDArray, BackendDevice, cpu, cuda, numpy

__all__ = ["NDArray", "BackendDevice", "cpu", "cuda", "numpy"]
