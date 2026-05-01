"""
minigrad/init/__init__.py
=========================
Weight initializers.

Implements:
  - xavier_uniform / xavier_normal  [Glorot & Bengio, 2010]
  - kaiming_uniform / kaiming_normal [He et al., 2015]

Based on CMU 10-714 HW2.
"""
import numpy as np
import math
from minigrad.autograd import Tensor


# ---------------------------------------------------------------------------
# Primitive random init helpers
# ---------------------------------------------------------------------------
def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """Uniform random [low, high)."""
    arr = np.random.uniform(low, high, shape).astype(dtype)
    return Tensor(arr, device=device, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """Normal random N(mean, std²)."""
    arr = (np.random.randn(*shape) * std + mean).astype(dtype)
    return Tensor(arr, device=device, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    arr = np.full(shape, c, dtype=dtype)
    return Tensor(arr, device=device, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(*shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(*shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    arr = np.eye(n, dtype=dtype)[i]
    return Tensor(arr, device=device, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Xavier (Glorot) initializers
# ---------------------------------------------------------------------------
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    """
    U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).

    Reference: Glorot & Bengio (2010)
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    """
    N(0, std²) where std = gain * sqrt(2 / (fan_in + fan_out)).

    Reference: Glorot & Bengio (2010)
    """
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)


# ---------------------------------------------------------------------------
# Kaiming (He) initializers
# ---------------------------------------------------------------------------
def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    """
    U(-bound, bound) where bound = gain * sqrt(3 / fan_in).
    Recommended gain for ReLU = sqrt(2).

    Reference: He et al. (2015)
    https://arxiv.org/pdf/1502.01852.pdf
    """
    gain = math.sqrt(2.0)  # ReLU gain
    bound = gain * math.sqrt(3.0 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    """
    N(0, std²) where std = gain / sqrt(fan_in).
    Recommended gain for ReLU = sqrt(2).

    Reference: He et al. (2015)
    """
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
