"""
minigrad/nn/nn_basic.py
========================
Core neural network modules.

Implements:
  Linear, ReLU, Sequential, LayerNorm1d, BatchNorm1d/2d,
  Dropout, Softmax, SoftmaxLoss, Embedding

Based on CMU 10-714 HW2.
"""
import numpy as np
from minigrad.autograd import Tensor
import minigrad.init as init
from minigrad.ops.ops_mathematic import (
    matmul, add_scalar, broadcast_to, reshape,
    summation, power_scalar, divide, multiply, negate
)
from minigrad.ops.ops_logarithmic import logsumexp


# ---------------------------------------------------------------------------
# Parameter (marks a Tensor as a learnable parameter)
# ---------------------------------------------------------------------------
class Parameter(Tensor):
    """A learnable parameter in a Module."""
    pass


def _unpack_params(value):
    """Recursively collect all Parameters from a value."""
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, dict):
        return sum([_unpack_params(v) for v in value.values()], [])
    if isinstance(value, (list, tuple)):
        return sum([_unpack_params(v) for v in value], [])
    if isinstance(value, Module):
        return value.parameters()
    return []


def _child_modules(value):
    """Recursively collect all Module children."""
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        return sum([_child_modules(v) for v in value.values()], [])
    if isinstance(value, (list, tuple)):
        return sum([_child_modules(v) for v in value], [])
    return []


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------
class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        return _unpack_params(self.__dict__)

    def _children(self):
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------
class Linear(Module):
    """
    y = xW^T + b

    Weight init: Kaiming Uniform (fan_in = in_features)
    Bias init:   Kaiming Uniform (fan_in = out_features)

    Note: weight is initialized before bias (required for reproducibility
    with reference test answers).
    """

    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features,
                                 device=device, dtype=dtype)
        )
        self.bias = None
        if bias:
            # Note: fan_in = out_features for the bias vector
            b = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = Parameter(reshape(b, (out_features,)))

    def forward(self, X: Tensor) -> Tensor:
        out = matmul(X, self.weight)
        if self.bias is not None:
            out = out + broadcast_to(self.bias, out.shape)
        return out


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from minigrad.ops.ops_mathematic import relu
        return relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        from minigrad.ops.ops_mathematic import tanh
        return tanh(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.ones_like(x.numpy())) / (
            Tensor(np.ones_like(x.numpy())) + (- x).exp()
        )


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------
class Sequential(Module):
    """Apply modules in order."""

    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class SoftmaxLoss(Module):
    """
    Numerically stable softmax cross-entropy loss.
    Takes raw logits and integer labels (not one-hot).
    """

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        n, k = logits.shape
        # LogSumExp over classes
        lse = logsumexp(logits, axes=(1,))
        # Sum of log-probabilities for true classes
        one_hot = init.one_hot(k, y.numpy().astype(int),
                               dtype=logits.dtype, requires_grad=False)
        true_logits = summation(logits * one_hot, axes=(1,))
        return summation(lse - true_logits) / Tensor(np.array(n, dtype=logits.dtype))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class LayerNorm1d(Module):
    """
    Layer normalization (Ba et al., 2016).
    Input: (N, D) — normalizes over D features per sample.
    """

    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        mean = broadcast_to(
            reshape(summation(x, axes=(1,)) / D, (N, 1)), x.shape
        )
        var = broadcast_to(
            reshape(summation((x - mean) ** 2, axes=(1,)) / D, (N, 1)),
            x.shape
        )
        x_norm = (x - mean) / (var + self.eps) ** 0.5
        w = broadcast_to(reshape(self.weight, (1, D)), x.shape)
        b = broadcast_to(reshape(self.bias, (1, D)), x.shape)
        return w * x_norm + b


class BatchNorm1d(Module):
    """
    Batch normalization over (N, D) inputs.
    Maintains running mean/var for inference.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        if self.training:
            mean = summation(x, axes=(0,)) / N
            var = summation((x - broadcast_to(reshape(mean, (1, D)), x.shape)) ** 2,
                            axes=(0,)) / N
            # Update running stats (no grad)
            self.running_mean.data = Tensor(
                (1 - self.momentum) * self.running_mean.numpy() +
                self.momentum * mean.numpy(),
                requires_grad=False,
            )
            self.running_var.data = Tensor(
                (1 - self.momentum) * self.running_var.numpy() +
                self.momentum * var.numpy(),
                requires_grad=False,
            )
        else:
            mean = self.running_mean
            var = self.running_var

        mean_b = broadcast_to(reshape(mean, (1, D)), x.shape)
        var_b = broadcast_to(reshape(var, (1, D)), x.shape)
        x_norm = (x - mean_b) / (var_b + self.eps) ** 0.5
        w = broadcast_to(reshape(self.weight, (1, D)), x.shape)
        b = broadcast_to(reshape(self.bias, (1, D)), x.shape)
        return w * x_norm + b


class BatchNorm2d(Module):
    """
    Batch normalization for (N, C, H, W) inputs.
    Wraps BatchNorm1d after reshaping spatial dims.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 device=None, dtype="float32"):
        super().__init__()
        self.bn = BatchNorm1d(num_features, eps=eps, momentum=momentum,
                              device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        # (N, C, H, W) -> (N*H*W, C)
        x_2d = reshape(x.transpose((0, 2, 3, 1)), (N * H * W, C))
        out_2d = self.bn(x_2d)
        # (N*H*W, C) -> (N, C, H, W)
        return reshape(out_2d, (N, H, W, C)).transpose((0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------
class Dropout(Module):
    """Randomly zero elements with probability p during training."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
        return x * Tensor(mask.astype(np.float32), requires_grad=False)


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------
class Residual(Module):
    """x + fn(x)"""

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


# ---------------------------------------------------------------------------
# Embedding (lookup table for sequence models)
# ---------------------------------------------------------------------------
class Embedding(Module):
    """
    Lookup table of shape (num_embeddings, embedding_dim).
    Forward: given integer indices, returns corresponding row vectors.
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: integer index tensor of shape (*)
        # returns: (*) × embedding_dim
        indices = x.numpy().astype(int).flatten()
        one_hot = np.eye(self.num_embeddings, dtype=np.float32)[indices]
        one_hot_t = Tensor(one_hot, requires_grad=False)
        out = matmul(one_hot_t, self.weight)
        new_shape = x.shape + (self.embedding_dim,)
        return reshape(out, new_shape)