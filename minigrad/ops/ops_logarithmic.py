"""
minigrad/ops/ops_logarithmic.py
================================
Numerically-stable logarithmic / softmax operators.

Key insight for LogSumExp:
  log(sum(exp(z))) = log(exp(M) * sum(exp(z - M))) = M + log(sum(exp(z - M)))
where M = max(z).  This avoids overflow while preserving accuracy.

Based on CMU 10-714 HW2.
"""
import numpy as array_api
from minigrad.autograd import TensorOp, Tensor
from minigrad.ops.ops_mathematic import (
    exp, log, summation, reshape, broadcast_to
)


class LogSumExp(TensorOp):
    """
    Numerically stable log-sum-exp over specified axes.

      LogSumExp(z, axes) = log(sum(exp(z - max(z)), axis=axes)) + max(z)
    """

    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, Z):
        # Max for numerical stability; keep dims for broadcasting
        M = array_api.max(Z, axis=self.axes, keepdims=True)
        M_sq = array_api.squeeze(M, axis=self.axes) if self.axes is not None else M.reshape(())
        return array_api.log(array_api.sum(array_api.exp(Z - M), axis=self.axes)) + M_sq

    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]
        # Reconstruct max along axes with keepdims
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(Z.shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        # Build shape for broadcasting: replace reduced axes with 1
        reduced_shape = list(Z.shape)
        for ax in axes:
            reduced_shape[ax] = 1

        # Compute softmax gradient
        M_raw = Tensor(array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True),
                       requires_grad=False)
        exp_Z = exp(Z - broadcast_to(M_raw, Z.shape))
        sum_exp = broadcast_to(reshape(summation(exp_Z, axes=self.axes), reduced_shape), Z.shape)
        softmax = exp_Z / sum_exp

        # Chain rule: out_grad broadcasts back
        out_grad_expanded = broadcast_to(
            reshape(out_grad, reduced_shape), Z.shape
        )
        return out_grad_expanded * softmax


def logsumexp(a, axes=None):
    return LogSumExp(axes)(a)


class LogSoftmax(TensorOp):
    """
    Numerically stable log-softmax along axis=1 (2D input assumed).

      LogSoftmax(z) = z - LogSumExp(z, axis=1, keepdims=True)
    """

    def compute(self, Z):
        M = array_api.max(Z, axis=1, keepdims=True)
        log_sum = array_api.log(array_api.sum(array_api.exp(Z - M), axis=1, keepdims=True))
        return Z - M - log_sum

    def gradient(self, out_grad: Tensor, node: Tensor):
        # d/dz_i [log_softmax(z)]_j = delta_{ij} - softmax(z)_j
        Z = node.inputs[0]
        softmax_Z = exp(log_softmax(Z))
        sum_og = broadcast_to(
            reshape(summation(out_grad, axes=(1,)), (out_grad.shape[0], 1)),
            out_grad.shape
        )
        return out_grad - softmax_Z * sum_og


def log_softmax(a):
    return LogSoftmax()(a)