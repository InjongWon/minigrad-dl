"""
minigrad/ops/ops_mathematic.py
==============================
All core mathematical operators with analytical backward passes.

Each Op subclass:
  - compute(*args: NDArray) -> NDArray   [forward on raw data]
  - gradient(out_grad, node) -> Tensor | Tuple[Tensor]  [VJP]

Based on CMU 10-714 HW1 + HW4.
"""
import numpy as array_api
from minigrad.autograd import TensorOp, Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Element-wise binary ops
# ---------------------------------------------------------------------------
class EWiseAdd(TensorOp):
    def compute(self, a, b):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a, b):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Raise a tensor to an integer power (scalar exponent)."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a):
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a = node.inputs[0]
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Element-wise power: a ** b."""

    def compute(self, a, b):
        return a ** b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        grad_a = out_grad * b * power_scalar(a, b - 1)
        grad_b = out_grad * log(a) * EWisePow()(a, b)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------
class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        # Handle broadcasting: if one input had fewer dims, sum over extra dims
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)

        # Reduce extra batch dimensions if needed
        if len(grad_a.shape) > len(a.shape):
            extra = len(grad_a.shape) - len(a.shape)
            grad_a = summation(grad_a, axes=tuple(range(extra)))
        if len(grad_b.shape) > len(b.shape):
            extra = len(grad_b.shape) - len(b.shape)
            grad_b = summation(grad_b, axes=tuple(range(extra)))

        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Transpose(TensorOp):
    """Swap two axes (default: last two)."""

    def __init__(self, axes: Optional[Tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        # Find axes that were broadcast (added or expanded)
        out_shape = self.shape
        ndim_diff = len(out_shape) - len(input_shape)
        # Pad input shape on the left
        padded = (1,) * ndim_diff + tuple(input_shape)
        axes = tuple(
            i for i, (si, so) in enumerate(zip(padded, out_shape)) if si != so
        )
        grad = summation(out_grad, axes=axes)
        return reshape(grad, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_shape = node.inputs[0].shape
        # Restore reduced dims as size-1 for broadcasting
        grad_shape = list(input_shape)
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        for ax in axes:
            grad_shape[ax] = 1
        return broadcast_to(reshape(out_grad, grad_shape), input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------
class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    """
    ReLU: max(0, x).
    Gradient: indicator I{x > 0}.
    Note: we access .realize_cached_data() here since ReLU is not
    twice-differentiable anyway; this avoids a circular dependency.
    """

    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        cached = node.realize_cached_data()
        mask = Tensor(cached > 0, requires_grad=False, dtype=out_grad.dtype)
        return out_grad * mask


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # d/dx tanh(x) = 1 - tanh²(x)
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)


def tanh(a):
    return Tanh()(a)


# ---------------------------------------------------------------------------
# Stack / Split (needed for LSTM / Transformer)
# ---------------------------------------------------------------------------
class Stack(TensorOp):
    """
    Stack a list of tensors along a new axis.
    All tensors must have the same shape.
    """

    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, *args):
        return array_api.stack(args, axis=self.axis)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return split(out_grad, self.axis)


def stack(tensors, axis):
    return Stack(axis)(*tensors)


class Split(TensorOp):
    """
    Split a tensor along an axis, returning a tuple of slices
    with that axis removed (inverse of Stack).
    """

    def __init__(self, axis: int):
        self.axis = axis

    def compute(self, a):
        n = a.shape[self.axis]
        slices = array_api.split(a, n, axis=self.axis)
        return tuple(s.squeeze(axis=self.axis) for s in slices)

    def gradient(self, out_grad, node):
        return stack(list(out_grad), self.axis)


def split(a, axis):
    return Split(axis)(a)


# ---------------------------------------------------------------------------
# Image ops (for Conv, HW4)
# ---------------------------------------------------------------------------
class Flip(TensorOp):
    """Flip a tensor along the specified axes."""

    def __init__(self, axes):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    """
    Dilate (upsample with zeros) a tensor along the given axes by dilation factor.
    Used for the backward pass of strided convolution.
    """

    def __init__(self, axes, dilation):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for ax in self.axes:
            new_shape[ax] = a.shape[ax] * (self.dilation + 1)
        out = array_api.zeros(new_shape, dtype=a.dtype)
        slices = tuple(
            slice(None, None, self.dilation + 1) if i in self.axes else slice(None)
            for i in range(len(a.shape))
        )
        out[slices] = a
        return out

    def gradient(self, out_grad: Tensor, node: Tensor):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes, dilation):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = tuple(
            slice(None, None, self.dilation + 1) if i in self.axes else slice(None)
            for i in range(len(a.shape))
        )
        return a[slices]

    def gradient(self, out_grad: Tensor, node: Tensor):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    """
    2D Convolution: (N, H, W, C_in) x (kH, kW, C_in, C_out) -> (N, H', W', C_out)
    where H' = H - kH + 1 (no padding, stride=1 by default).

    Implements im2col approach for clarity.
    """

    def __init__(self, stride: int = 1, padding: int = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # A: (N, H, W, C_in)  B: (kH, kW, C_in, C_out)
        if self.padding > 0:
            pad = [(0, 0), (self.padding, self.padding),
                   (self.padding, self.padding), (0, 0)]
            A = array_api.pad(A, pad)

        N, H, W, C_in = A.shape
        kH, kW, _, C_out = B.shape
        H_out = (H - kH) // self.stride + 1
        W_out = (W - kW) // self.stride + 1

        # im2col
        col = array_api.zeros((N, H_out, W_out, kH, kW, C_in), dtype=A.dtype)
        for i in range(kH):
            for j in range(kW):
                h_sl = slice(i, i + H_out * self.stride, self.stride)
                w_sl = slice(j, j + W_out * self.stride, self.stride)
                col[:, :, :, i, j, :] = A[:, h_sl, w_sl, :]

        col = col.reshape(N * H_out * W_out, kH * kW * C_in)
        out = col @ B.reshape(kH * kW * C_in, C_out)
        return out.reshape(N, H_out, W_out, C_out)

    def gradient(self, out_grad: Tensor, node: Tensor):
        A, B = node.inputs
        kH, kW, C_in, C_out = B.shape

        # Gradient w.r.t. B:
        #   out_grad: (N, H_out, W_out, C_out)
        #   Dilate if stride > 1, then convolve A_padded transposed with out_grad
        if self.stride > 1:
            out_grad_d = dilate(out_grad, (1, 2), self.stride - 1)
        else:
            out_grad_d = out_grad

        # grad_B via conv of A (padded) with out_grad_d
        # A_padded: (N, H+2p, W+2p, C_in) -> transpose to (C_in, H+2p, W+2p, N)
        # We use the fact that grad_B = conv(A_padded^T, out_grad_d^T)
        # Implementation: use flip + pad approach
        A_pad = _pad_nhwc(A, self.padding)
        # Transpose A: (N,H,W,Ci) -> (H,W,Ci,N) — treat N as out-channels
        A_t = Transpose((0, 3))(A_pad)  # (Ci, H_p, W_p, N)
        A_t = Transpose((0, 1))(A_t)   # (H_p, Ci, W_p, N)
        A_t = Transpose((1, 2))(A_t)   # (H_p, W_p, Ci, N)
        # Correct permutation: (H_p, W_p, Ci, N)

        # out_grad_d: (N, H_out, W_out, C_out) -> (H_out, W_out, N, C_out)
        og_t = Transpose((0, 1))(out_grad_d)
        og_t = Transpose((1, 2))(og_t)

        grad_B = Conv(stride=1, padding=0)(A_t, og_t)
        # grad_B shape: (kH, kW, Ci, C_out)  ✓

        # Gradient w.r.t. A:
        #   out_grad convolved with flipped B
        B_flip = flip(B, axes=(0, 1))
        # B: (kH, kW, Ci, Co) -> transpose to (kH, kW, Co, Ci)
        B_t = Transpose((2, 3))(B_flip)
        full_pad = kH - 1 + self.padding
        grad_A = Conv(stride=1, padding=full_pad)(out_grad_d, B_t)

        return grad_A, grad_B


def _pad_nhwc(A: Tensor, padding: int) -> Tensor:
    """Zero-pad a (N,H,W,C) tensor by `padding` on spatial dims."""
    if padding == 0:
        return A
    from minigrad.autograd import Tensor as T
    import numpy as np
    data = A.realize_cached_data()
    padded = np.pad(data, [(0, 0), (padding, padding), (padding, padding), (0, 0)])
    return T(padded, requires_grad=False)


def conv(a, b, stride=1, padding=0):
    return Conv(stride=stride, padding=padding)(a, b)