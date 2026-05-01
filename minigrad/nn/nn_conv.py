"""
minigrad/nn/nn_conv.py
======================
2D Convolutional layer.

Based on CMU 10-714 HW4.
"""
import numpy as np
from minigrad.autograd import Tensor
import minigrad.init as init
from minigrad.nn.nn_basic import Module, Parameter, BatchNorm2d
from minigrad.ops.ops_mathematic import conv


class Conv(Module):
    """
    2D convolution.  Input/output layout: (N, C_in, H, W).
    Internally converts to (N, H, W, C) for the Conv op, then back.

    Weight init: Kaiming Uniform
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2  # same-ish padding

        # Weight: (kH, kW, C_in, C_out)
        fan_in = kernel_size * kernel_size * in_channels
        self.weight = Parameter(
            init.kaiming_uniform(fan_in, out_channels,
                                 device=device, dtype=dtype).reshape(
                (kernel_size, kernel_size, in_channels, out_channels)
            )
        )
        self.bias = None
        if bias:
            b = init.kaiming_uniform(out_channels, 1, device=device, dtype=dtype)
            self.bias = Parameter(b.reshape((out_channels,)))

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in, H, W)  ->  (N, H, W, C_in)
        from minigrad.ops.ops_mathematic import reshape, broadcast_to
        x_nhwc = x.transpose((0, 2, 3, 1))
        out = conv(x_nhwc, self.weight, stride=self.stride, padding=self.padding)
        # out: (N, H_out, W_out, C_out) -> (N, C_out, H_out, W_out)
        out = out.transpose((0, 3, 1, 2))
        if self.bias is not None:
            b = broadcast_to(
                reshape(self.bias, (1, self.out_channels, 1, 1)), out.shape
            )
            out = out + b
        return out