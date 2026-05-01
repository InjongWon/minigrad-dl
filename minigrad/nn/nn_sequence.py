"""
minigrad/nn/nn_sequence.py
==========================
Recurrent sequence models.

Implements:
  - RNNCell / RNN
  - LSTMCell / LSTM

Based on CMU 10-714 HW4.
"""
import numpy as np
from minigrad.autograd import Tensor
import minigrad.init as init
from minigrad.nn.nn_basic import Module, Parameter, Tanh, ReLU
from minigrad.ops.ops_mathematic import (
    matmul, broadcast_to, reshape, summation, stack
)


class RNNCell(Module):
    """
    Single-step vanilla RNN cell.
    h_t = tanh(x_t W_ih + h_{t-1} W_hh + b)
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh",
                 device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = Tanh() if nonlinearity == "tanh" else ReLU()

        self.W_ih = Parameter(
            init.kaiming_uniform(input_size, hidden_size, device=device, dtype=dtype)
        )
        self.W_hh = Parameter(
            init.kaiming_uniform(hidden_size, hidden_size, device=device, dtype=dtype)
        )
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(
                init.zeros(hidden_size, device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.zeros(hidden_size, device=device, dtype=dtype)
            )

    def forward(self, x: Tensor, h: Tensor = None) -> Tensor:
        N = x.shape[0]
        if h is None:
            h = Tensor(np.zeros((N, self.hidden_size), dtype=np.float32))

        out = matmul(x, self.W_ih) + matmul(h, self.W_hh)
        if self.bias_ih is not None:
            out = out + broadcast_to(reshape(self.bias_ih, (1, self.hidden_size)), (N, self.hidden_size))
        if self.bias_hh is not None:
            out = out + broadcast_to(reshape(self.bias_hh, (1, self.hidden_size)), (N, self.hidden_size))
        return self.nonlinearity(out)


class RNN(Module):
    """
    Multi-layer / multi-step vanilla RNN.
    Returns all hidden states and final hidden state.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 nonlinearity="tanh", device=None, dtype="float32"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = [
            RNNCell(
                input_size if l == 0 else hidden_size,
                hidden_size, bias=bias, nonlinearity=nonlinearity,
                device=device, dtype=dtype
            )
            for l in range(num_layers)
        ]

    def forward(self, x: Tensor, h0: Tensor = None):
        """
        x: (seq_len, batch, input_size)
        h0: (num_layers, batch, hidden_size)  [optional]

        Returns:
          output: (seq_len, batch, hidden_size)
          h_n:    (num_layers, batch, hidden_size)
        """
        T, N, _ = x.shape
        if h0 is None:
            h0 = Tensor(np.zeros((self.num_layers, N, self.hidden_size), dtype=np.float32))

        # Split h0 over layers
        h_list = [h0[l] for l in range(self.num_layers)]

        # Split input over time
        inputs = [x[t] for t in range(T)]

        outputs = []
        for t in range(T):
            inp = inputs[t]
            new_h = []
            for l, cell in enumerate(self.cells):
                h_new = cell(inp, h_list[l])
                new_h.append(h_new)
                inp = h_new
            h_list = new_h
            outputs.append(inp)

        output = stack(outputs, axis=0)            # (T, N, H)
        h_n = stack(h_list, axis=0)                # (num_layers, N, H)
        return output, h_n


class LSTMCell(Module):
    """
    Single-step LSTM cell.

    Gates:
      i = sigmoid(x W_ii + h W_hi + b_i)
      f = sigmoid(x W_if + h W_hf + b_f)
      g = tanh(x W_ig + h W_hg + b_g)
      o = sigmoid(x W_io + h W_ho + b_o)
      c_t = f * c_{t-1} + i * g
      h_t = o * tanh(c_t)
    """

    def __init__(self, input_size, hidden_size, bias=True,
                 device=None, dtype="float32"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # W_ih: (input_size, 4 * hidden_size) — all 4 gates concatenated
        self.W_ih = Parameter(
            init.kaiming_uniform(input_size, 4 * hidden_size, device=device, dtype=dtype)
        )
        self.W_hh = Parameter(
            init.kaiming_uniform(hidden_size, 4 * hidden_size, device=device, dtype=dtype)
        )
        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(
                init.zeros(4 * hidden_size, device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.zeros(4 * hidden_size, device=device, dtype=dtype)
            )

    def forward(self, x: Tensor, state=None):
        """
        x: (batch, input_size)
        state: ((batch, hidden_size), (batch, hidden_size)) or None

        Returns: (h_t, c_t)
        """
        N = x.shape[0]
        H = self.hidden_size
        if state is None:
            h = Tensor(np.zeros((N, H), dtype=np.float32))
            c = Tensor(np.zeros((N, H), dtype=np.float32))
        else:
            h, c = state

        gates = matmul(x, self.W_ih) + matmul(h, self.W_hh)
        if self.bias_ih is not None:
            gates = gates + broadcast_to(
                reshape(self.bias_ih, (1, 4 * H)), (N, 4 * H)
            )
        if self.bias_hh is not None:
            gates = gates + broadcast_to(
                reshape(self.bias_hh, (1, 4 * H)), (N, 4 * H)
            )

        # Split into 4 gates
        gi, gf, gg, go = [gates[:, j * H:(j + 1) * H] for j in range(4)]
        i = _sigmoid(gi)
        f = _sigmoid(gf)
        g = _tanh(gg)
        o = _sigmoid(go)

        c_new = f * c + i * g
        h_new = o * _tanh(c_new)
        return h_new, c_new


class LSTM(Module):
    """Multi-layer / multi-step LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 device=None, dtype="float32"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = [
            LSTMCell(
                input_size if l == 0 else hidden_size,
                hidden_size, bias=bias, device=device, dtype=dtype
            )
            for l in range(num_layers)
        ]

    def forward(self, x: Tensor, h0=None):
        """
        x: (seq_len, batch, input_size)
        h0: (h, c) each (num_layers, batch, hidden_size) or None

        Returns:
          output: (seq_len, batch, hidden_size)
          (h_n, c_n): each (num_layers, batch, hidden_size)
        """
        T, N, _ = x.shape
        H = self.hidden_size
        L = self.num_layers

        if h0 is None:
            zeros = np.zeros((L, N, H), dtype=np.float32)
            h_list = [Tensor(zeros[l]) for l in range(L)]
            c_list = [Tensor(zeros[l]) for l in range(L)]
        else:
            h_init, c_init = h0
            h_list = [h_init[l] for l in range(L)]
            c_list = [c_init[l] for l in range(L)]

        inputs = [x[t] for t in range(T)]
        outputs = []

        for t in range(T):
            inp = inputs[t]
            new_h, new_c = [], []
            for l, cell in enumerate(self.cells):
                h_new, c_new = cell(inp, (h_list[l], c_list[l]))
                new_h.append(h_new)
                new_c.append(c_new)
                inp = h_new
            h_list, c_list = new_h, new_c
            outputs.append(inp)

        output = stack(outputs, axis=0)
        h_n = stack(h_list, axis=0)
        c_n = stack(c_list, axis=0)
        return output, (h_n, c_n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sigmoid(x: Tensor) -> Tensor:
    """σ(x) = 1 / (1 + exp(-x))"""
    from minigrad.ops.ops_mathematic import exp
    from minigrad.autograd import Tensor as T
    import numpy as np
    ones = T(np.ones(x.shape, dtype=np.float32), requires_grad=False)
    return ones / (ones + exp(-x))


def _tanh(x: Tensor) -> Tensor:
    from minigrad.ops.ops_mathematic import tanh
    return tanh(x)