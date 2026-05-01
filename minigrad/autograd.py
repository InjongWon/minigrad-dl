"""
minigrad/autograd.py
====================
Core reverse-mode automatic differentiation engine.

Demonstrates:
  - Computational graph construction via operator overloading
  - Topological sort (post-order DFS)
  - Reverse-mode AD: accumulate adjoints in reverse topological order
  - Higher-order differentiation (gradient graph is itself differentiable)

Based on CMU 10-714 HW1.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Back-end array abstraction
# ---------------------------------------------------------------------------
LAZY_MODE = False
TENSOR_COUNTER = 0


class Device:
    """Abstract base for compute devices (CPU / CUDA)."""
    pass


class CPUDevice(Device):
    def __repr__(self):
        return "minigrad.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True


def cpu():
    """Return the default CPU device."""
    return CPUDevice()


def all_devices():
    """All supported devices."""
    return [cpu()]


# ---------------------------------------------------------------------------
# Op base class
# ---------------------------------------------------------------------------
class Op:
    """
    An operator in the computation graph.

    Subclasses implement:
      - compute(*args):  forward pass on raw NDArray / numpy data
      - gradient(out_grad, node):  VJP — returns gradient for each input
    """

    def __call__(self, *args):
        raise NotImplementedError

    def compute(self, *args):
        raise NotImplementedError

    def gradient(self, out_grad: "Value", node: "Value"):
        raise NotImplementedError

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


# ---------------------------------------------------------------------------
# Value / Tensor
# ---------------------------------------------------------------------------
class Value:
    """
    A node in the computation graph.  Stores:
      - op:     the Op that produced this value (None for leaves)
      - inputs: the input Values to that Op
      - cached_data: the actual array (lazily evaluated)
      - grad:   accumulated gradient (set during backward)
    """
    op: Optional[Op]
    inputs: List["Value"]
    cached_data: object
    requires_grad: bool

    def realize_cached_data(self):
        """Trigger forward computation if not yet evaluated."""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None,
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(None, [], cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)
        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """
    Return nodes in a topological order (post-order DFS) suitable for
    forward evaluation.  Reverse this list for backward AD.
    """
    visited = set()
    topo_order = []

    def topo_sort_dfs(node: Value):
        if id(node) in visited:
            return
        visited.add(id(node))
        for inp in node.inputs:
            topo_sort_dfs(inp)
        topo_order.append(node)

    for node in node_list:
        topo_sort_dfs(node)
    return topo_order


def compute_gradient_of_variables(output_tensor: Value, out_grad: Value):
    """
    Reverse-mode AD: accumulate gradients for all nodes in the graph.

    For each node v (in reverse topological order):
        grad[v] = sum of out_grad contributions from all paths to output

    Stores the result in node.grad for every node in the graph.
    """
    # node_to_output_grads_list[v] = list of adjoint contributions to v
    node_to_output_grads_list: dict = {}
    node_to_output_grads_list[output_tensor] = [out_grad]

    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        # Sum all adjoint contributions
        node.grad = sum_node_list(node_to_output_grads_list[node])

        if node.is_leaf():
            continue

        # Propagate gradients to inputs
        input_grads = node.op.gradient_as_tuple(node.grad, node)
        for inp, g in zip(node.inputs, input_grads):
            if inp not in node_to_output_grads_list:
                node_to_output_grads_list[inp] = []
            node_to_output_grads_list[inp].append(g)


def sum_node_list(node_list):
    """Sum a non-empty list of Value nodes."""
    from functools import reduce
    from operator import add
    return reduce(add, node_list)


# ---------------------------------------------------------------------------
# Tensor (the user-facing type)
# ---------------------------------------------------------------------------
class Tensor(Value):
    """
    A multi-dimensional array living in the computation graph.

    Supports:
      - Operator overloading (+, -, *, /, @, **) that builds the graph
      - .backward() to trigger reverse-mode AD
      - .grad to read accumulated gradient
      - .numpy() to convert to numpy
    """
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs,
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(None, [], cached_data=cached_data, requires_grad=requires_grad)

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        # Accept None, CPUDevice, BackendDevice(name='cpu'), or string 'cpu'
        if device is None:
            return np.array(numpy_array, dtype=dtype)
        if isinstance(device, CPUDevice):
            return np.array(numpy_array, dtype=dtype)
        # BackendDevice or anything with a .name attribute
        name = getattr(device, 'name', None) or str(device)
        if 'cpu' in name.lower() or 'numpy' in name.lower():
            return np.array(numpy_array, dtype=dtype)
        raise NotImplementedError(f"Device {device} not supported; use CPUDevice or import cuda backend")

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (value.dtype, self.dtype)
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Return a new Tensor with the same data but no gradient tracking."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        if isinstance(data, np.ndarray):
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else Tensor(np.ones(self.shape, dtype=self.dtype), device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "minigrad.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if isinstance(data, np.ndarray):
            return data
        if np.isscalar(data):
            return np.array(data)
        return data.numpy()

    # ---- Operator overloads ----
    def __add__(self, other):
        if isinstance(other, Tensor):
            return minigrad_ops.EWiseAdd()(self, other)
        else:
            return minigrad_ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return minigrad_ops.EWiseMul()(self, other)
        else:
            return minigrad_ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return minigrad_ops.EWisePow()(self, other)
        else:
            return minigrad_ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return minigrad_ops.EWiseAdd()(self, minigrad_ops.Negate()(other))
        else:
            return minigrad_ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return minigrad_ops.EWiseDiv()(self, other)
        else:
            return minigrad_ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return minigrad_ops.MatMul()(self, other)

    def matmul(self, other):
        return minigrad_ops.MatMul()(self, other)

    def sum(self, axes=None):
        return minigrad_ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return minigrad_ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return minigrad_ops.Reshape(shape)(self)

    def __neg__(self):
        return minigrad_ops.Negate()(self)

    def transpose(self, axes=None):
        return minigrad_ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = lambda self, other: minigrad_ops.AddScalar(other)(minigrad_ops.Negate()(self))
    __rmatmul__ = __matmul__



class TensorOp(Op):
    """Base Op for ops that return a single Tensor."""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


# Deferred import to avoid circular dependency
import minigrad.ops.ops_mathematic as minigrad_ops  # noqa: E402