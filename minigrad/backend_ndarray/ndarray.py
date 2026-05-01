"""
minigrad/backend_ndarray/ndarray.py
====================================
Python wrapper for the NDArray library.

All structural operations (reshape, permute, broadcast, slicing, flip)
are handled here in pure Python by manipulating shape/strides/offset —
NO memory copying.

Only .compact() triggers a C++ copy into contiguous memory.

Based on CMU 10-714 HW3.
"""
from __future__ import annotations
import numpy as np
import operator
import math
from functools import reduce
from typing import Optional, Tuple

# Try to import the compiled C++ backend; fall back to NumPy stub
try:
    from . import ndarray_backend_cpu as _backend_cpu
    _CPU_AVAILABLE = True
except ImportError:
    _CPU_AVAILABLE = False

try:
    from . import ndarray_backend_cuda as _backend_cuda
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False

TILE = 4  # Tiling factor for matmul


# ---------------------------------------------------------------------------
# Backend device wrappers
# ---------------------------------------------------------------------------
class BackendDevice:
    def __init__(self, name: str, module=None):
        self.name = name
        self.mod = module

    def __repr__(self):
        return f"minigrad.{self.name}()"

    def __eq__(self, other):
        return isinstance(other, BackendDevice) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def Array(self, *args, **kwargs):
        return self.mod.Array(*args, **kwargs)

    def enabled(self):
        return self.mod is not None


def _numpy_backend():
    """Fallback NumPy-based backend for testing without compiled extensions."""
    import types
    m = types.SimpleNamespace()

    class Array:
        def __init__(self, size):
            self._data = np.zeros(size, dtype=np.float32)

        def numpy(self):
            return self._data

        def fill(self, val):
            self._data[:] = val

        def __len__(self):
            return len(self._data)

    m.Array = Array

    def compact(a, out, shape, strides, offset):
        cnt = 0
        total = reduce(operator.mul, shape, 1)
        idx = [0] * len(shape)
        for _ in range(total):
            flat = offset + sum(i * s for i, s in zip(idx, strides))
            out._data[cnt] = a._data[flat]
            cnt += 1
            # increment multi-dim index
            for d in range(len(shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < shape[d]:
                    break
                idx[d] = 0

    m.Compact = compact

    def ewise_setitem(a, out, shape, strides, offset):
        cnt = 0
        total = reduce(operator.mul, shape, 1)
        idx = [0] * len(shape)
        for _ in range(total):
            flat = offset + sum(i * s for i, s in zip(idx, strides))
            out._data[flat] = a._data[cnt]
            cnt += 1
            for d in range(len(shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < shape[d]:
                    break
                idx[d] = 0

    m.EwiseSetitem = ewise_setitem

    def scalar_setitem(size, val, out, shape, strides, offset):
        total = reduce(operator.mul, shape, 1)
        idx = [0] * len(shape)
        for _ in range(total):
            flat = offset + sum(i * s for i, s in zip(idx, strides))
            out._data[flat] = val
            for d in range(len(shape) - 1, -1, -1):
                idx[d] += 1
                if idx[d] < shape[d]:
                    break
                idx[d] = 0

    m.ScalarSetitem = scalar_setitem

    for name, fn in [
        ("EwiseAdd", np.add), ("ScalarAdd", None),
        ("EwiseMul", np.multiply), ("ScalarMul", None),
        ("EwiseDiv", np.divide), ("ScalarDiv", None),
        ("ScalarPower", None),
        ("EwiseMaximum", np.maximum), ("ScalarMaximum", None),
        ("EwiseEq", np.equal), ("ScalarEq", None),
        ("EwiseGe", np.greater_equal), ("ScalarGe", None),
        ("EwiseLog", np.log),
        ("EwiseExp", np.exp),
        ("EwiseTanh", np.tanh),
        ("Matmul", None),
        ("ReduceMax", None), ("ReduceSum", None),
    ]:
        pass  # will be handled by numpy fallback in NDArray

    m._numpy_fallback = True
    return m


_np_backend = _numpy_backend()


def cpu():
    if _CPU_AVAILABLE:
        return BackendDevice("cpu", _backend_cpu)
    return BackendDevice("cpu", _np_backend)


def cuda():
    if _CUDA_AVAILABLE:
        return BackendDevice("cuda", _backend_cuda)
    raise RuntimeError("CUDA backend not available. Build with 'make cuda'.")


def numpy():
    return BackendDevice("numpy", _np_backend)
    

_DEFAULT_DEVICE = cpu()


# ---------------------------------------------------------------------------
# NDArray
# ---------------------------------------------------------------------------
class NDArray:
    """
    An N-dimensional array with a flat backing store.

    Fields:
      _handle:  the underlying memory (BackendDevice.Array)
      _shape:   tuple of dimension sizes
      _strides: tuple of strides (in elements)
      _offset:  offset into _handle where the array starts
      _device:  BackendDevice
    """

    def __init__(self, other, device=None):
        if isinstance(other, NDArray):
            if device is None or device == other._device:
                self._shape = other._shape
                self._strides = other._strides
                self._offset = other._offset
                self._device = other._device
                self._handle = other._handle
            else:
                # Copy across devices
                self._init_from_numpy(other.numpy(), device)
        elif isinstance(other, np.ndarray):
            device = device or _DEFAULT_DEVICE
            self._init_from_numpy(other, device)
        else:
            raise TypeError(f"Cannot construct NDArray from {type(other)}")

    def _init_from_numpy(self, array: np.ndarray, device: BackendDevice):
        self._shape = tuple(array.shape)
        self._strides = NDArray._compact_strides(self._shape)
        self._offset = 0
        self._device = device
        self._handle = device.Array(array.size)
        flat = array.astype(np.float32).flatten()
        np_handle = self._handle.numpy()
        np_handle[:] = flat


    @staticmethod
    def _compact_strides(shape):
        """Row-major (C-order) strides for a given shape."""
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """
        Create an NDArray with specified metadata.
        If handle is None, allocates new memory.
        """
        arr = NDArray.__new__(NDArray)
        arr._shape = tuple(shape)
        arr._strides = strides if strides is not None else NDArray._compact_strides(shape)
        arr._offset = offset
        arr._device = device or _DEFAULT_DEVICE
        if handle is None:
            arr._handle = arr._device.Array(reduce(operator.mul, shape, 1))
        else:
            arr._handle = handle
        return arr

    # ---- Properties ----
    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.float32

    @property
    def device(self):
        return self._device

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return reduce(operator.mul, self._shape, 1)

    def __repr__(self):
        return f"NDArray({self.numpy().__repr__()}, device={self._device})"

    # ---- Compact / numpy conversion ----
    def is_compact(self):
        return (
            self._strides == NDArray._compact_strides(self._shape)
            and self._offset == 0
        )

    def compact(self):
        if self.is_compact():
            return self
        out = NDArray.make(self._shape, device=self._device)
        self._device.mod.Compact(
            self._handle, out._handle,
            list(self._shape),
            list(self._strides),
            self._offset
        )
        return out

    def numpy(self):
        return self.compact()._handle.numpy().reshape(self._shape)
    


    def fill(self, value):
        self._handle.fill(value)

    # ---- Shape manipulation (no-copy) ----
    def reshape(self, new_shape):
        if reduce(operator.mul, new_shape, 1) != self.size:
            raise ValueError("Reshape: size mismatch")
        if not self.is_compact():
            return self.compact().reshape(new_shape)
        return NDArray.make(
            new_shape,
            strides=NDArray._compact_strides(new_shape),
            device=self._device,
            handle=self._handle,
            offset=self._offset,
        )

    def permute(self, new_axes):
        new_shape = tuple(self._shape[i] for i in new_axes)
        new_strides = tuple(self._strides[i] for i in new_axes)
        return NDArray.make(
            new_shape, strides=new_strides,
            device=self._device, handle=self._handle, offset=self._offset
        )

    def broadcast_to(self, new_shape):
        assert len(new_shape) >= len(self._shape)
        # Align shapes from the right
        old_shape = (1,) * (len(new_shape) - len(self._shape)) + self._shape
        old_strides = (0,) * (len(new_shape) - len(self._strides)) + self._strides
        new_strides = tuple(
            0 if os == 1 and ns != 1 else s
            for os, ns, s in zip(old_shape, new_shape, old_strides)
        )
        return NDArray.make(
            new_shape, strides=new_strides,
            device=self._device, handle=self._handle, offset=self._offset
        )

    def __getitem__(self, idxs):
        """
        Slice without copying.  Returns a view with adjusted offset + strides.
        All indexing is "keepdims" (never reduces ndim).
        """
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        # Expand ellipsis
        idxs = tuple(idxs)
        # Pad with slice(None) if fewer indices than dims
        while len(idxs) < self.ndim:
            idxs = idxs + (slice(None),)

        new_offset = self._offset
        new_shape = []
        new_strides = []

        for i, (idx, dim, stride) in enumerate(
            zip(idxs, self._shape, self._strides)
        ):
            if isinstance(idx, int):
                if idx < 0:
                    idx += dim
                new_offset += idx * stride
                new_shape.append(1)
                new_strides.append(stride)
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(dim)
                n = max(0, math.ceil((stop - start) / step))
                new_offset += start * stride
                new_shape.append(n)
                new_strides.append(stride * step)
            else:
                raise IndexError(f"Unsupported index type: {type(idx)}")

        return NDArray.make(
            tuple(new_shape), strides=tuple(new_strides),
            device=self._device, handle=self._handle, offset=new_offset
        )

    def __setitem__(self, idxs, other):
        view = self[idxs]
        if isinstance(other, NDArray):
            other = other.compact()
            self._device.mod.EwiseSetitem(
                other._handle, view._handle,
                view._shape, view._strides, view._offset
            )
        else:
            self._device.mod.ScalarSetitem(
                view.size, float(other), view._handle,
                view._shape, view._strides, view._offset
            )

    # ---- Flip (for Conv backward) ----
    def flip(self, axes):
        """Flip along given axes using negative strides."""
        new_strides = list(self._strides)
        new_offset = self._offset
        for ax in axes:
            new_offset += (self._shape[ax] - 1) * self._strides[ax]
            new_strides[ax] = -self._strides[ax]
        return NDArray.make(
            self._shape, strides=tuple(new_strides),
            device=self._device, handle=self._handle, offset=new_offset
        ).compact()

    # ---- Pad (for Conv) ----
    def pad(self, axes_pad):
        """
        Zero-pad.  axes_pad is a sequence of (before, after) per dimension.
        """
        new_shape = tuple(
            s + p[0] + p[1] for s, p in zip(self._shape, axes_pad)
        )
        out = NDArray.make(new_shape, device=self._device)
        out.fill(0.0)
        slices = tuple(
            slice(p[0], p[0] + s)
            for s, p in zip(self._shape, axes_pad)
        )
        out[slices] = self.compact()
        return out

    # ---- Element-wise ops ----
    def _ewise_or_scalar(self, other, ewise_fn, scalar_fn):
        out = NDArray.make(self._shape, device=self._device)
        a = self.compact()
        if isinstance(other, NDArray):
            b = other.compact()
            ewise_fn(a._handle, b._handle, out._handle)
        else:
            scalar_fn(a._handle, float(other), out._handle)
        return out

    def __add__(self, other):
        return self._ewise_or_scalar(
            other,
            lambda a, b, o: self._device.mod.EwiseAdd(a, b, o),
            lambda a, s, o: self._device.mod.ScalarAdd(a, s, o),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * (-1)

    def __mul__(self, other):
        return self._ewise_or_scalar(
            other,
            lambda a, b, o: self._device.mod.EwiseMul(a, b, o),
            lambda a, s, o: self._device.mod.ScalarMul(a, s, o),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._ewise_or_scalar(
            other,
            lambda a, b, o: self._device.mod.EwiseDiv(a, b, o),
            lambda a, s, o: self._device.mod.ScalarDiv(a, s, o),
        )

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __pow__(self, scalar):
        out = NDArray.make(self._shape, device=self._device)
        self._device.mod.ScalarPower(self.compact()._handle, float(scalar), out._handle)
        return out

    def maximum(self, other):
        return self._ewise_or_scalar(
            other,
            lambda a, b, o: self._device.mod.EwiseMaximum(a, b, o),
            lambda a, s, o: self._device.mod.ScalarMaximum(a, s, o),
        )

    def __eq__(self, other):
        return self._ewise_or_scalar(
            other,
            lambda a, b, o: self._device.mod.EwiseEq(a, b, o),
            lambda a, s, o: self._device.mod.ScalarEq(a, s, o),
        )

    def __ge__(self, other):
        return self._ewise_or_scalar(
            other,
            lambda a, b, o: self._device.mod.EwiseGe(a, b, o),
            lambda a, s, o: self._device.mod.ScalarGe(a, s, o),
        )

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __le__(self, other):
        return (other >= self)

    def __lt__(self, other):
        return (other > self)

    def log(self):
        out = NDArray.make(self._shape, device=self._device)
        self._device.mod.EwiseLog(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self._shape, device=self._device)
        self._device.mod.EwiseExp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self._shape, device=self._device)
        self._device.mod.EwiseTanh(self.compact()._handle, out._handle)
        return out

    # ---- Reductions ----
    def sum(self, axis=None, keepdims=False):
        return self._reduce(self._device.mod.ReduceSum, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return self._reduce(self._device.mod.ReduceMax, axis, keepdims)

    def _reduce(self, reduce_fn, axis, keepdims):
        if axis is None:
            flat = self.compact().reshape((self.size,))
            out = NDArray.make((1,), device=self._device)
            reduce_fn(flat._handle, out._handle, self.size)
            if keepdims:
                return out.reshape(tuple(1 for _ in self._shape))
            return out.reshape(())
        ax = axis if axis >= 0 else axis + self.ndim
        # Move reduce axis to the end, then compact
        perm = list(range(self.ndim))
        perm.append(perm.pop(ax))
        arr = self.permute(perm).compact()
        reduce_size = self._shape[ax]
        new_shape = tuple(s for i, s in enumerate(self._shape) if i != ax)
        out_size = reduce(operator.mul, new_shape, 1)
        out = NDArray.make((out_size,), device=self._device)
        reduce_fn(arr._handle, out._handle, reduce_size)
        out = out.reshape(new_shape)
        if keepdims:
            out = out.reshape(new_shape[:ax] + (1,) + new_shape[ax:])
        return out

    # ---- Matrix multiplication ----
    def __matmul__(self, other):
        assert self.ndim == 2 and other.ndim == 2
        m, k = self._shape
        k2, n = other._shape
        assert k == k2
        a = self.compact()
        b = other.compact()
        # Use tiled matmul if dimensions divisible by TILE
        if m % TILE == 0 and k % TILE == 0 and n % TILE == 0:
            a_t = a.reshape((m // TILE, TILE, k // TILE, TILE)).permute((0, 2, 1, 3)).compact()
            b_t = b.reshape((k // TILE, TILE, n // TILE, TILE)).permute((0, 2, 1, 3)).compact()
            out_t = NDArray.make((m // TILE, n // TILE, TILE, TILE), device=self._device)
            self._device.mod.MatmulTiled(a_t._handle, b_t._handle, out_t._handle, m, n, k)
            return out_t.permute((0, 2, 1, 3)).compact().reshape((m, n))
        else:
            out = NDArray.make((m, n), device=self._device)
            self._device.mod.Matmul(a._handle, b._handle, out._handle, m, n, k)
            return out