"""
tests/test_ndarray.py
=====================
Unit tests for the NDArray Python layer (shape ops, strides, broadcast).
These tests use only the Python layer (no C++ backend needed).
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minigrad.backend_ndarray.ndarray import NDArray
from functools import reduce
import operator


def np_to_nd(arr):
    return NDArray(arr)


def test_shape():
    a = np_to_nd(np.arange(12).reshape(3, 4).astype(np.float32))
    assert a.shape == (3, 4)


def test_strides_compact():
    a = np_to_nd(np.arange(24).reshape(2, 3, 4).astype(np.float32))
    assert a._strides == (12, 4, 1)


def test_reshape():
    arr = np.arange(12).astype(np.float32)
    a = np_to_nd(arr).reshape((3, 4))
    assert a.shape == (3, 4)
    assert np.allclose(a.numpy(), arr.reshape(3, 4))


def test_permute():
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    a = np_to_nd(arr).permute((2, 0, 1))
    assert a.shape == (4, 2, 3)
    assert np.allclose(a.numpy(), arr.transpose(2, 0, 1))


def test_broadcast_to():
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # (1, 3)
    a = np_to_nd(arr).broadcast_to((4, 3))
    assert a.shape == (4, 3)
    expected = np.broadcast_to(arr, (4, 3))
    assert np.allclose(a.numpy(), expected)


def test_getitem_slice():
    arr = np.arange(20).reshape(4, 5).astype(np.float32)
    a = np_to_nd(arr)[1:3, 2:4]
    assert a.shape == (2, 2)
    assert np.allclose(a.numpy(), arr[1:3, 2:4])


def test_getitem_int():
    arr = np.arange(12).reshape(3, 4).astype(np.float32)
    a = np_to_nd(arr)[1]
    # Our NDArray always keepdims → shape (1, 4)
    assert a.shape[1] == 4


def test_setitem():
    arr = np.zeros((4, 4), dtype=np.float32)
    a = np_to_nd(arr)
    a[1:3, 1:3] = 5.0
    assert np.allclose(a.numpy()[1:3, 1:3], 5.0)


def test_numpy_roundtrip():
    arr = np.random.randn(5, 6).astype(np.float32)
    a = np_to_nd(arr)
    assert np.allclose(a.numpy(), arr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])