"""
tests/test_ops.py
=================
Unit tests for math operators (forward values).
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minigrad.autograd import Tensor
import minigrad.ops.ops_mathematic as ops
from minigrad.ops.ops_logarithmic import logsumexp


def test_add():
    a = Tensor(np.array([1.0, 2.0]))
    b = Tensor(np.array([3.0, 4.0]))
    assert np.allclose((a + b).numpy(), [4.0, 6.0])


def test_mul():
    a = Tensor(np.array([2.0, 3.0]))
    b = Tensor(np.array([4.0, 5.0]))
    assert np.allclose((a * b).numpy(), [8.0, 15.0])


def test_matmul():
    A = Tensor(np.eye(3, dtype=np.float32))
    B = Tensor(np.ones((3, 2), dtype=np.float32))
    out = A @ B
    assert out.shape == (3, 2)
    assert np.allclose(out.numpy(), np.ones((3, 2)))


def test_log_exp():
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(ops.log(ops.exp(x)).numpy(), x.numpy(), atol=1e-5)


def test_relu():
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    out = ops.relu(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.allclose(out.numpy(), expected)


def test_summation():
    x = Tensor(np.ones((3, 4), dtype=np.float32))
    assert ops.summation(x, axes=(0,)).numpy().shape == (4,)
    assert np.allclose(ops.summation(x, axes=(0,)).numpy(), 3.0)
    assert np.allclose(ops.summation(x, axes=(1,)).numpy(), 4.0)


def test_logsumexp():
    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    out = logsumexp(x, axes=(1,))
    expected = np.log(np.sum(np.exp(x.numpy()), axis=1))
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_tanh():
    x = Tensor(np.array([0.0, 1.0, -1.0]))
    out = ops.tanh(x)
    expected = np.tanh(x.numpy())
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_transpose():
    x = Tensor(np.random.randn(3, 4, 5).astype(np.float32))
    out = ops.transpose(x, axes=(0, 1))
    assert out.shape == (4, 3, 5)


def test_reshape():
    x = Tensor(np.arange(12).astype(np.float32))
    out = ops.reshape(x, (3, 4))
    assert out.shape == (3, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])