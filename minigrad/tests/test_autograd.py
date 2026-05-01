"""
tests/test_autograd.py
=======================
Unit tests for the reverse-mode autodiff engine.

Covers:
  - Topological sort
  - Gradient computation (numerical vs analytical)
  - Second-order gradients
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minigrad.autograd import Tensor, find_topo_sort
import minigrad.ops.ops_mathematic as ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def numerical_gradient(f, x: np.ndarray, eps=1e-4):
    """Finite-difference gradient estimate. Uses float64 for precision."""
    x = x.astype(np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = float(np.array(f(x)).sum())
        x[idx] = old - eps
        fm = float(np.array(f(x)).sum())
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = old
        it.iternext()
    return grad


def check_gradient(f, x_np: np.ndarray, tol=1e-2):
    """Compare analytical vs numerical gradient for a function."""
    x = Tensor(x_np.astype(np.float32), requires_grad=True)
    y = f(x)
    loss = y.sum() if y.shape != () else y
    loss.backward()
    analytical = x.grad.numpy().astype(np.float64)

    def scalar_f(xv):
        return float(np.array(f(Tensor(xv.astype(np.float32))).numpy()).sum())

    numerical = numerical_gradient(scalar_f, x_np.copy())
    return np.allclose(analytical, numerical, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_topo_sort_simple():
    a = Tensor(np.array([1.0]))
    b = Tensor(np.array([2.0]))
    c = a + b
    order = find_topo_sort([c])
    assert order.index(a) < order.index(c)
    assert order.index(b) < order.index(c)


def test_topo_sort_chain():
    a = Tensor(np.array([1.0]))
    b = a + a
    c = b + b
    order = find_topo_sort([c])
    assert order.index(a) < order.index(b) < order.index(c)


def test_gradient_add():
    x = np.random.randn(3, 4).astype(np.float32)
    assert check_gradient(lambda t: t + t, x)


def test_gradient_mul():
    x = np.random.randn(3, 4).astype(np.float32)
    assert check_gradient(lambda t: t * t, x)


def test_gradient_matmul():
    A_np = np.random.randn(4, 5).astype(np.float32)
    B_np = np.random.randn(5, 3).astype(np.float32)
    B = Tensor(B_np)

    def f(a):
        return ops.matmul(a, B)

    a = Tensor(A_np.copy(), requires_grad=True)
    out = f(a)
    out.backward(Tensor(np.ones_like(out.numpy())))
    analytical = a.grad.numpy()

    # Numerical: perturb A and measure total output change
    def scalar_f(av):
        return float(f(Tensor(av)).numpy().sum())

    numerical = numerical_gradient(scalar_f, A_np.copy())
    assert np.allclose(analytical, numerical, atol=1e-2, rtol=1e-2)


def test_gradient_log():
    x = np.abs(np.random.randn(3, 4)).astype(np.float32) + 0.1
    assert check_gradient(lambda t: ops.log(t), x)


def test_gradient_exp():
    x = np.random.randn(3, 4).astype(np.float32) * 0.5
    assert check_gradient(lambda t: ops.exp(t), x)


def test_gradient_relu():
    x = np.random.randn(3, 4).astype(np.float32)
    assert check_gradient(lambda t: ops.relu(t), x)


def test_gradient_summation():
    x = np.random.randn(3, 4).astype(np.float32)
    assert check_gradient(lambda t: ops.summation(t, axes=(0,)), x)
    assert check_gradient(lambda t: ops.summation(t, axes=(1,)), x)


def test_gradient_broadcast_to():
    x = np.random.randn(1, 4).astype(np.float32)
    assert check_gradient(lambda t: ops.broadcast_to(t, (3, 4)), x)


def test_gradient_reshape():
    x = np.random.randn(12).astype(np.float32)
    assert check_gradient(lambda t: ops.reshape(t, (3, 4)), x)


def test_second_order_gradient():
    """Gradient of gradient (higher-order AD)."""
    x = Tensor(np.array([2.0]), requires_grad=True)
    y = x ** 3         # y = x^3
    y.backward()
    dy_dx = x.grad    # should be 3x^2 = 12 at x=2
    # dy_dx is itself a Tensor in the graph
    assert np.isclose(dy_dx.numpy(), 12.0, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])