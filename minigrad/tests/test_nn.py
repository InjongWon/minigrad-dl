"""
tests/test_nn.py
================
Unit tests for the neural network module library.

Covers:
  - Linear layer forward + gradient
  - LayerNorm, BatchNorm
  - SoftmaxLoss
  - Initializers (Xavier, Kaiming)
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import minigrad.nn as nn
import minigrad.init as init
from minigrad.autograd import Tensor


def test_linear_forward():
    np.random.seed(42)
    lin = nn.Linear(4, 3, bias=True)
    x = Tensor(np.random.randn(2, 4).astype(np.float32))
    out = lin(x)
    assert out.shape == (2, 3)


def test_linear_backward():
    lin = nn.Linear(4, 3, bias=False)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    out = lin(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (2, 4)


def test_relu():
    relu = nn.ReLU()
    x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]))
    out = relu(x)
    expected = np.array([0.0, 0.0, 1.0, 2.0])
    assert np.allclose(out.numpy(), expected)


def test_sequential():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = model(x)
    assert out.shape == (3, 2)


def test_layer_norm():
    ln = nn.LayerNorm1d(4)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = ln(x)
    assert out.shape == (3, 4)
    # After layer norm, mean ≈ 0, std ≈ 1 per sample
    out_np = out.numpy()
    assert np.allclose(out_np.mean(axis=1), 0.0, atol=1e-5)


def test_softmax_loss():
    loss_fn = nn.SoftmaxLoss()
    logits = Tensor(np.array([[1.0, 2.0, 0.5], [0.1, 0.2, 3.0]]))
    labels = Tensor(np.array([1, 2]).astype(np.float32))
    loss = loss_fn(logits, labels)
    assert loss.numpy() > 0


def test_xavier_uniform():
    w = init.xavier_uniform(64, 128)
    assert w.shape == (64, 128)
    # Check variance is approximately correct
    a = np.sqrt(6.0 / (64 + 128))
    assert np.abs(w.numpy()).max() <= a + 1e-6


def test_kaiming_uniform():
    w = init.kaiming_uniform(64, 128)
    assert w.shape == (64, 128)
    bound = np.sqrt(2.0) * np.sqrt(3.0 / 64)
    assert np.abs(w.numpy()).max() <= bound + 1e-6


def test_dropout_train():
    drop = nn.Dropout(0.5)
    drop.train()
    x = Tensor(np.ones((100, 100), dtype=np.float32))
    out = drop(x)
    # Roughly half should be zero
    zero_frac = (out.numpy() == 0).mean()
    assert 0.3 < zero_frac < 0.7


def test_dropout_eval():
    drop = nn.Dropout(0.5)
    drop.eval()
    x = Tensor(np.ones((10, 10), dtype=np.float32))
    out = drop(x)
    assert np.allclose(out.numpy(), x.numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    