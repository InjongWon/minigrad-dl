"""
minigrad/optim/__init__.py
==========================
Gradient-based optimizers.

Implements:
  - SGD (with momentum and weight decay)
  - Adam (adaptive moment estimation)

Based on CMU 10-714 HW2.
"""
import numpy as np
from minigrad.autograd import Tensor


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        raise NotImplementedError

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and weight decay.

    Update rule (with momentum μ and weight decay λ):
        v_{t+1} = μ * v_t + (grad + λ * w_t)
        w_{t+1} = w_t - lr * v_{t+1}
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {id(p): np.zeros_like(p.numpy()) for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.numpy()
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.numpy()
            v = self.velocity[id(p)]
            v = self.momentum * v + grad
            self.velocity[id(p)] = v
            p.data = Tensor(p.numpy() - self.lr * v, requires_grad=False)


class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation.

    Update rule:
        m_t = β1 * m_{t-1} + (1 - β1) * grad
        v_t = β2 * v_{t-1} + (1 - β2) * grad²
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)
        w_t = w_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {id(p): np.zeros_like(p.numpy()) for p in self.params}
        self.v = {id(p): np.zeros_like(p.numpy()) for p in self.params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.numpy()
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.numpy()
            m = self.beta1 * self.m[id(p)] + (1 - self.beta1) * grad
            v = self.beta2 * self.v[id(p)] + (1 - self.beta2) * grad ** 2
            self.m[id(p)] = m
            self.v[id(p)] = v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data = Tensor(
                p.numpy() - self.lr * m_hat / (np.sqrt(v_hat) + self.eps),
                requires_grad=False,
            )