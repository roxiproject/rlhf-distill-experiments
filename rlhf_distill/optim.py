"""Adam optimizer implemented from scratch (no ML framework dependency)."""
from __future__ import annotations

import numpy as np


class Adam:
    """Standard Adam (Kingma & Ba, 2014) operating on a flat list of numpy arrays.

    Usage:
        opt = Adam(model.get_params(), lr=1e-3)
        ...
        grads = model.backward(dlogits)
        opt.step(model.get_params(), grads)
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """In-place update of `params` using `grads`. Both must be the same
        flat ordering used at construction time."""
        if len(params) != len(self.m):
            raise ValueError("params length does not match optimizer state")
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2 = 1.0 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
