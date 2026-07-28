"""A small multi-layer perceptron with a manually-derived backward pass.

No autodiff library is used anywhere in this module. Every gradient is
derived by hand and verified against finite differences in
tests/test_mlp_gradients.py.

Architecture: an arbitrary stack of Linear -> ReLU layers, ending in a
final Linear layer that produces raw logits (softmax/cross-entropy is
applied outside the model, in rlhf_distill.losses, so the same logits can
be reused for both hard-label CE and temperature-scaled distillation).
"""
from __future__ import annotations

import numpy as np


def _init_layer(fan_in: int, fan_out: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """He initialization, appropriate for ReLU networks."""
    scale = np.sqrt(2.0 / fan_in)
    w = rng.standard_normal((fan_in, fan_out)).astype(np.float64) * scale
    b = np.zeros(fan_out, dtype=np.float64)
    return w, b


class MLP:
    """A feed-forward network of Linear -> ReLU blocks ending in a Linear layer.

    Parameters
    ----------
    layer_sizes: sequence of ints, e.g. [in_dim, hidden1, hidden2, ..., out_dim]
    seed: seed for the parameter-initialization RNG.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 0):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least an input and output dimension")
        self.layer_sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            w, b = _init_layer(fan_in, fan_out, rng)
            self.weights.append(w)
            self.biases.append(b)
        self.n_layers = len(self.weights)
        # populated by forward(), consumed by backward()
        self._cache: dict | None = None

    def num_params(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def get_params(self) -> list[np.ndarray]:
        """Flat list alternating [W0, b0, W1, b1, ...] — the canonical param ordering."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w)
            params.append(b)
        return params

    def set_params(self, params: list[np.ndarray]) -> None:
        for i in range(self.n_layers):
            self.weights[i] = params[2 * i]
            self.biases[i] = params[2 * i + 1]

    def forward(self, x: np.ndarray, cache: bool = True) -> np.ndarray:
        """Forward pass. x has shape (N, in_dim). Returns logits of shape (N, out_dim)."""
        a = x
        pre_activations = []
        activations = [a]
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i < self.n_layers - 1:
                a = np.maximum(z, 0.0)
            else:
                a = z  # final layer: raw logits, no activation
            activations.append(a)
        if cache:
            self._cache = {"pre_activations": pre_activations, "activations": activations}
        return a

    def backward(self, dlogits: np.ndarray) -> list[np.ndarray]:
        """Backpropagate the gradient of the loss w.r.t. the output logits.

        dlogits has shape (N, out_dim) — dL/dlogits, already averaged/summed
        the way the caller wants (this function does not rescale by batch
        size; callers should pass in the correctly-normalized upstream
        gradient).

        Returns gradients in the same flat ordering as get_params():
        [dW0, db0, dW1, db1, ...].
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward(cache=True)")
        activations = self._cache["activations"]
        pre_activations = self._cache["pre_activations"]

        grads_w: list[np.ndarray | None] = [None] * self.n_layers
        grads_b: list[np.ndarray | None] = [None] * self.n_layers

        delta = dlogits  # gradient flowing into the current layer's pre-activation
        for i in reversed(range(self.n_layers)):
            a_prev = activations[i]  # input to layer i
            grads_w[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                d_a_prev = delta @ self.weights[i].T
                relu_mask = (pre_activations[i - 1] > 0).astype(np.float64)
                delta = d_a_prev * relu_mask

        flat = []
        for gw, gb in zip(grads_w, grads_b):
            flat.append(gw)
            flat.append(gb)
        return flat

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        from rlhf_distill.losses import softmax
        logits = self.forward(x, cache=False)
        return softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x, cache=False), axis=1)
