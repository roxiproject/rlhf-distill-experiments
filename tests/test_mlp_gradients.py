"""Gradient checks for the MLP backward pass against finite differences.

These are the mandatory correctness checks for the hand-derived backward
pass — every test in this file must pass.
"""
import numpy as np
import pytest

from rlhf_distill.losses import cross_entropy
from rlhf_distill.mlp import MLP


def _numerical_param_grad(model, x, y, param_idx, eps=1e-5):
    params = model.get_params()
    original = params[param_idx].copy()
    grad = np.zeros_like(original)
    it = np.nditer(original, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig_val = original[idx]

        params[param_idx][idx] = orig_val + eps
        model.set_params(params)
        loss_plus, _ = cross_entropy(model.forward(x, cache=False), y)

        params[param_idx][idx] = orig_val - eps
        model.set_params(params)
        loss_minus, _ = cross_entropy(model.forward(x, cache=False), y)

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        params[param_idx][idx] = orig_val
        it.iternext()
    model.set_params(params)
    return grad


def _analytic_grads(model, x, y):
    logits = model.forward(x)
    _, dlogits = cross_entropy(logits, y)
    return model.backward(dlogits)


@pytest.mark.parametrize("layer_sizes", [[3, 4, 2], [2, 5, 3, 2], [4, 6, 6, 3]])
def test_gradcheck_all_params(layer_sizes):
    rng = np.random.default_rng(0)
    model = MLP(layer_sizes, seed=1)
    n = 6
    x = rng.standard_normal((n, layer_sizes[0]))
    y = rng.integers(0, layer_sizes[-1], size=n)

    analytic = _analytic_grads(model, x, y)
    for idx in range(len(analytic)):
        numeric = _numerical_param_grad(model, x, y, idx)
        np.testing.assert_allclose(analytic[idx], numeric, atol=1e-4, rtol=1e-3)


def test_gradcheck_single_hidden_weight_matrix():
    rng = np.random.default_rng(2)
    model = MLP([3, 4, 2], seed=3)
    x = rng.standard_normal((5, 3))
    y = rng.integers(0, 2, size=5)
    analytic = _analytic_grads(model, x, y)
    numeric = _numerical_param_grad(model, x, y, 0)  # W0
    np.testing.assert_allclose(analytic[0], numeric, atol=1e-4, rtol=1e-3)


def test_gradcheck_bias_vectors():
    rng = np.random.default_rng(4)
    model = MLP([3, 4, 2], seed=5)
    x = rng.standard_normal((5, 3))
    y = rng.integers(0, 2, size=5)
    analytic = _analytic_grads(model, x, y)
    for bias_idx in [1, 3]:
        numeric = _numerical_param_grad(model, x, y, bias_idx)
        np.testing.assert_allclose(analytic[bias_idx], numeric, atol=1e-4, rtol=1e-3)


def test_gradcheck_output_layer_weight_matrix():
    rng = np.random.default_rng(6)
    model = MLP([3, 5, 4, 2], seed=7)
    x = rng.standard_normal((4, 3))
    y = rng.integers(0, 2, size=4)
    analytic = _analytic_grads(model, x, y)
    numeric = _numerical_param_grad(model, x, y, 4)  # W2 (final layer)
    np.testing.assert_allclose(analytic[4], numeric, atol=1e-4, rtol=1e-3)


def test_gradcheck_single_example():
    rng = np.random.default_rng(8)
    model = MLP([2, 3, 2], seed=9)
    x = rng.standard_normal((1, 2))
    y = rng.integers(0, 2, size=1)
    analytic = _analytic_grads(model, x, y)
    for idx in range(len(analytic)):
        numeric = _numerical_param_grad(model, x, y, idx)
        np.testing.assert_allclose(analytic[idx], numeric, atol=1e-4, rtol=1e-3)
