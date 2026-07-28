import numpy as np
import pytest

from rlhf_distill.data import generate_blobs, generate_moons
from rlhf_distill.losses import cross_entropy, distillation_loss, kl_distillation, softmax
from rlhf_distill.mlp import MLP
from rlhf_distill.optim import Adam


def test_softmax_single_class():
    logits = np.array([[5.0]])
    p = softmax(logits)
    np.testing.assert_allclose(p, [[1.0]])


def test_cross_entropy_single_example():
    logits = np.array([[1.0, 2.0, 3.0]])
    y = np.array([2])
    loss, grad = cross_entropy(logits, y)
    assert loss > 0
    assert grad.shape == (1, 3)


def test_kl_distillation_identical_but_scaled_temperature():
    logits = np.array([[1.0, 2.0, -1.0]])
    loss_t1, _ = kl_distillation(logits, logits, temperature=1.0)
    loss_t5, _ = kl_distillation(logits, logits, temperature=5.0)
    assert abs(loss_t1) < 1e-10
    assert abs(loss_t5) < 1e-10


def test_distillation_loss_extreme_alpha_bounds():
    rng = np.random.default_rng(0)
    student = rng.standard_normal((3, 2))
    teacher = rng.standard_normal((3, 2))
    y = rng.integers(0, 2, size=3)
    loss_a0, _ = distillation_loss(student, teacher, y, temperature=2.0, alpha=0.0)
    loss_a1, _ = distillation_loss(student, teacher, y, temperature=2.0, alpha=1.0)
    assert loss_a0 != loss_a1


def test_mlp_single_example_forward():
    m = MLP([3, 4, 2], seed=0)
    x = np.ones((1, 3))
    out = m.forward(x)
    assert out.shape == (1, 2)


def test_mlp_large_batch_forward():
    m = MLP([3, 4, 2], seed=0)
    x = np.random.default_rng(0).standard_normal((1000, 3))
    out = m.forward(x, cache=False)
    assert out.shape == (1000, 2)


def test_mlp_deep_network():
    layer_sizes = [4] + [8] * 10 + [3]
    m = MLP(layer_sizes, seed=0)
    x = np.random.default_rng(0).standard_normal((5, 4))
    out = m.forward(x, cache=False)
    assert out.shape == (5, 3)
    assert np.all(np.isfinite(out))


def test_adam_single_scalar_param():
    p = np.array([1.0])
    opt = Adam([p], lr=0.1)
    opt.step([p], [np.array([0.5])])
    assert p[0] != 1.0


def test_generate_blobs_two_classes():
    x, y = generate_blobs(n_samples=50, n_classes=2, seed=0)
    assert set(np.unique(y)) == {0, 1}


def test_generate_moons_small_sample():
    x, y = generate_moons(n_samples=4, seed=0)
    assert x.shape[0] == 4


def test_mlp_zero_input():
    m = MLP([3, 4, 2], seed=0)
    x = np.zeros((2, 3))
    out = m.forward(x, cache=False)
    np.testing.assert_allclose(out, np.broadcast_to(m.biases[-1], (2, 2)))


def test_cross_entropy_all_same_class():
    logits = np.random.default_rng(0).standard_normal((10, 3))
    y = np.zeros(10, dtype=np.int64)
    loss, grad = cross_entropy(logits, y)
    assert np.isfinite(loss)
    assert grad.shape == logits.shape


def test_kl_distillation_temperature_must_be_positive_produces_finite():
    rng = np.random.default_rng(0)
    student = rng.standard_normal((3, 2))
    teacher = rng.standard_normal((3, 2))
    loss, grad = kl_distillation(student, teacher, temperature=0.1)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


def test_mlp_num_params_deep_network():
    m = MLP([2, 3, 3, 3, 1], seed=0)
    expected = (2 * 3 + 3) + (3 * 3 + 3) + (3 * 3 + 3) + (3 * 1 + 1)
    assert m.num_params() == expected


def test_get_params_ordering_alternates_weights_biases():
    m = MLP([2, 3, 1], seed=0)
    params = m.get_params()
    assert params[0].shape == m.weights[0].shape
    assert params[1].shape == m.biases[0].shape
    assert params[2].shape == m.weights[1].shape
    assert params[3].shape == m.biases[1].shape
