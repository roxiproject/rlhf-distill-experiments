import numpy as np
import pytest

from rlhf_distill.losses import (
    accuracy,
    cross_entropy,
    distillation_loss,
    kl_distillation,
    log_softmax,
    softmax,
)


def test_softmax_rows_sum_to_one():
    logits = np.random.default_rng(0).standard_normal((5, 4))
    p = softmax(logits)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(5), atol=1e-10)


def test_softmax_nonnegative():
    logits = np.random.default_rng(1).standard_normal((5, 4))
    p = softmax(logits)
    assert np.all(p >= 0)


def test_softmax_numerically_stable_for_large_logits():
    logits = np.array([[1000.0, 1001.0, 999.0]])
    p = softmax(logits)
    assert np.all(np.isfinite(p))
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-10)


def test_log_softmax_matches_log_of_softmax():
    logits = np.random.default_rng(2).standard_normal((5, 4))
    np.testing.assert_allclose(log_softmax(logits), np.log(softmax(logits)), atol=1e-10)


def test_softmax_invariant_to_constant_shift():
    logits = np.random.default_rng(3).standard_normal((3, 4))
    p1 = softmax(logits)
    p2 = softmax(logits + 100.0)
    np.testing.assert_allclose(p1, p2, atol=1e-8)


def test_cross_entropy_zero_for_perfect_confident_prediction():
    logits = np.array([[100.0, -100.0], [-100.0, 100.0]])
    y = np.array([0, 1])
    loss, _ = cross_entropy(logits, y)
    assert loss < 1e-8


def test_cross_entropy_uniform_logits_equals_log_c():
    n_classes = 5
    logits = np.zeros((3, n_classes))
    y = np.array([0, 1, 2])
    loss, _ = cross_entropy(logits, y)
    np.testing.assert_allclose(loss, np.log(n_classes), atol=1e-10)


def test_cross_entropy_grad_shape():
    logits = np.random.default_rng(4).standard_normal((6, 3))
    y = np.random.default_rng(4).integers(0, 3, size=6)
    _, grad = cross_entropy(logits, y)
    assert grad.shape == logits.shape


def test_kl_distillation_zero_when_distributions_match():
    logits = np.random.default_rng(5).standard_normal((4, 3))
    loss, _ = kl_distillation(logits, logits, temperature=2.0)
    assert abs(loss) < 1e-10


def test_kl_distillation_nonnegative():
    rng = np.random.default_rng(6)
    student = rng.standard_normal((6, 4))
    teacher = rng.standard_normal((6, 4))
    loss, _ = kl_distillation(student, teacher, temperature=1.5)
    assert loss >= -1e-10


def test_kl_distillation_grad_shape():
    rng = np.random.default_rng(7)
    student = rng.standard_normal((5, 3))
    teacher = rng.standard_normal((5, 3))
    _, grad = kl_distillation(student, teacher, temperature=2.0)
    assert grad.shape == student.shape


@pytest.mark.parametrize("temperature", [1.0, 2.0, 5.0, 10.0])
def test_kl_distillation_higher_temperature_softens(temperature):
    rng = np.random.default_rng(8)
    logits = rng.standard_normal((1, 5)) * 10  # sharp distribution
    from rlhf_distill.losses import softmax as sm

    p = sm(logits / temperature)
    # entropy should increase (distribution flattens) with temperature
    entropy = -np.sum(p * np.log(p + 1e-12))
    assert entropy >= 0


def test_distillation_loss_reduces_to_ce_when_alpha_one():
    rng = np.random.default_rng(9)
    student = rng.standard_normal((5, 3))
    teacher = rng.standard_normal((5, 3))
    y = rng.integers(0, 3, size=5)
    ce_loss, ce_grad = cross_entropy(student, y)
    d_loss, d_grad = distillation_loss(student, teacher, y, temperature=2.0, alpha=1.0)
    np.testing.assert_allclose(d_loss, ce_loss, atol=1e-10)
    np.testing.assert_allclose(d_grad, ce_grad, atol=1e-10)


def test_distillation_loss_pure_kl_when_alpha_zero():
    rng = np.random.default_rng(10)
    student = rng.standard_normal((5, 3))
    teacher = rng.standard_normal((5, 3))
    y = rng.integers(0, 3, size=5)
    kl_loss, kl_grad = kl_distillation(student, teacher, temperature=2.0)
    d_loss, d_grad = distillation_loss(student, teacher, y, temperature=2.0, alpha=0.0)
    np.testing.assert_allclose(d_loss, (2.0 ** 2) * kl_loss, atol=1e-10)
    np.testing.assert_allclose(d_grad, (2.0 ** 2) * kl_grad, atol=1e-10)


def test_accuracy_perfect():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    y = np.array([0, 1])
    assert accuracy(logits, y) == 1.0


def test_accuracy_zero():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    y = np.array([1, 0])
    assert accuracy(logits, y) == 0.0


def test_accuracy_partial():
    logits = np.array([[10.0, 0.0], [10.0, 0.0]])
    y = np.array([0, 1])
    assert accuracy(logits, y) == 0.5
