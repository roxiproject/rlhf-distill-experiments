"""Mandatory finite-difference gradient checks for the distillation loss
(KL + CE) and its components, w.r.t. student logits.
"""
import numpy as np
import pytest

from rlhf_distill.losses import cross_entropy, distillation_loss, kl_distillation


def _numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0, 4.0])
def test_gradcheck_kl_distillation_wrt_student_logits(temperature):
    rng = np.random.default_rng(0)
    student = rng.standard_normal((4, 3))
    teacher = rng.standard_normal((4, 3))

    def f(s):
        loss, _ = kl_distillation(s, teacher, temperature)
        return loss

    analytic = kl_distillation(student, teacher, temperature)[1]
    numeric = _numerical_grad(f, student.copy())
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("temperature,alpha", [(1.0, 0.5), (2.0, 0.3), (4.0, 0.7), (0.5, 0.1)])
def test_gradcheck_distillation_loss_wrt_student_logits(temperature, alpha):
    rng = np.random.default_rng(1)
    student = rng.standard_normal((5, 4))
    teacher = rng.standard_normal((5, 4))
    y = rng.integers(0, 4, size=5)

    def f(s):
        loss, _ = distillation_loss(s, teacher, y, temperature=temperature, alpha=alpha)
        return loss

    analytic = distillation_loss(student, teacher, y, temperature=temperature, alpha=alpha)[1]
    numeric = _numerical_grad(f, student.copy())
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_gradcheck_cross_entropy_wrt_logits():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((6, 3))
    y = rng.integers(0, 3, size=6)

    def f(z):
        loss, _ = cross_entropy(z, y)
        return loss

    analytic = cross_entropy(logits, y)[1]
    numeric = _numerical_grad(f, logits.copy())
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_gradcheck_distillation_loss_single_sample():
    rng = np.random.default_rng(3)
    student = rng.standard_normal((1, 3))
    teacher = rng.standard_normal((1, 3))
    y = rng.integers(0, 3, size=1)

    def f(s):
        loss, _ = distillation_loss(s, teacher, y, temperature=2.0, alpha=0.4)
        return loss

    analytic = distillation_loss(student, teacher, y, temperature=2.0, alpha=0.4)[1]
    numeric = _numerical_grad(f, student.copy())
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_gradcheck_distillation_loss_larger_batch():
    rng = np.random.default_rng(4)
    student = rng.standard_normal((16, 5))
    teacher = rng.standard_normal((16, 5))
    y = rng.integers(0, 5, size=16)

    def f(s):
        loss, _ = distillation_loss(s, teacher, y, temperature=3.0, alpha=0.6)
        return loss

    analytic = distillation_loss(student, teacher, y, temperature=3.0, alpha=0.6)[1]
    numeric = _numerical_grad(f, student.copy())
    np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)
