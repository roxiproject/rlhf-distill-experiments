"""Numerically stable softmax/log-softmax, cross-entropy, and the
temperature-scaled KL distillation loss, with hand-derived gradients.

Every gradient here is checked against finite differences in
tests/test_losses_gradcheck.py.
"""
from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax via the log-sum-exp trick (numerically stable)."""
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    return z - lse


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(logits))


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean hard-label cross-entropy loss and its gradient w.r.t. logits.

    logits: (N, C), y: (N,) integer class indices.
    Returns (loss, dloss/dlogits) where the gradient is already averaged
    over the batch (divided by N), matching the loss's own averaging.
    """
    n = logits.shape[0]
    log_p = log_softmax(logits)
    loss = -log_p[np.arange(n), y].mean()
    p = np.exp(log_p)
    onehot = np.zeros_like(logits)
    onehot[np.arange(n), y] = 1.0
    grad = (p - onehot) / n
    return float(loss), grad


def kl_distillation(
    student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float
) -> tuple[float, np.ndarray]:
    """Mean KL(teacher_soft || student_soft) at the given temperature, and
    its gradient w.r.t. student_logits.

    Both distributions are softened by dividing logits by `temperature`
    before softmax-ing (Hinton et al. 2015 style distillation). This
    function returns the *raw* (un-T^2-scaled) mean KL divergence; the
    caller (distillation_loss below) applies the conventional T^2 scaling
    used to keep gradient magnitudes comparable across temperatures.
    """
    n = student_logits.shape[0]
    log_p_teacher = log_softmax(teacher_logits / temperature)
    log_q_student = log_softmax(student_logits / temperature)
    p_teacher = np.exp(log_p_teacher)
    q_student = np.exp(log_q_student)

    kl_per_sample = np.sum(p_teacher * (log_p_teacher - log_q_student), axis=1)
    loss = kl_per_sample.mean()

    # d/d(student_logits/T) [-sum p log q] = q - p  (standard CE-vs-logits gradient)
    # chain rule through logits/T: multiply by 1/T; then average over batch.
    grad = (q_student - p_teacher) / temperature / n
    return float(loss), grad


def distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    y: np.ndarray,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Combined distillation objective:

        L = alpha * CE(student_logits, y)
            + (1 - alpha) * T^2 * KL(teacher_soft || student_soft)

    The T^2 factor is the standard correction (Hinton et al., 2015) that
    keeps the magnitude of the soft-target gradient comparable to the
    hard-label gradient as temperature grows.

    Returns (loss, dloss/dstudent_logits).
    """
    ce_loss, ce_grad = cross_entropy(student_logits, y)
    kl_loss, kl_grad = kl_distillation(student_logits, teacher_logits, temperature)

    loss = alpha * ce_loss + (1.0 - alpha) * (temperature ** 2) * kl_loss
    grad = alpha * ce_grad + (1.0 - alpha) * (temperature ** 2) * kl_grad
    return float(loss), grad


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == y).mean())
