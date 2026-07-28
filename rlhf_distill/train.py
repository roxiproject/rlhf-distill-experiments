"""Training loops: plain supervised training, and distillation training.

All loops share the same minibatch-SGD-with-Adam structure so that
baseline and distillation runs are as directly comparable as possible.
"""
from __future__ import annotations

import numpy as np

from rlhf_distill.losses import cross_entropy, distillation_loss, accuracy
from rlhf_distill.mlp import MLP
from rlhf_distill.optim import Adam


def _iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = len(y)
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def train_supervised(
    model: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Train `model` with plain hard-label cross-entropy. Returns a history dict."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.get_params(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        epoch_losses = []
        for xb, yb in _iterate_minibatches(x_train, y_train, batch_size, rng):
            logits = model.forward(xb)
            loss, dlogits = cross_entropy(logits, yb)
            grads = model.backward(dlogits)
            opt.step(model.get_params(), grads)
            epoch_losses.append(loss)

        val_logits = model.forward(x_val, cache=False)
        val_loss, _ = cross_entropy(val_logits, y_val)
        val_acc = accuracy(val_logits, y_val)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"[supervised] epoch {epoch:3d}  train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    return history


def train_distillation(
    student: MLP,
    teacher: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-2,
    temperature: float = 2.0,
    alpha: float = 0.5,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Train `student` with the combined distillation objective against a
    fixed, already-trained `teacher`. Returns a history dict."""
    rng = np.random.default_rng(seed)
    opt = Adam(student.get_params(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        epoch_losses = []
        for xb, yb in _iterate_minibatches(x_train, y_train, batch_size, rng):
            teacher_logits = teacher.forward(xb, cache=False)
            student_logits = student.forward(xb)
            loss, dlogits = distillation_loss(
                student_logits, teacher_logits, yb, temperature=temperature, alpha=alpha
            )
            grads = student.backward(dlogits)
            opt.step(student.get_params(), grads)
            epoch_losses.append(loss)

        val_logits = student.forward(x_val, cache=False)
        val_loss, _ = cross_entropy(val_logits, y_val)  # report plain CE for comparability
        val_acc = accuracy(val_logits, y_val)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"[distill]     epoch {epoch:3d}  train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    return history
