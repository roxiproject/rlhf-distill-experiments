"""End-to-end experiment: train a teacher, then train two identically
initialized students — one with plain supervised cross-entropy (baseline),
one with the distillation objective against the trained teacher — on the
same generated toy dataset, and report real measured numbers.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from rlhf_distill.data import generate_moons, standardize, train_val_split
from rlhf_distill.losses import cross_entropy, accuracy
from rlhf_distill.mlp import MLP
from rlhf_distill.train import train_supervised, train_distillation


def run_experiment(
    n_samples: int = 800,
    noise: float = 0.2,
    teacher_hidden: int = 64,
    student_hidden: int = 4,
    teacher_epochs: int = 60,
    student_epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-2,
    temperature: float = 3.0,
    alpha: float = 0.3,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run the full teacher-training + baseline-vs-distillation comparison.
    Returns a dict of real measured results (no fabricated numbers)."""
    t0 = time.time()

    x, y = generate_moons(n_samples=n_samples, noise=noise, seed=seed)
    x, mean, std = standardize(x)
    x_train, y_train, x_val, y_val = train_val_split(x, y, val_frac=0.25, seed=seed)
    n_classes = int(y.max()) + 1
    in_dim = x.shape[1]

    if verbose:
        print(f"Toy dataset: {len(y)} points, {in_dim} features, {n_classes} classes "
              f"({len(y_train)} train / {len(y_val)} val)")

    # 1. Train the teacher on the toy task so its output distribution is non-trivial.
    teacher = MLP([in_dim, teacher_hidden, teacher_hidden, n_classes], seed=seed)
    if verbose:
        print("\n== Training teacher ==")
    teacher_history = train_supervised(
        teacher, x_train, y_train, x_val, y_val,
        epochs=teacher_epochs, batch_size=batch_size, lr=lr, seed=seed, verbose=verbose,
    )
    teacher_final_acc = teacher_history["val_acc"][-1]
    teacher_final_loss = teacher_history["val_loss"][-1]

    # 2. Two identically initialized students.
    student_baseline = MLP([in_dim, student_hidden, n_classes], seed=seed + 1000)
    student_distill = copy.deepcopy(student_baseline)

    if verbose:
        print("\n== Training student (baseline: hard-label CE only) ==")
    baseline_history = train_supervised(
        student_baseline, x_train, y_train, x_val, y_val,
        epochs=student_epochs, batch_size=batch_size, lr=lr, seed=seed, verbose=verbose,
    )

    if verbose:
        print("\n== Training student (distillation: KL + CE against trained teacher) ==")
    distill_history = train_distillation(
        student_distill, teacher, x_train, y_train, x_val, y_val,
        epochs=student_epochs, batch_size=batch_size, lr=lr,
        temperature=temperature, alpha=alpha, seed=seed, verbose=verbose,
    )

    baseline_final_acc = baseline_history["val_acc"][-1]
    baseline_final_loss = baseline_history["val_loss"][-1]
    distill_final_acc = distill_history["val_acc"][-1]
    distill_final_loss = distill_history["val_loss"][-1]

    results = {
        "n_samples": n_samples,
        "teacher_params": teacher.num_params(),
        "student_params": student_baseline.num_params(),
        "teacher_final_val_acc": teacher_final_acc,
        "teacher_final_val_loss": teacher_final_loss,
        "baseline_final_val_acc": baseline_final_acc,
        "baseline_final_val_loss": baseline_final_loss,
        "distill_final_val_acc": distill_final_acc,
        "distill_final_val_loss": distill_final_loss,
        "temperature": temperature,
        "alpha": alpha,
        "wall_time_seconds": time.time() - t0,
        "teacher_history": teacher_history,
        "baseline_history": baseline_history,
        "distill_history": distill_history,
    }

    if verbose:
        print("\n== Summary ==")
        print(f"Teacher params:  {results['teacher_params']}")
        print(f"Student params:  {results['student_params']}  "
              f"(compression ratio: {results['teacher_params'] / results['student_params']:.2f}x)")
        print(f"Teacher   final val_acc={teacher_final_acc:.4f}  val_loss={teacher_final_loss:.4f}")
        print(f"Baseline  final val_acc={baseline_final_acc:.4f}  val_loss={baseline_final_loss:.4f}")
        print(f"Distilled final val_acc={distill_final_acc:.4f}  val_loss={distill_final_loss:.4f}")
        delta = distill_final_acc - baseline_final_acc
        print(f"Distillation vs baseline accuracy delta: {delta:+.4f}")

    return results


if __name__ == "__main__":
    run_experiment()
