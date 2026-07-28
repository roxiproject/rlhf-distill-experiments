"""Ablation sweep over distillation temperature and alpha, run for real
and recorded into a table of measured results (no fabricated numbers).
"""
from __future__ import annotations

import copy

from rlhf_distill.data import generate_moons, standardize, train_val_split
from rlhf_distill.mlp import MLP
from rlhf_distill.train import train_supervised, train_distillation


def run_ablation(
    temperatures: list[float] = (1.0, 2.0, 4.0, 8.0),
    alphas: list[float] = (0.1, 0.3, 0.5, 0.7),
    n_samples: int = 800,
    noise: float = 0.2,
    teacher_hidden: int = 64,
    student_hidden: int = 4,
    teacher_epochs: int = 60,
    student_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-2,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """Sweep (temperature, alpha) combinations, actually training a fresh
    identically-initialized student for each, and return a list of
    real measured result rows."""
    x, y = generate_moons(n_samples=n_samples, noise=noise, seed=seed)
    x, mean, std = standardize(x)
    x_train, y_train, x_val, y_val = train_val_split(x, y, val_frac=0.25, seed=seed)
    n_classes = int(y.max()) + 1
    in_dim = x.shape[1]

    teacher = MLP([in_dim, teacher_hidden, teacher_hidden, n_classes], seed=seed)
    train_supervised(
        teacher, x_train, y_train, x_val, y_val,
        epochs=teacher_epochs, batch_size=batch_size, lr=lr, seed=seed, verbose=False,
    )

    base_student = MLP([in_dim, student_hidden, n_classes], seed=seed + 1000)

    rows = []
    for temperature in temperatures:
        for alpha in alphas:
            student = copy.deepcopy(base_student)
            history = train_distillation(
                student, teacher, x_train, y_train, x_val, y_val,
                epochs=student_epochs, batch_size=batch_size, lr=lr,
                temperature=temperature, alpha=alpha, seed=seed, verbose=False,
            )
            row = {
                "temperature": temperature,
                "alpha": alpha,
                "final_val_acc": history["val_acc"][-1],
                "final_val_loss": history["val_loss"][-1],
                "best_val_acc": max(history["val_acc"]),
            }
            rows.append(row)
            if verbose:
                print(f"T={temperature:>4.1f}  alpha={alpha:>4.2f}  "
                      f"final_val_acc={row['final_val_acc']:.4f}  "
                      f"final_val_loss={row['final_val_loss']:.4f}  "
                      f"best_val_acc={row['best_val_acc']:.4f}")
    return rows


def format_table(rows: list[dict]) -> str:
    header = f"| {'T':>5} | {'alpha':>5} | {'final_val_acc':>14} | {'final_val_loss':>15} | {'best_val_acc':>13} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['temperature']:>5.1f} | {r['alpha']:>5.2f} | {r['final_val_acc']:>14.4f} | "
            f"{r['final_val_loss']:>15.4f} | {r['best_val_acc']:>13.4f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_ablation()
    print()
    print(format_table(results))
