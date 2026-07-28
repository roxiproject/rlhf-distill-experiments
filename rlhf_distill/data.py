"""Toy 2D classification task generation, via numpy only.

Two generators are provided:
  - generate_blobs: several Gaussian blobs, one class each.
  - generate_moons: two interleaving crescents (a harder, non-linearly
    separable toy task), procedurally generated with numpy.
"""
from __future__ import annotations

import numpy as np


def generate_blobs(
    n_samples: int = 600,
    n_classes: int = 3,
    n_features: int = 2,
    cluster_std: float = 1.0,
    center_box: float = 6.0,
    seed: int = 0,
):
    """Generate procedurally-placed Gaussian blobs, one per class."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-center_box, center_box, size=(n_classes, n_features))
    per_class = n_samples // n_classes
    xs, ys = [], []
    for c in range(n_classes):
        n_this = per_class if c < n_classes - 1 else n_samples - per_class * (n_classes - 1)
        pts = centers[c] + rng.standard_normal((n_this, n_features)) * cluster_std
        xs.append(pts)
        ys.append(np.full(n_this, c, dtype=np.int64))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def generate_moons(n_samples: int = 600, noise: float = 0.15, seed: int = 0):
    """Generate a two-moons style toy binary classification task via numpy.

    Points on two interleaving half-circles, with Gaussian noise added.
    """
    rng = np.random.default_rng(seed)
    n_per = n_samples // 2
    theta_a = rng.uniform(0, np.pi, size=n_per)
    theta_b = rng.uniform(0, np.pi, size=n_samples - n_per)

    a_x = np.cos(theta_a)
    a_y = np.sin(theta_a)
    b_x = 1.0 - np.cos(theta_b)
    b_y = 0.5 - np.sin(theta_b)

    x = np.concatenate(
        [np.stack([a_x, a_y], axis=1), np.stack([b_x, b_y], axis=1)], axis=0
    )
    x = x + rng.standard_normal(x.shape) * noise
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_samples - n_per, dtype=np.int64)])

    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def train_val_split(x: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def standardize(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """Zero-mean unit-variance standardization. Returns (x_scaled, mean, std)
    so the same statistics can be reapplied to held-out data."""
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std
