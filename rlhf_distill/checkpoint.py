"""Save/load MLP weights to/from disk via np.savez."""
from __future__ import annotations

import numpy as np

from rlhf_distill.mlp import MLP


def save_checkpoint(model: MLP, path: str) -> None:
    arrays = {}
    for i, w in enumerate(model.weights):
        arrays[f"w{i}"] = w
    for i, b in enumerate(model.biases):
        arrays[f"b{i}"] = b
    arrays["layer_sizes"] = np.array(model.layer_sizes, dtype=np.int64)
    np.savez(path, **arrays)


def load_checkpoint(path: str) -> MLP:
    data = np.load(path)
    layer_sizes = data["layer_sizes"].tolist()
    model = MLP(layer_sizes, seed=0)  # weights immediately overwritten below
    n_layers = len(layer_sizes) - 1
    for i in range(n_layers):
        model.weights[i] = data[f"w{i}"]
        model.biases[i] = data[f"b{i}"]
    return model
