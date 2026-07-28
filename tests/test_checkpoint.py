import os

import numpy as np

from rlhf_distill.checkpoint import load_checkpoint, save_checkpoint
from rlhf_distill.mlp import MLP


def test_checkpoint_roundtrip_weights(tmp_path):
    model = MLP([3, 5, 2], seed=0)
    path = str(tmp_path / "model.npz")
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    for w1, w2 in zip(model.weights, loaded.weights):
        np.testing.assert_array_equal(w1, w2)
    for b1, b2 in zip(model.biases, loaded.biases):
        np.testing.assert_array_equal(b1, b2)


def test_checkpoint_roundtrip_layer_sizes(tmp_path):
    model = MLP([4, 8, 6, 3], seed=0)
    path = str(tmp_path / "model.npz")
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    assert loaded.layer_sizes == model.layer_sizes


def test_checkpoint_file_created(tmp_path):
    model = MLP([2, 3, 2], seed=0)
    path = str(tmp_path / "m.npz")
    save_checkpoint(model, path)
    assert os.path.exists(path)


def test_checkpoint_predictions_match_after_load(tmp_path):
    model = MLP([2, 6, 3], seed=1)
    x = np.random.default_rng(0).standard_normal((10, 2))
    preds_before = model.predict(x)

    path = str(tmp_path / "m.npz")
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    preds_after = loaded.predict(x)

    np.testing.assert_array_equal(preds_before, preds_after)


def test_checkpoint_logits_match_after_load(tmp_path):
    model = MLP([2, 6, 3], seed=2)
    x = np.random.default_rng(1).standard_normal((5, 2))
    logits_before = model.forward(x, cache=False)

    path = str(tmp_path / "m.npz")
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    logits_after = loaded.forward(x, cache=False)

    np.testing.assert_allclose(logits_before, logits_after, atol=1e-12)
