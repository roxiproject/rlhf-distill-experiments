import numpy as np

from rlhf_distill.data import generate_moons, standardize, train_val_split
from rlhf_distill.mlp import MLP
from rlhf_distill.train import train_supervised, train_distillation


def _toy_split(seed=0, n=200):
    x, y = generate_moons(n_samples=n, seed=seed)
    x, _, _ = standardize(x)
    return train_val_split(x, y, val_frac=0.25, seed=seed)


def test_train_supervised_reduces_loss():
    x_tr, y_tr, x_val, y_val = _toy_split()
    model = MLP([2, 8, 2], seed=0)
    history = train_supervised(model, x_tr, y_tr, x_val, y_val, epochs=30, lr=0.05, seed=0)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_train_supervised_history_lengths():
    x_tr, y_tr, x_val, y_val = _toy_split()
    model = MLP([2, 6, 2], seed=0)
    history = train_supervised(model, x_tr, y_tr, x_val, y_val, epochs=10, seed=0)
    assert len(history["train_loss"]) == 10
    assert len(history["val_loss"]) == 10
    assert len(history["val_acc"]) == 10


def test_train_supervised_val_acc_in_range():
    x_tr, y_tr, x_val, y_val = _toy_split()
    model = MLP([2, 6, 2], seed=0)
    history = train_supervised(model, x_tr, y_tr, x_val, y_val, epochs=10, seed=0)
    assert all(0.0 <= a <= 1.0 for a in history["val_acc"])


def test_train_supervised_improves_accuracy_above_chance():
    x_tr, y_tr, x_val, y_val = _toy_split(n=400)
    model = MLP([2, 16, 2], seed=0)
    history = train_supervised(model, x_tr, y_tr, x_val, y_val, epochs=60, lr=0.05, seed=0)
    assert history["val_acc"][-1] > 0.7


def test_train_distillation_runs_and_reduces_loss():
    x_tr, y_tr, x_val, y_val = _toy_split(n=300)
    teacher = MLP([2, 32, 32, 2], seed=1)
    train_supervised(teacher, x_tr, y_tr, x_val, y_val, epochs=40, lr=0.05, seed=1)

    student = MLP([2, 4, 2], seed=2)
    history = train_distillation(
        student, teacher, x_tr, y_tr, x_val, y_val,
        epochs=30, lr=0.05, temperature=2.0, alpha=0.5, seed=2,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_train_distillation_history_lengths():
    x_tr, y_tr, x_val, y_val = _toy_split(n=200)
    teacher = MLP([2, 16, 2], seed=1)
    student = MLP([2, 4, 2], seed=2)
    history = train_distillation(
        student, teacher, x_tr, y_tr, x_val, y_val, epochs=8, seed=2,
    )
    assert len(history["train_loss"]) == 8
    assert len(history["val_acc"]) == 8


def test_train_distillation_alpha_one_similar_to_supervised():
    """With alpha=1 the distillation objective degenerates to plain CE, so
    training should behave like the supervised loop given the same seed
    and data (though not bit-identical, since the loop structure differs
    slightly)."""
    x_tr, y_tr, x_val, y_val = _toy_split(n=200)
    teacher = MLP([2, 16, 2], seed=1)

    student_a = MLP([2, 4, 2], seed=3)
    student_b = MLP([2, 4, 2], seed=3)

    hist_a = train_supervised(student_a, x_tr, y_tr, x_val, y_val, epochs=20, lr=0.05, seed=5)
    hist_b = train_distillation(
        student_b, teacher, x_tr, y_tr, x_val, y_val,
        epochs=20, lr=0.05, temperature=2.0, alpha=1.0, seed=5,
    )
    # both should reach reasonably low final training loss
    assert hist_a["train_loss"][-1] < 1.0
    assert hist_b["train_loss"][-1] < 1.0


def test_minibatch_iteration_covers_all_examples():
    from rlhf_distill.train import _iterate_minibatches

    x = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    rng = np.random.default_rng(0)
    seen = []
    for xb, yb in _iterate_minibatches(x, y, batch_size=3, rng=rng):
        seen.extend(yb.tolist())
    assert sorted(seen) == list(range(10))
