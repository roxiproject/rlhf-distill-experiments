from rlhf_distill.experiment import run_experiment


def test_run_experiment_returns_expected_keys():
    results = run_experiment(
        n_samples=150, teacher_hidden=8, student_hidden=2,
        teacher_epochs=3, student_epochs=3, batch_size=16, verbose=False,
    )
    expected_keys = {
        "n_samples", "teacher_params", "student_params",
        "teacher_final_val_acc", "teacher_final_val_loss",
        "baseline_final_val_acc", "baseline_final_val_loss",
        "distill_final_val_acc", "distill_final_val_loss",
        "temperature", "alpha", "wall_time_seconds",
        "teacher_history", "baseline_history", "distill_history",
    }
    assert expected_keys.issubset(results.keys())


def test_run_experiment_teacher_bigger_than_student():
    results = run_experiment(
        n_samples=100, teacher_hidden=16, student_hidden=2,
        teacher_epochs=2, student_epochs=2, batch_size=16, verbose=False,
    )
    assert results["teacher_params"] > results["student_params"]


def test_run_experiment_accuracies_in_range():
    results = run_experiment(
        n_samples=100, teacher_hidden=8, student_hidden=2,
        teacher_epochs=2, student_epochs=2, batch_size=16, verbose=False,
    )
    for key in ["teacher_final_val_acc", "baseline_final_val_acc", "distill_final_val_acc"]:
        assert 0.0 <= results[key] <= 1.0


def test_run_experiment_students_start_from_same_init():
    """Baseline and distilled students must be identically initialized so
    the comparison isolates the effect of the training objective."""
    import copy

    from rlhf_distill.mlp import MLP

    student_a = MLP([2, 4, 2], seed=1000)
    student_b = copy.deepcopy(student_a)
    for wa, wb in zip(student_a.weights, student_b.weights):
        import numpy as np
        np.testing.assert_array_equal(wa, wb)


def test_run_experiment_is_reproducible_given_seed():
    r1 = run_experiment(
        n_samples=100, teacher_hidden=8, student_hidden=2,
        teacher_epochs=3, student_epochs=3, batch_size=16, seed=42, verbose=False,
    )
    r2 = run_experiment(
        n_samples=100, teacher_hidden=8, student_hidden=2,
        teacher_epochs=3, student_epochs=3, batch_size=16, seed=42, verbose=False,
    )
    assert r1["baseline_final_val_acc"] == r2["baseline_final_val_acc"]
    assert r1["distill_final_val_acc"] == r2["distill_final_val_acc"]
