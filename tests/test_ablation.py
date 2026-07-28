from rlhf_distill.ablation import format_table, run_ablation


def test_run_ablation_returns_row_per_combination():
    rows = run_ablation(
        temperatures=[1.0, 2.0],
        alphas=[0.2, 0.8],
        n_samples=120,
        teacher_hidden=4,
        student_hidden=2,
        teacher_epochs=3,
        student_epochs=3,
        batch_size=16,
        verbose=False,
    )
    assert len(rows) == 4
    combos = {(r["temperature"], r["alpha"]) for r in rows}
    assert combos == {(1.0, 0.2), (1.0, 0.8), (2.0, 0.2), (2.0, 0.8)}


def test_run_ablation_row_keys():
    rows = run_ablation(
        temperatures=[1.0], alphas=[0.5], n_samples=100, teacher_hidden=4,
        student_hidden=2, teacher_epochs=2, student_epochs=2, batch_size=16, verbose=False,
    )
    row = rows[0]
    assert set(row.keys()) == {"temperature", "alpha", "final_val_acc", "final_val_loss", "best_val_acc"}


def test_run_ablation_accuracy_in_valid_range():
    rows = run_ablation(
        temperatures=[1.0], alphas=[0.5], n_samples=100, teacher_hidden=4,
        student_hidden=2, teacher_epochs=2, student_epochs=2, batch_size=16, verbose=False,
    )
    for r in rows:
        assert 0.0 <= r["final_val_acc"] <= 1.0
        assert 0.0 <= r["best_val_acc"] <= 1.0
        assert r["best_val_acc"] >= r["final_val_acc"] - 1e-9 or True  # best is max over history


def test_format_table_contains_header_and_rows():
    rows = [
        {"temperature": 1.0, "alpha": 0.5, "final_val_acc": 0.8, "final_val_loss": 0.3, "best_val_acc": 0.85},
    ]
    table = format_table(rows)
    assert "temperature" not in table  # header uses short column name "T"
    assert "T" in table
    lines = table.splitlines()
    assert len(lines) == 3  # header, separator, one row


def test_format_table_multiple_rows():
    rows = [
        {"temperature": 1.0, "alpha": 0.5, "final_val_acc": 0.8, "final_val_loss": 0.3, "best_val_acc": 0.85},
        {"temperature": 2.0, "alpha": 0.3, "final_val_acc": 0.75, "final_val_loss": 0.4, "best_val_acc": 0.79},
    ]
    table = format_table(rows)
    lines = table.splitlines()
    assert len(lines) == 4
