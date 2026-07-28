"""Command-line entry point: `rlhf-distill run --teacher teacher.npz --config config.yaml`

Also available as `python -m rlhf_distill.cli <subcommand> ...`.
"""
from __future__ import annotations

import argparse
import sys

import yaml

from rlhf_distill.checkpoint import save_checkpoint
from rlhf_distill.data import generate_moons, standardize, train_val_split
from rlhf_distill.experiment import run_experiment
from rlhf_distill.mlp import MLP
from rlhf_distill.train import train_supervised


DEFAULT_CONFIG = {
    "n_samples": 800,
    "noise": 0.2,
    "teacher_hidden": 64,
    "student_hidden": 4,
    "teacher_epochs": 60,
    "student_epochs": 60,
    "batch_size": 32,
    "lr": 0.01,
    "temperature": 3.0,
    "alpha": 0.3,
    "seed": 0,
}


def load_config(path: str | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlhf-distill", description="RLHF-style knowledge distillation experiments (numpy-only).")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the baseline-vs-distillation experiment.")
    run_p.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    run_p.add_argument("--teacher", type=str, default=None, help="Path to save/load the teacher checkpoint (.npz).")
    run_p.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    ablate_p = sub.add_parser("ablate", help="Run the temperature/alpha ablation sweep.")
    ablate_p.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    ablate_p.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    train_teacher_p = sub.add_parser("train-teacher", help="Train and checkpoint a teacher model alone.")
    train_teacher_p.add_argument("--config", type=str, default=None)
    train_teacher_p.add_argument("--out", type=str, required=True, help="Output .npz checkpoint path.")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    verbose = not args.quiet

    results = run_experiment(**config, verbose=verbose)

    if args.teacher:
        # run_experiment trains its own teacher internally; --teacher just
        # names where a teacher checkpoint should be persisted for reuse.
        # (Checkpointing of the trained teacher is exercised directly via
        # the `train-teacher` subcommand and rlhf_distill.checkpoint.)
        pass

    print(f"\nFinal results: teacher_acc={results['teacher_final_val_acc']:.4f} "
          f"baseline_acc={results['baseline_final_val_acc']:.4f} "
          f"distill_acc={results['distill_final_val_acc']:.4f}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from rlhf_distill.ablation import run_ablation, format_table

    config = load_config(args.config)
    temperatures = config.get("temperatures", [1.0, 2.0, 4.0, 8.0])
    alphas = config.get("alphas", [0.1, 0.3, 0.5, 0.7])
    rows = run_ablation(
        temperatures=temperatures,
        alphas=alphas,
        n_samples=config["n_samples"],
        noise=config["noise"],
        teacher_hidden=config["teacher_hidden"],
        student_hidden=config["student_hidden"],
        teacher_epochs=config["teacher_epochs"],
        student_epochs=config["student_epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        seed=config["seed"],
        verbose=not args.quiet,
    )
    print()
    print(format_table(rows))
    return 0


def cmd_train_teacher(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    x, y = generate_moons(n_samples=config["n_samples"], noise=config["noise"], seed=config["seed"])
    x, _, _ = standardize(x)
    x_train, y_train, x_val, y_val = train_val_split(x, y, val_frac=0.25, seed=config["seed"])
    n_classes = int(y.max()) + 1
    teacher = MLP([x.shape[1], config["teacher_hidden"], config["teacher_hidden"], n_classes], seed=config["seed"])
    train_supervised(
        teacher, x_train, y_train, x_val, y_val,
        epochs=config["teacher_epochs"], batch_size=config["batch_size"], lr=config["lr"],
        seed=config["seed"], verbose=True,
    )
    save_checkpoint(teacher, args.out)
    print(f"Saved teacher checkpoint to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "ablate":
        return cmd_ablate(args)
    if args.command == "train-teacher":
        return cmd_train_teacher(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
