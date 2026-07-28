import yaml

from rlhf_distill.cli import DEFAULT_CONFIG, build_parser, load_config, main


def test_parser_run_subcommand_defaults():
    parser = build_parser()
    args = parser.parse_args(["run"])
    assert args.command == "run"
    assert args.config is None
    assert args.teacher is None
    assert args.quiet is False


def test_parser_run_with_config_and_teacher():
    parser = build_parser()
    args = parser.parse_args(["run", "--config", "cfg.yaml", "--teacher", "t.npz", "--quiet"])
    assert args.config == "cfg.yaml"
    assert args.teacher == "t.npz"
    assert args.quiet is True


def test_parser_ablate_subcommand():
    parser = build_parser()
    args = parser.parse_args(["ablate", "--config", "cfg.yaml"])
    assert args.command == "ablate"
    assert args.config == "cfg.yaml"


def test_parser_train_teacher_requires_out():
    import pytest

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train-teacher"])  # missing required --out


def test_parser_train_teacher_with_out():
    parser = build_parser()
    args = parser.parse_args(["train-teacher", "--out", "teacher.npz"])
    assert args.out == "teacher.npz"


def test_parser_no_command_exits():
    import pytest

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_load_config_defaults_when_no_path():
    config = load_config(None)
    assert config == DEFAULT_CONFIG


def test_load_config_overrides_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump({"student_hidden": 99, "lr": 0.5}))
    config = load_config(str(path))
    assert config["student_hidden"] == 99
    assert config["lr"] == 0.5
    # unspecified keys keep their default
    assert config["teacher_hidden"] == DEFAULT_CONFIG["teacher_hidden"]


def test_load_config_empty_yaml_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    config = load_config(str(path))
    assert config == DEFAULT_CONFIG


def test_main_train_teacher_end_to_end(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump({
        "n_samples": 100, "teacher_hidden": 4, "teacher_epochs": 2, "batch_size": 16,
    }))
    out_path = tmp_path / "teacher.npz"
    rc = main(["train-teacher", "--config", str(cfg_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()


def test_main_run_end_to_end_quiet(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump({
        "n_samples": 100, "teacher_hidden": 4, "student_hidden": 2,
        "teacher_epochs": 2, "student_epochs": 2, "batch_size": 16,
    }))
    rc = main(["run", "--config", str(cfg_path), "--quiet"])
    assert rc == 0
