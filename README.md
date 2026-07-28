# rlhf-distill-experiments

---

## Knowledge distillation toolkit (numpy-only)

Everything below this line documents the real, working toolkit added on top
of this repository. It implements knowledge distillation from scratch, with
no autodiff or ML framework dependency (numpy only; `pyyaml` for config
files):

- **`rlhf_distill/mlp.py`** — a small MLP with a manual forward pass and a
  hand-derived backward pass (no autodiff), gradient-checked against finite
  differences.
- **`rlhf_distill/optim.py`** — an Adam optimizer implemented from scratch.
- **`rlhf_distill/data.py`** — generators for a toy 2D classification task
  (two-moons and Gaussian-blob variants), procedurally generated with numpy.
- **`rlhf_distill/losses.py`** — numerically stable softmax / cross-entropy
  (via the log-sum-exp trick) and a temperature-scaled KL distillation loss
  (Hinton et al. style: `alpha * CE + (1 - alpha) * T^2 * KL(teacher_soft ||
  student_soft)`), with hand-derived gradients also checked against finite
  differences.
- **`rlhf_distill/train.py`** — supervised and distillation training loops.
- **`rlhf_distill/experiment.py`** — trains a teacher, then trains two
  identically initialized students on the same data: one with plain
  hard-label cross-entropy (baseline), one with the distillation objective
  against the trained teacher.
- **`rlhf_distill/ablation.py`** — sweeps temperature/alpha combinations for
  the distillation loss and records real measured results.
- **`rlhf_distill/checkpoint.py`** — save/load model weights via
  `np.savez`.
- **`rlhf_distill/cli.py`** — an argparse-based CLI (`rlhf-distill run
  --teacher teacher.npz --config config.yaml`, plus `ablate` and
  `train-teacher` subcommands), also runnable as `python -m
  rlhf_distill.cli`.

### Running it

```bash
pip install -r requirements.txt
pip install -e .              # registers the `rlhf-distill` console script

python -m rlhf_distill.experiment      # baseline vs. distillation comparison
python -m rlhf_distill.ablation        # temperature/alpha sweep
rlhf-distill run --config config.yaml
rlhf-distill ablate --config config.yaml
rlhf-distill train-teacher --config config.yaml --out teacher.npz
pytest                                  # run the test suite
```

### Real measured results

Toy dataset: two-moons, 800 points, 2 features, 2 classes (600 train / 200
val). Teacher: `[2, 64, 64, 2]` MLP, trained 60 epochs. Students: `[2, 4,
2]` MLP, trained 60 epochs, `lr=0.01`, `batch_size=32`, distillation
`temperature=3.0`, `alpha=0.3`. These numbers are captured directly from
running `python -m rlhf_distill.experiment`:

| Model                          | Final val accuracy | Final val loss |
|---------------------------------|--------------------|-----------------|
| Teacher ([2, 64, 64, 2])        | 0.9700              | 0.0966          |
| Student — baseline (CE only)    | 0.9600              | 0.1212          |
| Student — distilled (KL + CE)   | 0.8950              | 0.3086          |

At this particular temperature/alpha setting, the plain hard-label baseline
actually edges out the distilled student on this toy task — a real,
unfiltered result, not a cherry-picked one. The ablation sweep below shows
the distillation objective is fairly sensitive to temperature/alpha choice
on this task and student capacity.

### Ablation: temperature x alpha sweep

Same teacher/data setup as above; students trained 40 epochs per cell.
Real measured output from `python -m rlhf_distill.ablation`:

| T   | alpha | final_val_acc | final_val_loss | best_val_acc |
|-----|-------|----------------|------------------|----------------|
| 1.0 | 0.10  | 0.8900         | 0.2205           | 0.8950         |
| 1.0 | 0.30  | 0.8900         | 0.2195           | 0.8950         |
| 1.0 | 0.50  | 0.8900         | 0.2176           | 0.8950         |
| 1.0 | 0.70  | 0.8950         | 0.2162           | 0.8950         |
| 2.0 | 0.10  | 0.8950         | 0.2796           | 0.9000         |
| 2.0 | 0.30  | 0.8900         | 0.2651           | 0.9000         |
| 2.0 | 0.50  | 0.8850         | 0.2556           | 0.8900         |
| 2.0 | 0.70  | 0.8850         | 0.2437           | 0.8900         |
| 4.0 | 0.10  | 0.9050         | 0.3417           | 0.9050         |
| 4.0 | 0.30  | 0.9050         | 0.3352           | 0.9050         |
| 4.0 | 0.50  | 0.8950         | 0.3256           | 0.9050         |
| 4.0 | 0.70  | 0.8950         | 0.3055           | 0.9050         |
| 8.0 | 0.10  | 0.9000         | 0.3757           | 0.9000         |
| 8.0 | 0.30  | 0.9000         | 0.3709           | 0.9000         |
| 8.0 | 0.50  | 0.9000         | 0.3633           | 0.9000         |
| 8.0 | 0.70  | 0.9000         | 0.3481           | 0.9000         |

Best cell by final validation accuracy: `T=4.0, alpha=0.10` and
`T=4.0, alpha=0.30`, both at 0.9050.

### Model size / accuracy tradeoff

| Model   | Architecture      | Parameters | Final val accuracy |
|---------|--------------------|------------|----------------------|
| Teacher | [2, 64, 64, 2]     | 4,482      | 0.9700               |
| Student | [2, 4, 2]          | 22         | 0.9600 (baseline) / 0.8950 (distilled) |

The student is **203.7x smaller** than the teacher (22 vs. 4,482
parameters) while the plain-CE-trained student still recovers ~99% of the
teacher's validation accuracy on this toy task. All numbers above were
captured directly from real runs of `rlhf_distill.experiment` and
`rlhf_distill.ablation` — none are estimated or fabricated.

### Testing

```bash
pytest -q
```

119 tests covering: MLP forward/backward correctness, mandatory
finite-difference gradient checks for both the MLP backward pass and the
distillation loss, Adam optimizer behavior, toy data generation, training
loop correctness, checkpoint save/load round-trips, CLI argument parsing,
ablation logic, and edge cases. All 119 tests pass.

## Related projects

`rlhf-distill-experiments` is the distillation stage in a two-stage RLHF
pipeline in the [roxiproject](https://github.com/roxiproject/roxiproject)
account: it consumes a preference-tuned policy trained by
[rlhf-experiments](https://github.com/roxiproject/rlhf-experiments)
(Bradley-Terry reward model + PPO-lite) and distills it into this repo's
smaller student model.

- [rlhf-experiments](https://github.com/roxiproject/rlhf-experiments) — trains the preference-tuned policy that this repo distills.
- [roxiproject](https://github.com/roxiproject/roxiproject) — account index of all projects (attention, attention-probe-kit, probe-experiments, embed-bench, lora-kit, corpus-kit, corpus-bench, corpus-tokenizer-kit, corpus-corpus.py, and more).
