# Single-Objective Analytic Approach — Rule-Based MDS

Trust quantification for the rule-based Misbehavior Detection System (MDS):
this approach maps the 8 plausibility-check scores of the rule-based MDS (the
trust evidence) to one Subjective Logic (SL) opinion `(b, d, u)` per V2X
message, with belief `b` that the V2X message is benign, disbelief `d` and
uncertainty `u`.

Each trust evidence attribute `x_i` is mapped to a per-attribute SL opinion by
a predefined analytic mapping function with a threshold `thr_i` and the shape
parameters `alpB`, `betB`, `alpD` and `betD`. The per-attribute opinions are
then fused with a fusion operator (CBF, AVG or WBF). This is the analytic
quantification approach of Hermann et al., originally proposed for MDSs.

The **single-objective analytic approach** is the baseline: Optuna selects the
thresholds, shape parameters and fusion operator by F1 alone. All uncertainty
metrics are still computed and logged per trial, which shows what happens when
uncertainty quality is not optimized: lower uncertainty on incorrect than on
correct decisions (the *confident-wrong* problem).

This directory is one of the four quantification approaches in
`quantification-approach/rule-based_mds/`. All four use the same trust
evidence, metrics and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline) (this directory)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach

## Files

| File | Purpose |
|---|---|
| `sweep_optuna_f1only.py` | **Entry point.** Single-objective study (F1 only), best-trial selection, final test. |
| `sweep_optuna.py` | Shared machinery reused from the multi-objective sweep: worker pool (`MetricsPoolHP`, `_trial_task`), search-space construction (`_build_trial_params`), `run_final_test`. Also runnable standalone as the multi-objective sweep. |
| `main.py` | Closed-form per-feature SL opinions, fusion operators (CBF/AVG/WBF), trust discount, `compute_opinions`. |
| `metrics.py` | AURC, misclassification-AUROC, Δu, ECE, Brier, Sensoy evidential loss, macro variants. |
| `data_cache.py` | Builds and reads the binary `.npz` feature cache from the raw per-message JSON dataset. |
| `paper_outputs.py` | Renders Tables 1–5 and the opinion boxplot from an evaluated method. |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 recommended.

## Data

`data_cache.py` expects the raw dataset in this layout:

```
<data-root>/<scenario>_<density>_<attack>/<Train|Validation|Test>/**/*.json
```

Each JSON file holds a list of messages with a `check` dict (8 plausibility/
consistency check outputs in `[0, 1]`, `-1` = not applicable) and an
`attacker` ground-truth label (`0` = benign). Each check output is
converted to an **implausibility** `x = 1 - plausibility`. Data source:
**VeReMi NextGen** (<https://veremi-dataset.github.io/veremi-nextgen>), via
the 8 plausibility/consistency check outputs produced by
the rule-based MDS in `detection-systems/rule-based_mds`.

Features (in column order, `FEATURE_ORDER`, must match `data_cache.py` and
`main.py`): `range_plaus`, `pos_plaus`, `speed_plaus`, `pos_cons`,
`speed_cons`, `pos_speed_cons`, `pos_head_cons`, `intersection`.

Label convention: `y = 1` benign, `y = 0` attacker.

## Using the shipped dataset

The repository ships the ready-made feature cache of the rule-based MDS in
`dataset/rule-based_mds/` (`features_{train,validation,test}.npz` + `metadata.json`,
tracked with Git LFS). To use it, point the scripts to it from this directory:

```bash
export MBD_CACHE_DIR=../../../dataset/rule-based_mds
```

Building the cache from the raw VeReMi NextGen JSON files
(`python data_cache.py --build`) is then not needed. Do not run
`data_cache.py --clear` or `--rebuild` on this directory: both delete the
whole cache directory.

## Usage

```bash
# 1. Build the feature cache (once)
python data_cache.py --build --data-root /path/to/rule_mbd_data --cache-dir /path/to/cache

# 2. Run the sweep
python sweep_optuna_f1only.py --n-trials 2000 --cache-dir /path/to/cache

# Resume an existing study
python sweep_optuna_f1only.py --resume --n-trials 5000

# Micro-F1 instead of macro-F1, or a different sampler
python sweep_optuna_f1only.py --metric f1
python sweep_optuna_f1only.py --sampler tpe

# Only best-trial selection + final test on an existing study
python sweep_optuna_f1only.py --analyze-only
```

### `data_cache.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--build` | – | Build the cache if it doesn't exist yet |
| `--rebuild` | – | Delete and rebuild the cache unconditionally |
| `--info` | – | Print cache status (sample counts, size, build time) |
| `--clear` | – | Delete the cache directory |
| `--data-root` | `../data` (or `$MBD_DATA_ROOT`) | Root folder containing `<scenario>_<density>_<attack>/<split>/**/*.json` |
| `--cache-dir` | `../data/cache` (or `$MBD_CACHE_DIR`) | Where to write/read `features_{train,validation,test}.npz` + `metadata.json` |
| `--num-workers` | `30` | Parallel JSON-parsing workers |

### `sweep_optuna_f1only.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--n-trials` | `40000` | Target number of completed trials. This default matches the 40,000 trials used for the paper results. |
| `--n-workers` | `cpu_count // 2` | Worker processes for parallel trial evaluation. |
| `--cache-dir` | `data_cache.CACHE_DIR_DEFAULT` | Where the `.npz` feature cache lives. |
| `--metric` | `macro_f1` | `macro_f1` (averaged per attack) or `f1` (micro/pooled) — which F1 variant is optimized. |
| `--sampler` | `nsga2` | `nsga2` (identical config to the multi-objective analytic approach, so the objective set is the only difference) or `tpe` (more sample-efficient). |
| `--resume` | off | Resume an existing study instead of starting fresh. |
| `--analyze-only` | off | Skip the sweep; only re-run best-trial reporting + final test on the existing study. |

The Optuna study is stored in a local SQLite file (`f1only_rule_mbd_optuna.db`
by default, via `STUDY_DB`/`STUDY_NAME` constants at the top of
`sweep_optuna_f1only.py`) — delete it (or bump `STUDY_NAME`) if you change
the search-space schema and want a fresh study rather than a `KeyError` on
resume.

## Model

Each of the 8 plausibility checks gets its own closed-form opinion mapping
from its implausibility value `X ∈ [0, 1]` (`X = -1` = not applicable):

```
X <= thr:  b = alpB * (1 - exp(-betB * (X - thr)^2)),  d = 0
X >  thr:  d = alpD * (1 - exp(-betD * (X - thr)^2)),  b = 0
X == N/A:  b = 0, d = 0, u = 1  (vacuous opinion)
u = 1 - b - d
```

The 8 per-feature opinions are then combined into one fused opinion via the
searched fusion operator (see below), and the fused opinion drives the
benign/attacker decision at the fixed operating point `decision_thr = 0.5`.

## Search space

### Per-feature hyperparameters (8 features × 5 params = 40 dims)

| Parameter | Search space | Meaning |
|---|---|---|
| `<feature>_thr` | `[0.0, 1.0]` (linear) | Decision boundary in implausibility space; `X <= thr` → evidence for benign, `X > thr` → evidence for attacker |
| `<feature>_alpB` | `[0.01, 0.99]` (linear) | Maximum belief mass achievable for this feature (saturation of `b`) |
| `<feature>_alpD` | `[0.01, 0.99]` (linear) | Maximum disbelief mass achievable for this feature (saturation of `d`) |
| `<feature>_betB` | `[0.01, 1e9]` (log) | Growth rate of belief with distance from `thr` on the benign side |
| `<feature>_betD` | `[0.01, 1e9]` (log) | Growth rate of disbelief with distance from `thr` on the attacker side |

### Fusion operator (categorical)

| Parameter | Values | Meaning |
|---|---|---|
| `fusion_op` | `cbf`, `avg`, `wbf` | How the 8 per-feature opinions are combined into one fused opinion: Cumulative Belief Fusion (independent sources), Averaging Belief Fusion (dependent/redundant sources), or Weighted Belief Fusion (lower-uncertainty sources weighted more) |

### `kl_weight` — searched here, unlike the multi-objective sweep

| Parameter | Search space | Meaning |
|---|---|---|
| `kl_weight` | `[0.0, 1.0]` (linear) | Lambda in Sensoy's evidential loss `L = MSE + lambda*KL`. **Diagnostic only** — feeds solely into the logged `evidential_loss` metric, has no effect on predictions or F1. Range follows Sensoy et al.'s annealing convention (capped at 1.0). `multi-objective_analytic/`'s `sweep_optuna.py` keeps this fixed at 1.0; this script overrides it locally (in `sweep_optuna_f1only.py`, not in the shared `sweep_optuna.py`) so the diagnostic reflects a swept lambda instead of an arbitrary fixed value. |

## Fixed parameters

| Parameter | Value | Why it's fixed |
|---|---|---|
| `decision_thr` | `0.5` | Identical to `multi-objective_analytic/` (same shared `_build_trial_params`, `DECISION_THR_FIXED`): the only threshold with no free evidence bias (`p >= 0.5 <=> b >= d`), and making it free would let it be tuned against the diagnostic uncertainty metrics. |
| `trust` (per-feature reliability weight) | disabled (`USE_TRUST_DISCOUNT = False`) | Same reasoning as `multi-objective_analytic/`: a previous run with trust enabled let it discount `u` globally, collapsing `macro_misclass_auroc` to 0.40 (worse than random). |

## Objectives

Single objective, maximized on the **validation** split, macro-averaged over
the 13 attack types (see "Attack whitelist" below) unless `--metric f1`
selects the micro variant:

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` (default) / `f1` | maximize | Macro- or micro-averaged F1, attacker = positive class |

All other metrics from the multi-objective sweep (`macro_aurc`,
`macro_misclass_auroc`, `macro_delta_u`, `macro_belief_correctness`,
`evidential_loss`) are still computed and stored per trial as
`trial.user_attrs` — purely diagnostic, not part of the search.

## Attack whitelist

The sweep and the final evaluation both restrict to a fixed set of 13 attack
types (`selected_attacks`, defined in `sweep_optuna.py`), in this order (so
group index `k` means the same attack across every split and every method):

```
constantPositionOffset, randomPositionOffset, positionMirroring,
suddenStop, accelerationMultiplication, feignedBraking,
constantSpeedOffset, randomSpeedOffset, suddenConstantSpeed,
zeroSpeedReport, reversedHeading, dataReplay, dosAttack
```

`timeDelayAttack` and `trafficCongestionSybil` are deliberately excluded:
they cannot be detected by local MDSs.

Any other attack label present in the cache is dropped before computing
metrics, on both validation and test.

## Trial selection

No knee-point selection needed — single objective, so the winner is simply
`study.best_trial` (highest F1). `report_winner_tradeoff()` then prints the
winner's non-optimized diagnostic metrics alongside its F1, to show the
trade-off directly.

## Final test evaluation

Run automatically after the sweep (or standalone via `--analyze-only`) for
the best trial, reusing `sweep_optuna.run_final_test` unchanged (only the
output label differs, `method_prefix="F1only"`):

- Forward pass on Test at the fixed operating point (`decision_thr = 0.5`).
- Outputs: Tables 1–5, per-sample CSV, 12-category extended opinion boxplot,
  results JSON, model JSON.

All final-test artifacts are written under `<SWEEP_ROOT>/final_test/`.

## Outputs

Written under `SWEEP_ROOT` (`sweep_output/f1only`, relative to the working directory, see
`sweep_optuna_f1only.py`):

```
<SWEEP_ROOT>/
├── sweep_summary.csv   # one row per trial: optimized metric, all diagnostic
│                       # metrics, and every search-space parameter
├── best_trial.json     # winning trial's number, objective value, all
│                       # diagnostic metrics, full parameter set
└── final_test/
    ├── table_1_method_metrics_F1only*.csv
    ├── table_2_per_attack_f1_F1only*.csv
    ├── table_3_risk_coverage_F1only*.csv
    ├── table_4_opinions_per_attack_F1only*.csv
    ├── table_5_delta_u_per_attack_F1only*.csv
    ├── boxplot_opinions[_extended]_F1only*.png/.pdf
    ├── opinions_per_sample_F1only*.csv
    └── results_paper_F1only*.json
```

## Reproducing / extending

- Changing the search-space schema (parameter names, adding/removing a
  feature) requires deleting `f1only_rule_mbd_optuna.db` or bumping
  `STUDY_NAME` before resuming — otherwise `--resume`/`--analyze-only` will
  fail with a schema-mismatch error or a `KeyError` while replaying old
  trials.
- This directory intentionally does not modify `sweep_optuna.py` — any fix
  or feature added there benefits both the multi-objective analytic approach and
  this baseline automatically, as long as `_build_trial_params`'s return
  signature stays the same.
