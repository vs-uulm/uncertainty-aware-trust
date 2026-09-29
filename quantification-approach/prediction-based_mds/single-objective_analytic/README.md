# Single-Objective Analytic Approach — Prediction-Based MDS

Trust quantification for the prediction-based Misbehavior Detection System
(MDS): this approach maps the 6 prediction errors of the prediction-based MDS
(the trust evidence) to one Subjective Logic (SL) opinion `(b, d, u)` per V2X
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
`quantification-approach/prediction-based_mds/`. All four use the same trust
evidence, metrics and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline) (this directory)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach

## Files

| File | Purpose |
|---|---|
| `sweep_optuna_f1only.py` | Single-objective Optuna sweep, best-trial selection, final test evaluation. **Entry point.** |
| `sweep_optuna.py` | Not run directly — imported as `import sweep_optuna as hp` to reuse its worker pool, hyperparameter suggestion and paper-output machinery. Nearly identical to `multi-objective_analytic/sweep_optuna.py`; the only difference is its own (unused, since `main()` here is never called) `SWEEP_ROOT`. |
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

Same 6 continuous prediction-error features as `multi-objective_analytic/`
in this tree — see that README's "Data" section for the full field mapping.
Label convention: `y = 1` benign, `y = 0` attacker.

## Using the shipped dataset

The repository ships the ready-made feature cache of the prediction-based MDS in
`dataset/prediction-based_mds/` (`features_{train,validation,test}.npz` + `metadata.json`,
tracked with Git LFS). To use it, point the scripts to it from this directory:

```bash
export MBD_CACHE_DIR=../../../dataset/prediction-based_mds
```

Building the cache from the raw VeReMi NextGen JSON files
(`python data_cache.py --build`) is then not needed. Do not run
`data_cache.py --clear` or `--rebuild` on this directory: both delete the
whole cache directory.

## Usage

```bash
# 1. Build the feature cache (once)
python data_cache.py --build --data-root /path/to/data --cache-dir /path/to/cache

# 2. Run the sweep (optimizing macro_f1 by default)
python sweep_optuna_f1only.py --n-trials 2000 --cache-dir /path/to/cache --output-dir ./sweep_output

# Optimize micro-F1 instead
python sweep_optuna_f1only.py --metric f1 --cache-dir /path/to/cache --output-dir ./sweep_output

# Use the sample-efficient TPE sampler instead of NSGA-II
python sweep_optuna_f1only.py --sampler tpe --cache-dir /path/to/cache --output-dir ./sweep_output

# Resume / analysis only
python sweep_optuna_f1only.py --resume --n-trials 5000 --cache-dir /path/to/cache --output-dir ./sweep_output
python sweep_optuna_f1only.py --analyze-only --cache-dir /path/to/cache --output-dir ./sweep_output
```

### `data_cache.py` arguments

Identical to `multi-objective_analytic/data_cache.py` — see that README.

### `sweep_optuna_f1only.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--n-trials` | `40000` | Total number of completed trials to reach (cumulative — safe to re-run). This default matches the 40,000 trials used for the paper results. |
| `--n-workers` | `cpu_count() // 2` | Parallel worker processes evaluating trials |
| `--cache-dir` | `dc.CACHE_DIR_DEFAULT` (`../data/cache`) | Feature cache built in step 1 |
| `--output-dir` | `sweep_output` | Where the summary CSV, `best_trial.json`, `final_test/` are written |
| `--metric` | `macro_f1` | `macro_f1` (averaged over the 13 attack types — `timeDelayAttack` and `trafficCongestionSybil` are excluded because local MDSs cannot detect them — fair vs. the multi-objective sweep) or `f1` (micro, pooled) |
| `--sampler` | `nsga2` | `nsga2` = identically configured to the multi-objective analytic approach (the objective set is the only difference); `tpe` = Optuna's more sample-efficient default single-objective sampler |
| `--resume` | off | Resume the existing Optuna study instead of failing/recreating |
| `--analyze-only` | off | Skip the sweep; only run best-trial selection + final test on an existing study |

The Optuna study lives in its own SQLite file (`f1only_optuna.db`,
`STUDY_NAME = "f1only_singleobj_meandu"`) — a deliberately separate namespace from
the multi-objective sweep's `hp_optuna_v2.db`, so the two never collide even
though both import the same `sweep_optuna.py` module.

## Search space and fixed parameters

**Identical** search space to `multi-objective_analytic/` in this tree,
because both reuse `sweep_optuna._build_trial_params`/`_suggest_feature`:

- Per-feature hyperparameters (6 features × 5 params = 30 dims):
  `<feature>_thr ∈ [0.0, 1.0]`, `<feature>_alpB, <feature>_alpD ∈ [0.01, 0.99]`,
  `<feature>_betB, <feature>_betD ∈ [0.01, 1e9]` (log).
- Fusion operator: `fusion_op ∈ {cbf, avg, wbf}` (categorical).
- Fixed: `decision_thr = 0.5`, `kl_weight = 1.0` (evidential-loss λ, logged
  only — see `multi-objective_analytic/README.md` for the full rationale on
  both), `trust` disabled.

See `multi-objective_analytic/README.md` for the full parameter tables and
the derivation of why `decision_thr` is fixed at 0.5.

## The one real difference: a single objective, no knee-point selection

| | `multi-objective_analytic/` | `single-objective_analytic/` (this sweep) |
|---|---|---|
| Objectives | 5 (`macro_f1`, `macro_aurc`, `macro_misclass_auroc`, `macro_delta_u`, `macro_belief_correctness`) | 1 (`--metric`, default `macro_f1`) |
| Study direction | 5-tuple of `maximize`/`minimize` | single `direction="maximize"` |
| Selection | Pareto front + 4-method knee-point consensus | `study.best_trial` (no Pareto front, no knee methods needed) |
| Sampler | NSGA-II only, `population_size=50` | NSGA-II (same config, default) or TPE (`--sampler tpe`) |

Every trial still computes and logs **all** metrics — including the 4 it
does not optimize against — as Optuna `user_attrs` (`ALL_METRIC_KEYS`).
`report_winner_tradeoff()` prints exactly this trade-off for the winning
trial: the optimized F1 alongside the un-optimized `macro_misclass_auroc`,
`macro_delta_u`, `macro_aurc`, `macro_belief_correctness`, so the collateral
damage from not co-optimizing uncertainty is visible immediately after the
sweep finishes.

## Final test evaluation

Reuses `sweep_optuna.run_final_test` unchanged, with `method_prefix="F1only"`
so its output filenames are distinguishable from the multi-objective sweep's
`HP_balanced_*` files. Produces the same final-test artifacts as
`multi-objective_analytic/` (Tables 1–5, per-sample CSV, extended boxplot,
model JSON) under `<output-dir>/final_test/` — see that README's "Final test
evaluation" section for details.

## Reproducing / extending

- `sweep_optuna.py` in this directory is a near-duplicate of
  `multi-objective_analytic/sweep_optuna.py` kept here so
  `sweep_optuna_f1only.py` can `import sweep_optuna as hp` without a
  cross-directory import. If you change the search space or worker logic in
  one, mirror the change in the other, or the two sweeps will silently
  diverge.
- Changing `ALL_METRIC_KEYS` or the number of objectives changes what gets
  logged/selected, but not the study's storage schema (it's always
  single-objective here) — no `STUDY_NAME` bump is required for that,
  unlike the multi-objective sweeps.
