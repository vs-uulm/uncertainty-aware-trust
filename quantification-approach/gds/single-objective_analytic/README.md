# Single-Objective Analytic Approach — GDS

Trust quantification for the GNSS Spoofing Detection System (GDS): this
approach maps the 4 anomaly scores `d_power`, `d_shape`, `d_track` and `d_dyn`
of the GDS (the trust evidence) to one Subjective Logic (SL) opinion `(b, d,
u)` per 0.5 s window, with belief `b` that the 0.5 s window is benign,
disbelief `d` and uncertainty `u`.

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
`quantification-approach/gds/`. All four use the same trust evidence, metrics
and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline) (this directory)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach

The directory is **self-contained**: `sweep_optuna_f1only.py` only imports
modules from this directory. The shared modules are verbatim copies of the
ones in `multi-objective_analytic/`, so both sweeps run exactly the same model,
data loading and metrics code. If you change them there, copy them here again.

## Files

| File | Purpose |
|---|---|
| `sweep_optuna_f1only.py` | **Entry point.** Single-objective study (macro-F1 only), winner report, final test. |
| `sweep_optuna.py` | Shared machinery, imported as `hp` (copy of `multi-objective_analytic/sweep_optuna.py`): worker pool (`MetricsPoolHP`, `_trial_task`), search space (`_build_trial_params`), attack groups, `run_final_test`. |
| `sl_model.py` | The SL opinion model: closed-form per-feature opinions, fusion operators (CBF/AVG/WBF), `compute_opinions`, fixed `DECISION_THR = 0.5`. |
| `metrics.py` | AURC, misclassification AUROC, Δu, ECE, Brier, Sensoy evidential loss, macro variants. |
| `data_cache_gnss.py` | Loads the three GNSS split CSVs and normalizes the distances to `[0, 1)`. |
| `paper_outputs.py` | Writes Tables 1–5 and the opinion boxplot for an evaluated method. |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Same data as `multi-objective_analytic/`: the three split CSVs from
`detection-systems/gds/detector_oneclass.py`, which runs on
**TEXBAT** recordings (<https://radionavlab.ae.utexas.edu/texbat/>):

```
<data-dir>/oneclass_sl_train.csv
<data-dir>/oneclass_sl_val.csv
<data-dir>/oneclass_sl_test.csv
```

- The default `--data-dir` is `../data`. The CSVs shipped with this
  repository are in `dataset/gds/`, i.e. `--data-dir ../../../dataset/gds` from
  this directory (as in the examples below).
- Features: `d_power`, `d_shape`, `d_track`, `d_dyn`, mapped to `[0, 1)`
  with `d / (d + s)`.
- Labels: the CSV uses `label = 1` for spoofed. Internally `y = 1` means
  benign and `y = 0` means attacker.

See `multi-objective_analytic/README.md` for details.

## Usage

```bash
# Run the sweep
python sweep_optuna_f1only.py --n-trials 2000 --data-dir ../../../dataset/gds --output-dir results/analytic_f1only

# Resume an existing study
python sweep_optuna_f1only.py --resume --n-trials 5000 --data-dir ../../../dataset/gds --output-dir results/analytic_f1only

# Only winner report + final test on an existing study
python sweep_optuna_f1only.py --analyze-only --data-dir ../../../dataset/gds --output-dir results/analytic_f1only
```

### `sweep_optuna_f1only.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--n-trials` | `40000` | Total number of completed trials to reach (cumulative). This default matches the 40,000 trials used for the paper results. |
| `--n-workers` | `cpu_count() // 2` | Worker processes for parallel trial evaluation |
| `--data-dir` | `../data` | Directory containing the three split CSVs (`--cache-dir` is accepted as an alias) |
| `--output-dir` | `results/analytic_f1only` | Where the summary CSV, best-trial JSON and `final_test/` are written |
| `--resume` | off | Resume the existing study |
| `--analyze-only` | off | Skip the sweep; only re-run the winner report + final test |

The Optuna study is stored in `hp_optuna_gnss_f1only_v1.db` (the
`STUDY_DB`/`STUDY_NAME` constants in `sweep_optuna_f1only.py`). Delete it or
bump `STUDY_NAME` whenever the search space changes.

## Model

Identical to `multi-objective_analytic/`. Each normalized distance
`X ∈ [0, 1)` gets its own closed-form opinion mapping:

```
X <= thr:  b = alpB * (1 - exp(-betB * (X - thr)^2)),  d = 0
X >  thr:  d = alpD * (1 - exp(-betD * (X - thr)^2)),  b = 0
u = 1 - b - d
```

The four opinions are fused with the searched fusion operator. The decision
is benign ⇔ `p = b + 0.5·u ≥ 0.5` ⇔ `b ≥ d`.

## Search space

The model search space is identical to the multi-objective sweep, taken from
the shared `_build_trial_params`.

| Parameter | Search space | Meaning |
|---|---|---|
| `<feature>_thr` | `[0.0, 1.0]` (linear) | Decision boundary in normalized distance space |
| `<feature>_alpB` | `[0.01, 0.99]` (linear) | Maximum belief mass |
| `<feature>_alpD` | `[0.01, 0.99]` (linear) | Maximum disbelief mass |
| `<feature>_betB` | `[0.01, 1e3]` (log) | Growth rate of belief |
| `<feature>_betD` | `[0.01, 1e3]` (log) | Growth rate of disbelief |
| `fusion_op` | `cbf`, `avg`, `wbf` | Fusion operator |

### `kl_weight` (λ): searched here, unlike the multi-objective sweep

| Parameter | Search space | Meaning |
|---|---|---|
| `kl_weight` | `[0.0, 1.0]` (linear, starts at 0) | λ in Sensoy's evidential loss `L = MSE + λ·KL`. **Diagnostic only:** it feeds only the logged `evidential_loss` metric and has no effect on predictions or F1. The range follows Sensoy et al.'s annealing convention (capped at 1.0). `multi-objective_analytic/` keeps λ fixed at 1.0. |

## Fixed parameters

| Parameter | Value | Why it's fixed |
|---|---|---|
| `decision_thr` | `0.5` | Same as `multi-objective_analytic/` (the shared `DECISION_THR_FIXED`): the only threshold with no built-in evidence bias (`p ≥ 0.5 ⇔ b ≥ d`) |
| `trust` | disabled | Same reason as in `multi-objective_analytic/` |
| Sampler | NSGA-II, `population_size=50`, `UniformCrossover`, `crossover_prob=0.9`, `swapping_prob=0.5`, `seed=42` | Configured exactly like the multi-objective sweep, so only the objective set differs. With one objective, NSGA-II behaves like a genetic algorithm with elitism. |

## Objective

A single objective, maximized on the **validation** split:

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` | maximize | F1 per attack group (spoofed = positive class), averaged over `ds3`, `ds4`, `ds7`, `ds8` |

`macro_aurc`, `macro_misclass_auroc`, `macro_delta_u`,
`macro_belief_correctness` and `evidential_loss` are still computed and
stored per trial as `trial.user_attrs`. They are purely diagnostic.

## Trial selection

No knee-point selection. With a single objective, the winner is simply
`study.best_trial` (highest macro-F1). `report_winner()` prints the winner's
non-optimized metrics next to the best and the median values over all
trials, which shows the price of optimizing for F1 only.

## Final test evaluation

Uses the shared `sweep_optuna.run_final_test` unchanged, with the method
prefix `F1only`:

- Forward pass on Test at the fixed threshold `0.5`.
- Outputs: Tables 1–5, per-sample CSV, opinion boxplots, results JSON and
  model JSON.

## Outputs

```
<output-dir>/
├── sweep_summary_f1only.csv   # one row per trial: objective, all diagnostic metrics, all parameters
├── best_trial_f1only.json     # winner: trial number, objective value, parameters, diagnostic metrics
└── final_test/
    ├── table_{1..5}_*_F1only_trial<NNN>_<fusion_op>*.csv
    ├── boxplot_opinions[_extended]_F1only_*.png/.pdf
    ├── opinions_per_sample_F1only_*.csv
    ├── results_paper_F1only_*.json
    └── sl_hp_trust_model_F1only_*.json
```
