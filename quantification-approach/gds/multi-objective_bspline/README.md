# Multi-Objective B-Spline Approach — GDS

Trust quantification for the GNSS Spoofing Detection System (GDS): this
approach maps the 4 anomaly scores `d_power`, `d_shape`, `d_track` and `d_dyn`
of the GDS (the trust evidence) to one Subjective Logic (SL) opinion `(b, d,
u)` per 0.5 s window, with belief `b` that the 0.5 s window is benign,
disbelief `d` and uncertainty `u`.

The **multi-objective B-Spline approach** replaces the predefined mapping
function of the analytic approach with one learned from data. Each trust
evidence attribute is expanded into a B-Spline basis, whose learned
coefficients yield belief and disbelief evidence. The summed evidence is
passed through a softplus and parametrizes a Dirichlet distribution, from
which the SL opinion follows.

The coefficients are trained with Adam in every trial. Optuna (NSGA-II) tunes
the training-loss hyperparameters on the same five objectives as the
multi-objective analytic approach (F1, AUROC, AURC, Δu, belief correctness),
and a knee-point consensus selects one balanced trial, whose trained mapping
is evaluated on the test split.

This directory is one of the four quantification approaches in
`quantification-approach/gds/`. All four use the same trust evidence, metrics
and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach (this directory)
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach

## Files

| File | Purpose |
|---|---|
| `sweep_optuna_bspline_gnss.py` | **Entry point:** sweep, knee-point selection, final test run |
| `data_cache_gnss_bspline.py` | Reads the GNSS CSVs once, validates them and caches each split as `.npz` |
| `sl_metrics.py` | Evaluation metrics (AURC, misclassification AUROC, ECE, Brier, ...) |
| `paper_outputs.py` | Writes the paper tables (CSV) and the opinion boxplot |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is used automatically if one is available (`--force-cpu`
disables it).

## Data

The input CSVs come from the GDS in
`detection-systems/gds` (`detector_oneclass.py`). That detector runs on
recordings from the **TEXBAT** dataset
(<https://radionavlab.ae.utexas.edu/texbat/>). `--csv` accepts either:

1. **a directory** (recommended, default `../data`; the CSVs shipped with this
   repository are in `../../../dataset/gds`) that contains the
   detector's chronological 60/20/20 split:
   ```
   <dir>/oneclass_sl_train.csv
   <dir>/oneclass_sl_val.csv
   <dir>/oneclass_sl_test.csv
   ```
2. **a single combined CSV** with a `split` column (`train`/`val`/`test`).

Required columns: `scenario`, `label`, `d_power`, `d_shape`, `d_track`,
`d_dyn`. The columns `d_total`, `p_spoof`, `pred`, `captured`, `prn` and
`t_s` are deliberately **not** used as inputs. Rows with non-finite
distances are dropped, and negative distances raise an error.

- **Features** (column order): `d_power`, `d_shape`, `d_track`, `d_dyn`.
  They are used `raw` by default. `--feature-transform log1p` and
  `--feature-transform chi2_cdf` are also available.
- **Labels:** the CSV uses `label = 0` for clean and `1` for spoofed.
  Internally `y = 1` means benign and `y = 0` means spoofed.
- Both **Train and Validation must contain benign and spoofed samples**.
  The model is supervised; the script raises an error otherwise.

The cache is stored as `.npz` in `--cache-dir` (default `cache/`). It is
rebuilt automatically when the source files, the feature transform or the
schema version change.

```bash
python data_cache_gnss_bspline.py --csv ../../../dataset/gds --build   # build (skipped if up to date)
python data_cache_gnss_bspline.py --csv ../../../dataset/gds --info    # show cache metadata
python data_cache_gnss_bspline.py --csv ../../../dataset/gds --rebuild # delete + rebuild
```

## Usage

```bash
# 1. Run the sweep (builds the cache if needed; training + analysis + final test)
python sweep_optuna_bspline_gnss.py --csv ../../../dataset/gds --n-trials 20 --out-dir results/bspline_sweep

# 2. Continue an existing study up to 30 completed trials
python sweep_optuna_bspline_gnss.py --csv ../../../dataset/gds --resume --n-trials 30 --out-dir results/bspline_sweep

# 3. Re-run only the analysis / knee selection / final test on an existing study
python sweep_optuna_bspline_gnss.py --csv ../../../dataset/gds --analyze-only --out-dir results/bspline_sweep
python sweep_optuna_bspline_gnss.py --csv ../../../dataset/gds --final-test-only --out-dir results/bspline_sweep
```

### `sweep_optuna_bspline_gnss.py` arguments

| Option | Default | Description |
|---|---|---|
| `--csv` | `../data` | Directory with the three split CSVs, or one combined CSV with a `split` column |
| `--cache-dir` | `cache` | Feature cache directory |
| `--feature-transform` | `raw` | `raw`, `log1p` or `chi2_cdf` preprocessing of the four distances |
| `--rebuild-cache` | off | Delete and rebuild the cache before starting |
| `--out-dir` | `results/bspline_sweep` | Output directory (trial states, CSV, plots, `final_test/`) |
| `--n-trials` | `1000` | Total number of completed trials to reach. This default matches the 1,000 trials used for the paper results. |
| `--num-epochs` | `3000` | Max. training epochs per trial |
| `--resume` | off | Continue an existing study |
| `--analyze-only` | off | Skip the sweep; run analysis, knee selection and final test |
| `--final-test-only` | off | Like `--analyze-only`, but without the summary CSV and Pareto plots |
| `--force-cpu` | off | Do not use CUDA |

The Optuna study is stored in `bspline_gnss_optuna_v5_mean_du_thr05_lambda0.db`
(the `STUDY_DB`/`STUDY_NAME` constants). Bump `STUDY_NAME` whenever the
search space or objectives change. The best weights of every trial go to
`<out-dir>/trial_states.pt`. `--analyze-only` therefore needs the same
`--out-dir` as the sweep.

## Model

The B-Spline bases `Φ_f(x)` are fitted on **Train only**
(`sklearn.preprocessing.SplineTransformer`: 16 quantile knots, cubic,
constant extrapolation). The same bases are applied to Validation and Test,
and the per-feature bases are concatenated into `Φ`. Two weight vectors
(initialized to zero) give the evidence for each class:

```
e_b = softplus(Φ · w_b),   e_d = softplus(Φ · w_u)
S   = (e_b + 1) + (e_d + 1)
b = e_b / S,   d = e_d / S,   u = max(2 / S, U_MIN)
p(benign) = b + a · u         (base rate a = 0.5)
decision:  benign  ⇔  p ≥ 0.5  ⇔  b ≥ d
```

### Training loss

Each Train scenario (TEXBAT recording) is a separate training set, so a long
recording cannot dominate the loss through its row count alone. At epoch `t`:

```
L = 1/|A| · Σ_a [ L_EDL-MSE(a) + κ(t) · L_KL(a)
                  + λ_smooth · Σ_f [ smooth(w_b,f) + smooth(w_u,f) ]
                  + λ_Δu(t) · L_hinge(a; m)
                  + λ_bel  · L_belief(a) ]
```

- `L_EDL-MSE`: evidential MSE loss. Within each set, the spoofed class is
  weighted by `n_benign / n_spoofed`.
- `L_KL`: evidential KL regularizer, annealed with `κ(t) = min(1, t / KL_ANNEAL_STEPS)`.
- `L_hinge`: pairwise hinge loss that pushes `u(misclassified) ≥ u(correct) + m`.
- `L_belief`: class-balanced belief loss, `½ [ E(1-b)² │ benign + E(1-d)² │ spoofed ]`.
- `λ_Δu(t) = DELTA_U_WEIGHT_MAX · min(1, t / DELTA_U_ANNEAL_STEPS)`.
- `smooth`: second-order finite-difference penalty on the spline weights.

A term whose λ is 0 is skipped entirely.

**Checkpoint selection.** The validation set is evaluated every
`VAL_CHECK_EVERY` epochs. The checkpoint with the lowest score is kept:

```
score = L_EDL(val) + λ_smooth · smooth − λ_sel · Δu(val)
```

with `Δu = mean(u │ wrong) − mean(u │ correct)`. Tracking starts once the
Δu warm-up has finished (epoch 300). Training stops early after `PATIENCE`
checks without improvement.

## Search space

All λ weights start at **0**, so the sweep can also switch the matching term
off completely. They are sampled linearly, because a log scale cannot contain
0 (`SEARCH_SPACE` at the top of the script).

| Parameter | Symbol | Range | Sampling | Role |
|---|---|---|---|---|
| `DELTA_U_WEIGHT_MAX` | λ_Δu | [0.0, 20.0] | uniform | Final weight of the Δu hinge loss (after warm-up) |
| `DELTA_U_MARGIN` | m | [0.20, 0.60] | uniform, step 0.05 | Margin of the Δu hinge loss |
| `BELIEF_WEIGHT` | λ_bel | [0.0, 5.0] | uniform | Weight of the belief-correctness loss |
| `COMPOSITE_DELTA_U_WEIGHT` | λ_sel | [0.0, 5.0] | uniform | Δu weight in the checkpoint-selection score |

## Fixed parameters

### Training

| Constant | Value | Description |
|---|---|---|
| `NUM_EPOCHS_PER_TRIAL` | 3000 | Max. epochs per trial (`--num-epochs`) |
| Learning rate | 0.03 | Adam learning rate |
| Seed | 42 | Torch seed per trial and NSGA-II sampler seed |
| `val_check_every` | 25 | Epochs between validation checks |
| `patience` | 60 | Early stopping, counted in validation checks |
| `KL_ANNEAL_STEPS` | 200 | Epochs to anneal the KL weight from 0 to 1 |
| `DELTA_U_START_EPOCH` | 0 | First epoch of the Δu loss |
| `DELTA_U_ANNEAL_STEPS` | 250 | Epochs to ramp λ_Δu from 0 to `DELTA_U_WEIGHT_MAX` |
| `DELTA_U_MAX_PAIRS_PER_CLASS` | 1500 | Subsampled correct/wrong samples per hinge-loss batch |
| `LAM_SMOOTH` | 1e-3 | λ_smooth, weight of the spline smoothness penalty |

### Model

| Constant | Value | Description |
|---|---|---|
| Spline knots / degree | 16 / 3 | Quantile knots on Train; 18 basis functions per feature |
| `U_MIN` | 1e-3 | Lower clamp for `u` |
| `BASE_RATE` | 0.5 | SL base rate `a` |
| `DECISION_THRESHOLD` | 0.5 | `p ≥ 0.5` → benign. **Fixed** in training diagnostics, validation objectives and the final test; never fitted |

### NSGA-II sampler

| Setting | Value |
|---|---|
| Population size | 10 |
| Crossover | `UniformCrossover`, probability 0.9 |
| Swapping probability | 0.5 |

## Objectives

All computed on the **validation** split and macro-averaged over the
validation scenarios that contain both classes:

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` | maximize | F1 with spoofed as the positive class |
| `macro_aurc` | minimize | Area under the risk–coverage curve (sorted by `u`) |
| `macro_misclass_auroc` | maximize | AUROC of `u` for separating wrong from correct predictions |
| `macro_delta_u` | maximize | `mean(u │ wrong) − mean(u │ correct)` |
| `macro_belief_correctness` | maximize | `½ [ mean(b │ benign, correct) + mean(d │ spoofed, correct) ]` |

A trial is pruned if any objective is NaN.

## Attack groups

On Test, the macro groups are the scenarios that contain spoofed samples.
They are derived from the data at start-up and normally are `ds3`, `ds4`,
`ds7` and `ds8`. `cleanStatic` (benign only) is not a macro group, but it
still counts in the micro metrics and the per-sample CSV.

## Trial selection

All objectives are min-max normalized over the completed trials, with 1 =
best. Four knee-point methods each vote for one trial: `closest_to_utopia`,
`farthest_from_nadir`, `chebyshev` and `weighted_sum`. The trial with the
most votes wins. Ties are broken in the order `chebyshev` →
`closest_to_utopia` → `weighted_sum` → `farthest_from_nadir`.

## Final test evaluation

Only the balanced trial is evaluated on **Test**, at the fixed decision
threshold 0.5. The test macro Δu uses the same statistic (mean) as the sweep
objective.

## Outputs

```
<out-dir>/
├── trial_states.pt                 # best weights of every trial
├── sweep_summary.csv               # parameters + objectives of all trials, Pareto flag
├── pareto_<obj_x>_vs_<obj_y>.png   # 2-D Pareto projections (10 pairs)
├── balanced_trial.json             # knee-point votes and the selected trial
└── final_test/
    ├── table_1_method_metrics_<m>.csv       # F1/AURC/MAUROC/ECE/Brier/Δu, macro + micro
    ├── table_2_per_attack_f1_<m>.csv
    ├── table_3_risk_coverage_<m>.csv        # risk at 100/90/80/70 % coverage
    ├── table_4_opinions_per_attack_<m>.csv  # mean b/d/u, correct vs misclassified
    ├── table_5_delta_u_per_attack_<m>.csv
    ├── boxplot_opinions[_extended]_<m>.png/.pdf
    ├── opinions_per_sample_<m>.csv
    ├── results_paper_<m>.json
    └── bspline_trust_model_<m>.pt           # splines, weights, threshold
```

Here `<m>` is `BSpline_GNSS_balanced_trial<NNN>`.
