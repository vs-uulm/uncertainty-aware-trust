# Multi-Objective MLP Approach — Prediction-Based MDS

Trust quantification for the prediction-based Misbehavior Detection System
(MDS): this approach maps the 6 prediction errors of the prediction-based MDS
(the trust evidence) to one Subjective Logic (SL) opinion `(b, d, u)` per V2X
message, with belief `b` that the V2X message is benign, disbelief `d` and
uncertainty `u`.

The **multi-objective MLP approach** maps all trust evidence attributes
jointly: a multilayer perceptron (MLP) outputs belief and disbelief evidence
(softplus), which parametrizes a Dirichlet distribution from which the SL
opinion follows. Unlike the analytic and B-Spline approaches, the MLP can
represent evidence that depends on several attributes jointly.

The network is trained with Adam in every trial. Optuna (NSGA-II) tunes the
loss weights and the architecture on the same five objectives as the
multi-objective analytic approach (F1, AUROC, AURC, Δu, belief correctness),
and a knee-point consensus selects one balanced trial, whose trained network
is evaluated on the test split.

This directory is one of the four quantification approaches in
`quantification-approach/prediction-based_mds/`. All four use the same trust
evidence, metrics and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach (this directory)

## Files

| File | Purpose |
|---|---|
| `sweep_optuna_mlp.py` | Optuna multi-objective sweep with inline training, knee-point selection, final test evaluation. Entry point. |
| `data_cache.py` | Builds and reads the binary `.npz` feature cache from the raw per-message JSON dataset. |
| `sl_metrics.py` | Brier, ECE, AURC, misclassification-AUROC, risk-coverage, Cohen's d for Δu, `evaluate_all`. |
| `paper_outputs.py` | Renders Tables 1–5 and the opinion boxplot from an evaluated method. |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is used automatically if available (`torch.cuda.is_available()`);
pass `--force-cpu` to always run on CPU. Python ≥ 3.9 recommended.

## Data

`data_cache.py` expects the same raw dataset layout and 6 continuous
prediction-error features as `multi-objective_analytic/` and
`multi-objective_bspline/` in this tree:

```
<data-root>/<scenario>_<density>_<attack>/<Train|Validation|Test>/**/*.json
```

| `FEATURE_NAMES` entry | Raw JSON field | Meaning |
|---|---|---|
| `pos_err` | `relative_position_error` | Deviation between reported and predicted position |
| `speed_err` | `sender_speed_error` | Deviation between reported and predicted speed |
| `acc_err` | `sender_acceleration_error` | Deviation between reported and predicted acceleration |
| `road_edge` | `distance_to_road_edge_error` | Implausible distance to the road edge |
| `time_err` | `receiver_time_error` | Timestamp/latency inconsistency |
| `heading_err` | `sender_heading_error` | Deviation between reported and predicted heading |

Label convention: `y = 1` benign, `y = 0` attacker. Each feature is
standardized (zero mean, unit variance) using statistics computed on Train
only, then applied to Val/Test.

`sweep_optuna_mlp.py` itself does not expose `--data-root`/`--cache-dir`
flags — it calls `data_cache.ensure_cache()`/`load_split()` with no
override, so it always uses `data_cache.py`'s own
`DATA_ROOT_DEFAULT`/`CACHE_DIR_DEFAULT` constants (`../data` and
`../data/cache`, relative to the working directory). Build the cache at that
location first, or edit those constants, if your data lives elsewhere. Keep the
cache in its own directory: `--clear`/`--rebuild` delete the whole cache
directory.

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
# 1. Build the feature cache (once, at data_cache.py's default location)
python data_cache.py --build

# 2. Run the sweep
python sweep_optuna_mlp.py --n-trials 20 --output-dir ./sweep_output

# Resume an existing study
python sweep_optuna_mlp.py --resume --n-trials 30 --output-dir ./sweep_output

# Only knee-point selection + final test on an existing study
python sweep_optuna_mlp.py --analyze-only --output-dir ./sweep_output

# Only the final test, re-using a cached best_state (no re-selection)
python sweep_optuna_mlp.py --final-test-only --output-dir ./sweep_output
```

### `data_cache.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--build` | – | Build the cache if it doesn't exist yet |
| `--rebuild` | – | Delete and rebuild the cache unconditionally |
| `--info` | – | Print cache status (sample counts, size, build time) |
| `--clear` | – | Delete the cache directory |
| `--data-root` | `../data` | Root folder containing `<scenario>_<density>_<attack>/<split>/**/*.json` |
| `--cache-dir` | `../data/cache` | Where to write/read `features_{train,validation,test}.npz` + `metadata.json` |
| `--num-workers` | `30` | Parallel JSON-parsing workers |

### `sweep_optuna_mlp.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--n-trials` | `1000` | Total number of completed trials to reach (cumulative — safe to re-run). This default matches the 1,000 trials used for the paper results. |
| `--num-epochs` | `3000` (`NUM_EPOCHS_PER_TRIAL`) | Max training epochs per trial (early stopping usually ends it sooner) |
| `--output-dir` | `sweep_output` | Where CSVs, Pareto plots, `balanced_trial.json`, `trial_states.pt`, `final_test/` are written |
| `--resume` | off | Resume the existing Optuna study instead of failing/recreating |
| `--analyze-only` | off | Skip the sweep; only run knee-point selection + final test on an existing study |
| `--final-test-only` | off | Skip both the sweep and the knee-point re-selection; just re-run the final test |
| `--force-cpu` | off | Force CPU even if a CUDA GPU is available |

The Optuna study is stored in a local SQLite file (`mlp_optuna_v2.db` by
default, via `STUDY_DB`/`STUDY_NAME` at the top of `sweep_optuna_mlp.py`).
Per-trial best model weights are cached separately in `trial_states.pt`
under `--output-dir`, since the final test needs to re-evaluate the winning
trial's actual trained weights, not just its hyperparameters.

## Model

For each of the 6 standardized features (1 value each, no basis expansion —
unlike `multi-objective_bspline/`), the 6 values are concatenated into a
6-dimensional input vector and passed through `EvidentialMLPHead`:

```
hidden layers: Linear(prev, h) -> ReLU -> Dropout   for h in (HIDDEN_DIM_1, HIDDEN_DIM_2)
output:        Linear(prev, 2) -> softplus          -> (e_benign, e_attacker), both >= 0

alpha_b = e_benign + 1,  alpha_a = e_attacker + 1,  S = alpha_b + alpha_a
b = e_benign / S,  d = e_attacker / S,  u = clip(2 / S, min=U_MIN)
p = b + BASE_RATE * u
```

All MLP weights are the only trainable parameters, optimized per trial with
Adam (`lr=LR_MLP`, `weight_decay=WEIGHT_DECAY`) for up to `--num-epochs`
epochs, with validation-based early stopping (`val_check_every=25`,
`patience=15`).

### Training loss (per attack subset, then averaged)

```
loss = evidential_loss(alpha, y, neg_weight, kl_weight)
     + delta_u_weight * delta_u_pairwise_hinge_loss(u, p, y, margin=DELTA_U_MARGIN)
     + belief_weight * belief_correctness_loss(b, d, y)
```

- `evidential_loss` = Sensoy's MSE + `kl_weight` · KL term. `kl_weight`
  **anneals from 0 to 1** over the first `KL_ANNEAL_STEPS` epochs
  (`kl_weight = min(1.0, epoch / KL_ANNEAL_STEPS)`) — not searched, but it
  does start at 0 by construction, exactly like `multi-objective_bspline/`.
- `delta_u_pairwise_hinge_loss` pushes `u` on wrong predictions above `u` on
  correct ones by at least `DELTA_U_MARGIN`; its weight anneals from 0 to
  `DELTA_U_WEIGHT_MAX` over `DELTA_U_ANNEAL_STEPS` epochs starting at
  `DELTA_U_START_EPOCH`.
- `belief_correctness_loss` is the class-balanced `(1-b)^2`/`(1-d)^2` term
  pushing belief/disbelief toward 1 on correct predictions; only active when
  `BELIEF_WEIGHT > 0`.
- `neg_weight` per attack subset = max(1, # benign / # attacker) in that
  subset, balancing the evidential loss when an attack group is skewed.
- Unlike `multi-objective_bspline/`, there is **no smoothness penalty** here
  (there are no spline coefficients to regularize).

Early-stopping model selection uses the same composite validation score as
`multi-objective_bspline/`: `val_base - COMPOSITE_DELTA_U_WEIGHT * delta_u_val`
(`val_base` = evidential loss on Val).

The fused opinion drives the benign/attacker decision at the fixed operating
point `decision_thr = 0.5`.

## Search space (7 dims — the model weights are learned, not searched)

| Parameter | Search space | Meaning |
|---|---|---|
| `DELTA_U_WEIGHT_MAX` | `[2.0, 20.0]` (log) | Maximum weight of the `delta_u` hinge-loss term, reached after annealing |
| `DELTA_U_MARGIN` | `[0.20, 0.60]` (step 0.05) | Minimum required `u`-gap between wrong and correct predictions in the hinge loss |
| `BELIEF_WEIGHT` | `[0.2, 5.0]` (log) | Weight of the belief-correctness term in the training loss |
| `HIDDEN_DIM_1` | `{32, 64, 128}` (categorical) | Width of the first hidden layer |
| `HIDDEN_DIM_2` | `{16, 32, 64}` (categorical) | Width of the second hidden layer |
| `DROPOUT` | `[0.0, 0.3]` (step 0.05) | Dropout probability after each hidden layer |
| `LR_MLP` | `[1e-4, 1e-2]` (log) | Adam learning rate |
| `COMPOSITE_DELTA_U_WEIGHT` | `[0.3, 5.0]` (log) | Weight of the `delta_u` bonus in the early-stopping/model-selection score (not the training loss) |

## Fixed parameters

| Parameter | Value | Why it's fixed |
|---|---|---|
| `decision_thr` | `0.5` | Operating point `p = b + 0.5*u >= decision_thr`, identical rationale and value as the other two prediction-based sweeps — the only threshold with no free evidence bias, shared across all methods for Table 1. See the comment above `DECISION_THR_FIXED`/inside `run_final_test` for the full derivation. |
| `kl_weight` (evidential-loss λ) | anneals `0 → 1` over `KL_ANNEAL_STEPS = 200` epochs | Standard Sensoy annealing schedule; not an Optuna search parameter, and not one of the 5 Pareto objectives. |
| `WEIGHT_DECAY` | `1e-5` | Fixed L2 regularization for the Adam optimizer; not searched (only `LR_MLP` is). |
| `DELTA_U_START_EPOCH` / `DELTA_U_ANNEAL_STEPS` | `0` / `250` | Epoch schedule over which the `delta_u` hinge-loss weight ramps from 0 up to the searched `DELTA_U_WEIGHT_MAX`. |
| `DELTA_U_MAX_PAIRS_PER_CLASS` | `1500` | Caps the number of correct/wrong samples used per batch in the pairwise hinge loss, for memory/speed. |
| `U_MIN` | `1e-3` | Floor on `u` to avoid division blow-ups in `S = 2/u`. |
| `BASE_RATE` | `0.5` | Base rate used in `p = b + BASE_RATE * u`. |
| `val_check_every` / `patience` | `25` / `15` | Validation-check cadence and early-stopping patience (in validation checks, not epochs). |

## Pareto objectives

All 5 are computed on the **validation** split (per attack, then macro
averaged over those of the 13 attack types in `selected_attacks` that are
actually present in `ctx.attacks_va_list`; `timeDelayAttack` and `trafficCongestionSybil` are excluded because local MDSs cannot detect them):

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` | maximize | Macro-averaged F1 (attacker = positive class) |
| `macro_aurc` | minimize | Macro-averaged Area Under the Risk-Coverage curve (confidence = `1 - u`) |
| `macro_misclass_auroc` | maximize | Macro-averaged AUROC of `u` as a misclassification detector |
| `macro_delta_u` | maximize | Macro-averaged **mean**-`u` gap between wrong and correct predictions |
| `macro_belief_correctness` | maximize | Macro-averaged, class-balanced mean of `b` on correct benign predictions and `d` on correct attacker predictions |

`macro_delta_u` uses the **mean** gap at the per-attack level, not the
median — a prior version used the median, which has a 50% breakdown point
and is therefore blind to a confident-wrong tail (see the `STUDY_NAME`
comment in `sweep_optuna_mlp.py` for a concrete numeric example: median
0.7262 vs. mean 0.3412 on identical data). Both `macro_delta_u_median` and
`macro_delta_u_cohens_d` are still computed and logged per trial as
non-optimized diagnostics, to show how much the estimator choice moves the
result — but only the mean feeds the objective.

Sampler: `optuna.samplers.NSGAIISampler` with `population_size=24`,
`UniformCrossover`, `crossover_prob=0.9`, `swapping_prob=0.5`, `seed=42` —
matching `multi-objective_bspline/`, and smaller than the multi-objective
analytic approach's 50 since this search space is much lower-dimensional.

## From Pareto front to one model: knee-point selection

Identical procedure to `multi-objective_bspline/` — no constraint
pre-filter, all completed trials go directly into the vote:

1. **Min-max normalize** the 5 objectives to `[0, 1]` (`1` = best) over all
   completed trials.
2. **4 knee-point methods** vote on a winner: `closest_to_utopia`,
   `farthest_from_nadir`, `chebyshev`, `weighted_sum` (equal weights).
3. **Majority vote**; ties are broken by `TIEBREAKER_PRIORITY = [chebyshev,
   closest_to_utopia, weighted_sum, farthest_from_nadir]`.

The winning trial's cached `best_state` (trained weights + architecture
metadata) from `trial_states.pt` is what gets re-evaluated on the test
split.

## Final test evaluation

Run automatically after the sweep (or standalone via `--analyze-only` /
`--final-test-only`) for the single balanced trial:

- Forward pass on Test at the fixed operating point (`decision_thr = 0.5`,
  no refit — same rationale as the other two prediction-based sweeps).
- Tables 1–5, per-sample CSV (`opinions_per_sample_*.csv`), 12-category
  extended opinion boxplot, results JSON, model checkpoint (`.pt`, feature
  means/stds + MLP state dict + architecture + threshold).

All final-test artifacts are written under `<output-dir>/final_test/`.

## Reproducing / extending

- Changing the number of objectives or any Optuna parameter name changes the
  search-space schema. Either delete `mlp_optuna_v2.db` or bump
  `STUDY_NAME` before resuming — otherwise `--resume`/`--analyze-only` will
  fail with a schema-mismatch error. `STUDY_NAME` has already been bumped
  twice in this file's history for exactly this reason (see the comment
  above it) — a prior version's objectives were mislabeled macro but were
  actually micro, and another used the median instead of the mean for
  `macro_delta_u`.
- `trial_states.pt` grows with every trial (it holds the full best-weight
  checkpoint + architecture metadata per trial number). It lives under
  `--output-dir` alongside the study DB; delete it together with the DB when
  starting a genuinely fresh study, not independently.
