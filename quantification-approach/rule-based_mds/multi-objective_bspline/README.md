# Multi-Objective B-Spline Approach — Rule-Based MDS

Trust quantification for the rule-based Misbehavior Detection System (MDS):
this approach maps the 8 plausibility-check scores of the rule-based MDS (the
trust evidence) to one Subjective Logic (SL) opinion `(b, d, u)` per V2X
message, with belief `b` that the V2X message is benign, disbelief `d` and
uncertainty `u`.

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
`quantification-approach/rule-based_mds/`. All four use the same trust
evidence, metrics and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach (this directory)
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach

## Files

| File               | Purpose                                                                  |
|--------------------|--------------------------------------------------------------------------|
| `sweep_optuna.py`  | Entry point: sweep, knee-point selection, final test run            |
| `data_cache.py`    | Parses the raw JSON dataset once and caches each split as `.npz`         |
| `sl_metrics.py`    | Evaluation metrics (AURC, misclassification AUROC, ECE, Brier, ...)      |
| `paper_outputs.py` | Writes the paper tables (CSV) and the opinion box plot                    |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is used automatically if one is available (`--force-cpu` disables it).

## Data

`data_cache.py` expects the dataset in this layout:

```
<data-root>/<scenario>_<density>_<attack>/<Train|Validation|Test>/**/*.json
```

Each JSON file contains a list of messages with a `check` dict (plausibility-check outputs
in `[0, 1]`, `-1` = not applicable) and an `attacker` label (`0` = benign).
Each check output becomes an **implausibility** `x = 1 - plausibility`.
Missing, negative or non-finite values are stored as `-1`, meaning "check not
applicable". Such a feature gets an all-zero spline basis, so it contributes no
evidence (a vacuous opinion).

Features (in column order): `range_plaus`, `pos_plaus`, `speed_plaus`,
`pos_cons`, `speed_cons`, `pos_speed_cons`, `pos_head_cons`, `intersection`.

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
python data_cache.py --build --data-root /path/to/dataset --cache-dir cache

# 2. Run the sweep (training + analysis + final test)
python sweep_optuna.py --n-trials 20 --cache-dir cache --out-dir runs/bspline_sweep

# 3. Continue an existing study up to 30 completed trials
python sweep_optuna.py --resume --n-trials 30

# 4. Re-run only analysis / knee selection / final test on an existing study
python sweep_optuna.py --analyze-only
python sweep_optuna.py --final-test-only
```

Default paths can also be set with environment variables:
`MBD_DATA_ROOT` (raw dataset), `MBD_CACHE_DIR` (feature cache),
`MBD_SWEEP_DIR` (sweep output).

### `data_cache.py` arguments

| Flag | Meaning |
|---|---|
| `--build` | Build the cache (skipped if it already exists) |
| `--rebuild` | Delete and rebuild the cache |
| `--info` | Show cache status |
| `--clear` | Delete the cache |
| `--data-root` | Raw JSON root |
| `--cache-dir` | Cache directory |
| `--num-workers` | Parallel parsing workers |

### `sweep_optuna.py` arguments

| Option              | Default                           | Description                                              |
|---------------------|-----------------------------------|----------------------------------------------------------|
| `--n-trials`        | `1000`                                | Total number of completed trials to reach. This default matches the 1,000 trials used for the paper results. |
| `--num-epochs`      | 3000                              | Max. training epochs per trial                           |
| `--population-size` | 10                                | NSGA-II population size                                  |
| `--seed`            | 42                                | Sampler seed                                             |
| `--study-name`      | `bspline_pareto_5obj`             | Optuna study name                                        |
| `--storage`         | `sqlite:///<out-dir>/optuna.db`   | Optuna storage URL                                       |
| `--out-dir`         | `runs/bspline_sweep`              | Output directory (env `MBD_SWEEP_DIR`)                   |
| `--cache-dir`       | `../data/cache`                   | Feature cache (env `MBD_CACHE_DIR`)                      |
| `--data-root`       | `../data`                         | Raw dataset, only used if the cache is missing (env `MBD_DATA_ROOT`) |
| `--num-workers`     | 30                                | Worker processes for building the cache                  |
| `--resume`          | off                               | Continue an existing study                               |
| `--analyze-only`    | off                               | Skip the sweep; run analysis, knee selection, final test |
| `--final-test-only` | off                               | Like `--analyze-only`, but without the CSV and Pareto plots |
| `--per-sample-csv`  | off                               | Also write per-message opinions for the test set (large) |
| `--force-cpu`       | off                               | Do not use CUDA                                          |

## Model

For every feature `f`, the B-Spline basis `Φ_f(x)` is concatenated into `Φ`.
Two weight vectors give the evidence for each class:

```
e_b = softplus(Φ · w_b),   e_d = softplus(Φ · w_u)
S   = (e_b + 1) + (e_d + 1)
b = e_b / S,   d = e_d / S,   u = max(2 / S, U_MIN)
p(benign) = b + a · u         (base rate a = 0.5)
decision:  benign  ⇔  p ≥ 0.5  ⇔  b ≥ d
```

### Training loss

Each attack type `A` is a separate training set. At epoch `t`, the loss is

```
L = λ_smooth · Σ_f [ smooth(w_b,f) + smooth(w_u,f) ]
  + 1/|A| · Σ_a [ L_EDL-MSE(a) + κ(t) · L_KL(a)
                  + λ_Δu(t) · L_hinge(a; m)
                  + λ_bel  · L_belief(a) ]
```

- `L_EDL-MSE`: evidential MSE loss. The attacker class is weighted by `n_benign / n_attacker` within each attack set.
- `L_KL`: evidential KL regulariser, annealed with `κ(t) = min(1, t / KL_ANNEAL_STEPS)`.
- `L_hinge`: pairwise hinge loss that pushes `u(misclassified) ≥ u(correct) + m`.
- `L_belief`: class-balanced belief loss, `½ [ E(1-b)² | benign + E(1-d)² | attacker ]`.
- `λ_Δu(t) = DELTA_U_WEIGHT_MAX · min(1, t / DELTA_U_ANNEAL_STEPS)`.
- `smooth`: second-order finite-difference penalty on the spline weights.

**Checkpoint selection.** The validation set is evaluated every
`VAL_CHECK_EVERY` epochs, starting after the Δu warm-up. The checkpoint with
the lowest score below is kept:

```
score = L_EDL(val) + λ_smooth · smooth − λ_sel · Δu(val)
```

Here `Δu = mean(u | wrong) − mean(u | correct)`. Training stops early after
`PATIENCE` checks without improvement.

## Search space

All λ weights start at **0**, so the sweep can also switch the matching term
off completely. Sampling is linear, because a log scale cannot include 0.

| Parameter                  | Symbol     | Range        | Sampling           | Role                                               |
|----------------------------|------------|--------------|--------------------|----------------------------------------------------|
| `DELTA_U_WEIGHT_MAX`       | λ_Δu       | [0.0, 20.0]  | uniform            | Final weight of the Δu hinge loss (after warm-up)  |
| `DELTA_U_MARGIN`           | m          | [0.20, 0.60] | uniform, step 0.05 | Margin of the Δu hinge loss                        |
| `BELIEF_WEIGHT`            | λ_bel      | [0.0, 5.0]   | uniform            | Weight of the belief-correctness loss              |
| `COMPOSITE_DELTA_U_WEIGHT` | λ_sel      | [0.0, 5.0]   | uniform            | Δu weight in the checkpoint-selection score        |

## Fixed parameters

### Training

| Constant                      | Value     | Description                                             |
|-------------------------------|-----------|-----------------------------------------------------------|
| `NUM_EPOCHS_PER_TRIAL`        | 3000      | Max. epochs per trial (`--num-epochs`)                  |
| `LEARNING_RATE`               | 0.03      | Adam learning rate                                      |
| `SEED`                        | 42        | Torch seed per trial and NSGA-II sampler seed (`--seed`)|
| `VAL_CHECK_EVERY`             | 25        | Epochs between validation checks                        |
| `PATIENCE`                    | 60        | Early stopping, counted in validation checks            |
| `KL_ANNEAL_STEPS`             | 200       | Epochs to anneal the KL weight from 0 to 1              |
| `DELTA_U_START_EPOCH`         | 0         | First epoch of the Δu loss                              |
| `DELTA_U_ANNEAL_STEPS`        | 250       | Epochs to ramp λ_Δu from 0 to `DELTA_U_WEIGHT_MAX`      |
| `DELTA_U_MAX_PAIRS_PER_CLASS` | 1500      | Subsampled correct/wrong samples per hinge-loss batch   |
| `LAM_SMOOTH`                  | 1e-3      | λ_smooth, weight of the spline smoothness penalty       |
| `EVAL_CHUNK_SIZE`             | 200 000   | Chunk size for validation/test forward passes           |

Checkpoints are only tracked from epoch
`DELTA_U_START_EPOCH + DELTA_U_ANNEAL_STEPS + 50` (= 300) onwards.

### Model / basis constants

| Constant                | Value      | Description                                                      |
|-------------------------|------------|--------------------------------------------------------------------|
| `SPLINE_N_KNOTS`        | 16         | Knots per feature spline                                         |
| `SPLINE_DEGREE`         | 3          | Cubic B-Splines (16 + 3 − 1 = 18 basis functions per feature)    |
| `SPLINE_KNOTS_STRATEGY` | `quantile` | Knot placement on valid Train values; falls back to `uniform`    |
| `NA_SENTINEL`           | −1.0       | Marks "check not applicable" (all-zero basis)                  |
| `U_MIN`                 | 1e-3       | Lower clamp for `u`                                              |
| `BASE_RATE`             | 0.5        | SL base rate `a`                                                 |
| `DECISION_THRESHOLD`    | 0.5        | `p ≥ 0.5` → benign — **fixed**, unlike `multi-objective_analytic/`'s `decision_thr`, which is searched |

### NSGA-II sampler

| Constant               | Value            | Description                            |
|------------------------|------------------|------------------------------------------|
| `NSGA_POPULATION_SIZE` | 10               | Population size (`--population-size`)  |
| crossover              | UniformCrossover |                                        |
| `NSGA_CROSSOVER_PROB`  | 0.9              | Crossover probability                  |
| `NSGA_SWAPPING_PROB`   | 0.5              | Gene swapping probability              |

## Objectives

All computed on the **validation** split, macro-averaged over the 13 attack
types (see "Attack whitelist" below):

| Objective                  | Direction | Meaning                                                    |
|----------------------------|-----------|--------------------------------------------------------------|
| `macro_f1`                 | maximize  | F1 with the attacker as positive class                     |
| `macro_aurc`               | minimize  | Area under the risk–coverage curve (sorted by `u`)         |
| `macro_misclass_auroc`     | maximize  | AUROC of `u` for separating wrong from correct predictions |
| `macro_delta_u`            | maximize  | `mean(u | wrong) − mean(u | correct)`                      |
| `macro_belief_correctness` | maximize  | `½ [ mean(b | benign, correct) + mean(d | attacker, correct) ]` |

A trial is pruned if any objective is NaN.

## Attack whitelist

The sweep and the final evaluation both restrict to a fixed set of 13 attack
types (`SELECTED_ATTACKS`), in this order:

```
constantPositionOffset, randomPositionOffset, positionMirroring,
suddenStop, accelerationMultiplication, feignedBraking,
constantSpeedOffset, randomSpeedOffset, suddenConstantSpeed,
zeroSpeedReport, reversedHeading, dataReplay, dosAttack
```

`timeDelayAttack` and `trafficCongestionSybil` are deliberately excluded:
they cannot be detected by local MDSs.

Identical list to `multi-objective_analytic/` and `multi-objective_mlp/`, so
results are directly comparable across all three approaches.

## Trial selection

All objectives are min-max normalised over the completed trials, with 1 = best.
Four knee-point methods each vote for one trial: `closest_to_utopia`,
`farthest_from_nadir`, `chebyshev` and `weighted_sum`. The trial with the most
votes wins. Ties are broken in this order: `chebyshev` → `closest_to_utopia` →
`weighted_sum` → `farthest_from_nadir`.

## Final test evaluation

The balanced trial is evaluated once on **TEST**, at the fixed decision
threshold 0.5.

## Outputs

```
<out-dir>/
├── optuna.db                      # Optuna study (resumable)
├── trial_states.pt                # best weights of every trial
├── sweep_summary.csv              # parameters + objectives of all trials, Pareto flag
├── pareto_<obj_x>_vs_<obj_y>.png  # 2-D Pareto projections (10 pairs)
├── balanced_trial.json            # knee-point votes and the selected trial
└── final_test/
    ├── table_1_method_metrics_<m>.csv       # F1/AURC/MAUROC/ECE/Brier/Δu, macro + micro
    ├── table_1_extended_<m>.csv             # Cohen's d, confident-wrong fraction, Δb, Δd
    ├── table_2_per_attack_f1_<m>.csv
    ├── table_3_risk_coverage_<m>.csv        # risk at 100/90/80/70 % coverage
    ├── table_4_opinions_per_attack_<m>.csv  # median b/d/u, correct vs misclassified
    ├── table_5_delta_u_per_attack_<m>.csv
    ├── boxplot_opinions[_extended]_<m>.png/.pdf
    ├── results_paper_<m>.json
    ├── opinions_per_sample_<m>.csv          # only with --per-sample-csv
    └── bspline_trust_model_<m>.pt           # splines, weights, threshold
```
