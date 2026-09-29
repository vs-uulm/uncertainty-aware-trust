# Multi-Objective MLP Approach — Rule-Based MDS

Trust quantification for the rule-based Misbehavior Detection System (MDS):
this approach maps the 8 plausibility-check scores of the rule-based MDS (the
trust evidence) to one Subjective Logic (SL) opinion `(b, d, u)` per V2X
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
`quantification-approach/rule-based_mds/`. All four use the same trust
evidence, metrics and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach (this directory)

## Files

| File | Purpose |
|---|---|
| `sweep_optuna.py` | Entry point: sweep, Pareto analysis, knee-point selection, final test run |
| `data_cache.py` | Parses the raw JSON plausibility-check outputs once and caches them as `.npz` |
| `sl_metrics.py` | Metrics (F1, AURC, misclassification AUROC, ECE, Brier, Δu, ...) |
| `paper_outputs.py` | Writes the paper tables (1–5) and boxplots |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |

## Setup

```bash
pip install -r requirements.txt
```

## Data

```
<data_root>/<scenario>_<density>_<attack>/<Train|Validation|Test>/**/*.json
```

Each JSON record holds the check outputs in `check` (8 plausibility checks,
`-1` = not applicable) and the ground-truth label in `attacker` (`0` = benign).
Each check output becomes an **implausibility** `x = 1 - plausibility`.

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
# 1) Build the feature cache (once)
python data_cache.py --build --data-root <path/to/raw_json> --cache-dir <path/to/cache>

# 2) Run the sweep (continues until 20 trials are completed)
python sweep_optuna.py --cache-dir <path/to/cache> --n-trials 20 --out-dir results/

# Resume an existing study and extend it to 30 trials
python sweep_optuna.py --cache-dir <path/to/cache> --resume --n-trials 30 --out-dir results/

# Only analysis + knee selection + final test on existing trials
python sweep_optuna.py --cache-dir <path/to/cache> --analyze-only --out-dir results/
```

Instead of passing `--data-root` / `--cache-dir` every time, you can set the
environment variables `MBD_DATA_ROOT` and `MBD_CACHE_DIR`. Without either, the
defaults `../data` and `../data/cache` are used. If the cache is missing,
`sweep_optuna.py` builds it automatically from `--data-root`.

### `data_cache.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--build` | – | Build the cache (skipped if it already exists) |
| `--rebuild` | – | Delete and rebuild the cache |
| `--info` | – | Show cache status |
| `--clear` | – | Delete the cache |
| `--data-root` | `$MBD_DATA_ROOT` or `../data` | Raw JSON root |
| `--cache-dir` | `$MBD_CACHE_DIR` or `../data/cache` | Cache directory |
| `--num-workers` | CPU count | Parallel parser processes |

### `sweep_optuna.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--data-root` | `$MBD_DATA_ROOT` or `../data` | Raw JSON root (only needed if the cache must be built) |
| `--cache-dir` | `$MBD_CACHE_DIR` or `../data/cache` | Feature cache directory |
| `--n-trials` | `1000` | Total number of completed trials to reach. This default matches the 1,000 trials used for the paper results. |
| `--num-epochs` | `3000` | Maximum number of epochs per trial (early stopping may end sooner) |
| `--out-dir` | `results` | Output directory |
| `--storage` | `sqlite:///mlp_optuna.db` | Optuna storage URL |
| `--study-name` | `mlp_pareto_v6_5obj_meandu_naimpute_lambda0` | Optuna study name |
| `--resume` | off | Continue an existing study |
| `--analyze-only` | off | Skip the sweep; write the summary CSV and Pareto plots, then run knee selection and the final test |
| `--final-test-only` | off | Like `--analyze-only`, but without the summary CSV and Pareto plots |
| `--force-cpu` | off | Use the CPU even if CUDA is available |

## Model

- **Input**: 8 implausibilities of the rule-based MDS checks in `[0, 1]`. Standardization uses
  mean/std of valid train values only. Missing values (N/A) are mean-imputed, and
  each feature can get an optional missingness mask.
- **Network**: `Linear → ReLU → Dropout` ×2 → `Linear(2)` → softplus, which gives the
  evidence `(e_benign, e_attacker)`.
- **SL opinion**: `S = e_b + e_a + 2`, `b = e_b/S`, `d = e_a/S`, `u = 2/S`,
  `p = b + 0.5·u`.

### Training loss

Per attack set:
```
L = L_MSE-evidential + κ(t)·L_KL + λ_Δu(t)·L_hinge + λ_belief·L_belief
```
Computed full-batch per attack. Gradients are averaged over attacks, with one
optimizer step per epoch.

**Checkpoint selection**: minimum of `L_val = L_evidential − λ_val · Δu_val`,
tracked once all warm-ups have finished.

## Search space

All loss weights (λ) start at **0**, so the sweep can turn the matching loss term off
completely. Because a log scale cannot contain 0, these weights are sampled on a
linear scale. The search space is defined in `SEARCH_SPACE_FLOAT` and
`SEARCH_SPACE_CATEGORICAL` at the top of `sweep_optuna.py`.

| Parameter | Range | Scale / step | Description |
|---|---|---|---|
| `DELTA_U_WEIGHT_MAX` (λ<sub>Δu</sub>) | 0.0 – 20.0 | linear, continuous | Max. weight of the pairwise Δu hinge loss (linearly annealed) |
| `DELTA_U_MARGIN` | 0.20 – 0.60 | step 0.05 | Margin of the Δu hinge loss |
| `BELIEF_WEIGHT` (λ<sub>belief</sub>) | 0.0 – 5.0 | linear, continuous | Weight of the class-balanced belief-correctness loss |
| `COMPOSITE_DELTA_U_WEIGHT` (λ<sub>val</sub>) | 0.0 – 5.0 | linear, continuous | Weight of Δu in the validation loss used for early stopping / checkpoint selection |
| `HIDDEN_DIM_1` | {32, 64, 128} | categorical | Width of the first hidden layer |
| `HIDDEN_DIM_2` | {16, 32, 64} | categorical | Width of the second hidden layer |
| `DROPOUT` | 0.0 – 0.3 | step 0.05 | Dropout rate |
| `LR_MLP` | 1e-4 – 1e-2 | log | Adam learning rate |

## Fixed parameters

| Parameter | Value | Description |
|---|---|---|
| `WEIGHT_DECAY` | 1e-5 | Adam weight decay |
| `NUM_EPOCHS_PER_TRIAL` | 3000 | Max. epochs per trial |
| `KL_ANNEAL_STEPS` | 200 | Epochs for linear KL-regularizer warm-up (0 → 1) |
| `DELTA_U_START_EPOCH` | 0 | Epoch at which the Δu loss starts |
| `DELTA_U_ANNEAL_STEPS` | 250 | Epochs for linear warm-up of λ<sub>Δu</sub> |
| `DELTA_U_MAX_PAIRS_PER_CLASS` | 1500 | Subsampled correct/wrong samples per hinge-loss evaluation |
| `VAL_CHECK_EVERY` | 25 | Validation interval (epochs) |
| `PATIENCE` | 15 | Early-stopping patience (validation checks) |
| `DECISION_THR_FIXED` | 0.5 | Decision threshold on `p = b + a·u` — **fixed**, unlike `multi-objective_analytic/`'s `decision_thr`, which is searched |
| `BASE_RATE` | 0.5 | SL base rate `a` |
| `U_MIN` | 1e-3 | Lower clamp for `u` |
| `USE_NA_MASK` | True | Add a binary missingness column per feature (input dim 8 → 16) |
| `NA_SENTINEL` | −1.0 | Marker for "check not applicable" |
| `NSGA_POPULATION_SIZE` | 24 | NSGA-II population size |
| `NSGA_CROSSOVER_PROB` | 0.9 | NSGA-II crossover probability (uniform crossover) |
| `NSGA_SWAPPING_PROB` | 0.5 | NSGA-II swapping probability |
| `SEED` | 42 | Seed for sampler, model init and boxplot subsampling |

## Objectives

All objectives are **macro averages over the 13 attack types**. Each is computed on
the validation split at the fixed decision threshold `p ≥ 0.5` (equivalent to `b ≥ d`).

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` | maximize | F1 score with the attacker class as positive class |
| `macro_aurc` | minimize | Area under the risk–coverage curve (uncertainty-ranked) |
| `macro_misclass_auroc` | maximize | AUROC of `u` for separating wrong from correct predictions |
| `macro_delta_u` | maximize | `mean(u | wrong) − mean(u | correct)` |
| `macro_belief_correctness` | maximize | ½ · (mean `b` on correct benign + mean `d` on correct attackers) |

## Attack whitelist

The sweep and the final evaluation both restrict to a fixed set of 13 attack
types (`selected_attacks`), in this order:

```
constantPositionOffset, randomPositionOffset, positionMirroring,
suddenStop, accelerationMultiplication, feignedBraking,
constantSpeedOffset, randomSpeedOffset, suddenConstantSpeed,
zeroSpeedReport, reversedHeading, dataReplay, dosAttack
```

`timeDelayAttack` and `trafficCongestionSybil` are deliberately excluded:
they cannot be detected by local MDSs.

Identical list to `multi-objective_analytic/` and `multi-objective_bspline/`,
so results are directly comparable across all three approaches.

## Trial selection

Four knee-point methods (closest-to-utopia, farthest-from-nadir, Chebyshev,
weighted sum) each vote for one trial after min-max normalizing the 5
objectives (1 = best). The majority wins; ties are broken in the order
Chebyshev → utopia → weighted sum → nadir.

## Final test evaluation

Only the selected trial runs on the test split, at the fixed decision
threshold. The script writes paper tables, boxplots, a per-sample CSV and
the saved model.

## Outputs

```
<out-dir>/
├── trial_states.pt              # best checkpoint of every trial
├── sweep_summary.csv            # parameters + objectives of all trials, Pareto flag
├── pareto_<obj_a>_vs_<obj_b>.png
├── balanced_trial.json          # knee votes and selected trial
└── final_test/
    ├── table_{1..5}_*.csv       # paper tables
    ├── boxplot_opinions*.png/.pdf
    ├── opinions_per_sample_*.csv
    ├── results_paper_*.json
    └── mlp_raw_model_*.pt       # model + standardization
```
