# Multi-Objective MLP Approach — GDS

Trust quantification for the GNSS Spoofing Detection System (GDS): this
approach maps the 4 anomaly scores `d_power`, `d_shape`, `d_track` and `d_dyn`
of the GDS (the trust evidence) to one Subjective Logic (SL) opinion `(b, d,
u)` per 0.5 s window, with belief `b` that the 0.5 s window is benign,
disbelief `d` and uncertainty `u`.

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
`quantification-approach/gds/`. All four use the same trust evidence, metrics
and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline)
- `multi-objective_analytic/` — multi-objective analytic approach
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach (this directory)

## Files

| File | Purpose |
|---|---|
| `sweep_optuna_mlp.py` | **Entry point:** sweep, Pareto analysis, knee-point selection, final test run |
| `data_cache_gnss.py` | Loads the three GNSS split CSVs (chi² CDF normalization of the distances) |
| `sl_metrics.py` | Metrics (AURC, misclassification AUROC, ECE, Brier, ...) |
| `paper_outputs.py` | Writes the paper tables (1–5) and boxplots |
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
(<https://radionavlab.ae.utexas.edu/texbat/>). It writes a chronological
60/20/20 split, stratified by label, of every scenario:

```
<data-dir>/oneclass_sl_train.csv
<data-dir>/oneclass_sl_val.csv
<data-dir>/oneclass_sl_test.csv
```

The default `--data-dir` is `../data`, i.e. `quantification-approach/gds/data/`.
The CSVs shipped with this repository are in `dataset/gds/`, i.e.
`--data-dir ../../../dataset/gds` from this directory (as in the examples below).

Required columns: `scenario`, `label`, `d_power`, `d_shape`, `d_track`,
`d_dyn`. All other columns are ignored.

- **Features** (column order): `d_power`, `d_shape`, `d_track`, `d_dyn`.
  They are mapped to `[0, 1]` with the chi² CDF (degrees of freedom 2, 3, 3,
  1 = number of raw features per detector group; `NORMALIZE_CHI2` in
  `data_cache_gnss.py`).
- **Labels:** the CSV uses `label = 1` for spoofed. Internally `y = 1` means
  benign and `y = 0` means spoofed.

```bash
python data_cache_gnss.py --data-dir ../../../dataset/gds   # print a summary of the three splits
```

## Usage

```bash
# 1) Run the sweep (continues until 20 trials are completed)
python sweep_optuna_mlp.py --data-dir ../../../dataset/gds --n-trials 20 --out-dir results/mlp_sweep

# Resume an existing study and extend it to 30 trials
python sweep_optuna_mlp.py --data-dir ../../../dataset/gds --resume --n-trials 30 --out-dir results/mlp_sweep

# Only analysis + knee selection + final test on existing trials
python sweep_optuna_mlp.py --data-dir ../../../dataset/gds --analyze-only --out-dir results/mlp_sweep
```

### `sweep_optuna_mlp.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `../data` | Directory with `oneclass_sl_{train,val,test}.csv` |
| `--out-dir` | `results/mlp_sweep` | Output directory (trial states, CSV, plots, `final_test/`) |
| `--n-trials` | `1000` | Total number of completed trials to reach. This default matches the 1,000 trials used for the paper results. |
| `--num-epochs` | `3000` | Maximum number of epochs per trial (early stopping may end sooner) |
| `--resume` | off | Continue an existing study |
| `--analyze-only` | off | Skip the sweep; write the summary CSV and Pareto plots, then run knee selection and the final test |
| `--final-test-only` | off | Like `--analyze-only`, but without the summary CSV and Pareto plots |
| `--force-cpu` | off | Use the CPU even if CUDA is available |

The Optuna study is stored in `mlp_optuna_v4_mean_du_thr05_lambda0.db`
(the `STUDY_DB`/`STUDY_NAME` constants). Bump `STUDY_NAME` whenever the
search space or objectives change. The best weights of every trial go to
`<out-dir>/trial_states.pt`. `--analyze-only` therefore needs the same
`--out-dir` as the sweep.

## Model

- **Input:** the 4 normalized distances, standardized with the mean/std of
  the Train split.
- **Network:** `Linear → ReLU → Dropout` ×2 → `Linear(2)` → softplus, which
  gives the evidence `(e_benign, e_spoofed)`.
- **SL opinion:** `S = e_b + e_a + 2`, `b = e_b/S`, `d = e_a/S`,
  `u = max(2/S, U_MIN)`, `p = b + 0.5·u`. Decision: benign ⇔ `p ≥ 0.5` ⇔
  `b ≥ d`.

### Training loss

The loss is computed separately for each Train scenario that contains
spoofed samples (`cleanStatic` is benign-only and is skipped):

```
L = L_MSE-evidential + κ(t)·L_KL + λ_Δu(t)·L_hinge + λ_belief·L_belief
```

- `L_MSE-evidential`: the spoofed class is weighted by
  `max(1, n_benign / n_spoofed)`.
- `κ(t) = min(1, t / KL_ANNEAL_STEPS)`.
- `λ_Δu(t) = DELTA_U_WEIGHT_MAX · min(1, t / DELTA_U_ANNEAL_STEPS)`.
- `L_hinge`: pairwise loss pushing `u(wrong) ≥ u(correct) + m`.
- `L_belief`: class-balanced `½ [ E(1-b)² │ benign + E(1-d)² │ spoofed ]`.

Each loss is computed full-batch per scenario. The per-scenario losses are
averaged, with one optimizer step per epoch. A term whose λ is 0 is skipped.

**Checkpoint selection:** minimum of `L_val = L_evidential − λ_val · Δu_val`
on the validation split, tracked once all warm-ups have finished
(epoch 300).

## Search space

All loss weights (λ) start at **0**, so the sweep can turn the matching loss
term off completely. They are sampled on a linear scale, because a log scale
cannot contain 0. The search space is defined in `SEARCH_SPACE_FLOAT` and
`SEARCH_SPACE_CATEGORICAL` at the top of `sweep_optuna_mlp.py`.

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
| `NUM_EPOCHS_PER_TRIAL` | 3000 | Max. epochs per trial (`--num-epochs`) |
| `KL_ANNEAL_STEPS` | 200 | Epochs for the linear KL-regularizer warm-up (0 → 1) |
| `DELTA_U_START_EPOCH` | 0 | Epoch at which the Δu loss starts |
| `DELTA_U_ANNEAL_STEPS` | 250 | Epochs for the linear warm-up of λ<sub>Δu</sub> |
| `DELTA_U_MAX_PAIRS_PER_CLASS` | 1500 | Subsampled correct/wrong samples per hinge-loss evaluation |
| `val_check_every` | 25 | Validation interval (epochs) |
| `patience` | 60 | Early-stopping patience (validation checks) |
| `DECISION_THRESHOLD` | 0.5 | Decision threshold on `p = b + a·u`. **Fixed** in training, validation objectives and the final test; never fitted |
| `BASE_RATE` | 0.5 | SL base rate `a` |
| `U_MIN` | 1e-3 | Lower clamp for `u` |
| `DELTA_U_STATISTIC` | `mean` | Δu statistic in the objective and in the tables |
| NSGA-II | population 10, `UniformCrossover` p = 0.9, swapping 0.5 | Sampler settings |
| Seed | 42 | Seed for sampler, model init and boxplot subsampling |

## Objectives

Each objective is computed on the **validation** split at the fixed decision
threshold `p ≥ 0.5` (equivalent to `b ≥ d`).

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` | maximize | F1 score with spoofed as the positive class |
| `macro_aurc` | minimize | Area under the risk–coverage curve (uncertainty-ranked) |
| `macro_misclass_auroc` | maximize | AUROC of `u` for separating wrong from correct predictions |
| `macro_delta_u` | maximize | `mean(u │ wrong) − mean(u │ correct)` |
| `macro_belief_correctness` | maximize | ½ · (mean `b` on correct benign + mean `d` on correct spoofed) |

A trial is pruned if any objective is NaN.

## Attack groups

The final test evaluates the fixed list `selected_attacks = ["ds3", "ds4",
"ds7", "ds8"]`. Macro metrics are averaged over these four scenarios. Micro
metrics and the per-sample CSV cover the same four scenarios; the benign-only
`cleanStatic` scenario is not included.

## Trial selection

Four knee-point methods (closest-to-utopia, farthest-from-nadir, Chebyshev,
weighted sum) each vote for one trial after the 5 objectives are min-max
normalized (1 = best). The majority wins. Ties are broken in the order
Chebyshev → utopia → weighted sum → nadir.

## Final test evaluation

Only the selected trial runs on the test split. The script writes the paper
tables, boxplots, a per-sample CSV and the saved model, all at the fixed
decision threshold 0.5.

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

Here the method name is `MLP_balanced_trial<NNN>`.
