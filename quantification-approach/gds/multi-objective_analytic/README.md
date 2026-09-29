# Multi-Objective Analytic Approach — GDS

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

The **multi-objective analytic approach** uses the same mapping, but Optuna
(NSGA-II) optimizes five objectives jointly on the validation split: F1
(detection performance), AUROC and AURC (uncertainty ranking), Δu (uncertainty
margin) and belief correctness (belief–disbelief margin). From the resulting
Pareto front, one balanced trial is selected by a knee-point consensus and
evaluated on the test split.

This directory is one of the four quantification approaches in
`quantification-approach/gds/`. All four use the same trust evidence, metrics
and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline)
- `multi-objective_analytic/` — multi-objective analytic approach (this directory)
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach

## Files

| File | Purpose |
|---|---|
| `sweep_optuna.py` | **Entry point.** Optuna multi-objective sweep, knee-point selection, final test evaluation. |
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

Python ≥ 3.9 recommended.

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
`d_dyn`. Other columns (`prn`, `t_s`, `captured`, `d_total`, `p_spoof`,
`pred`) are ignored.

- **Features** (column order `FEATURE_ORDER`, must match `sl_model.Attribute`):
  `d_power`, `d_shape`, `d_track`, `d_dyn`. These are squared Mahalanobis
  distances. They are mapped to `[0, 1)` with `d / (d + s)`, using
  per-feature scales `s = 50, 5, 5, 2` (`NORMALIZE = "tanh"` in
  `data_cache_gnss.py`). This mapping never saturates; a chi² CDF would push
  most spoofed `d_power` values to exactly 1.0 and flatten the uncertainty
  signal.
- **Labels:** the CSV uses `label = 1` for spoofed. Internally this is
  inverted to `y = 1` benign, `y = 0` attacker.
- **Groups:** the TEXBAT scenario name serves as the attack group for all
  macro metrics.

Check what the loader sees:

```bash
python data_cache_gnss.py --data-dir ../../../dataset/gds
```

## Usage

```bash
# Run the sweep (cumulative: stops once 2000 trials are completed)
python sweep_optuna.py --n-trials 2000 --data-dir ../../../dataset/gds --output-dir results/analytic_sweep

# Resume an existing study
python sweep_optuna.py --resume --n-trials 5000 --data-dir ../../../dataset/gds --output-dir results/analytic_sweep

# Only knee-point selection + final test on an existing study
python sweep_optuna.py --analyze-only --data-dir ../../../dataset/gds --output-dir results/analytic_sweep
```

### `sweep_optuna.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--n-trials` | `40000` | Total number of completed trials to reach (cumulative, so re-running is safe). This default matches the 40,000 trials used for the paper results. |
| `--n-workers` | `cpu_count() // 2` | Parallel worker processes evaluating trials |
| `--data-dir` | `../data` | Directory containing the three split CSVs (`--cache-dir` is accepted as an alias) |
| `--output-dir` | `results/analytic_sweep` | Where CSVs, Pareto plots, `balanced_trial.json` and `final_test/` are written |
| `--resume` | off | Resume the existing Optuna study |
| `--analyze-only` | off | Skip the sweep; only run knee-point selection + final test |

The Optuna study is stored in a local SQLite file, `hp_optuna_gnss_v1.db`,
set by the `STUDY_DB`/`STUDY_NAME` constants at the top of `sweep_optuna.py`.
If you change the search space, the objectives, `selected_attacks` or the
decision threshold, delete that file or bump `STUDY_NAME`. Otherwise a resume
fails with a schema-mismatch error or a `KeyError`.

## Model

Each distance gets its own closed-form opinion mapping from its normalized
value `X ∈ [0, 1)`:

```
X <= thr:  b = alpB * (1 - exp(-betB * (X - thr)^2)),  d = 0
X >  thr:  d = alpD * (1 - exp(-betD * (X - thr)^2)),  b = 0
u = 1 - b - d
```

The four per-feature opinions are fused sequentially with the searched fusion
operator. The fused opinion gives `p = b + 0.5·u`, and the decision is
benign ⇔ `p ≥ 0.5` ⇔ `b ≥ d`.

## Search space

### Per-feature hyperparameters (4 features × 5 params = 20 dims)

| Parameter | Search space | Meaning |
|---|---|---|
| `<feature>_thr` | `[0.0, 1.0]` (linear) | Decision boundary in normalized distance space; `X <= thr` → evidence for benign, `X > thr` → evidence for spoofed |
| `<feature>_alpB` | `[0.01, 0.99]` (linear) | Maximum belief mass for this feature (saturation of `b`) |
| `<feature>_alpD` | `[0.01, 0.99]` (linear) | Maximum disbelief mass for this feature (saturation of `d`) |
| `<feature>_betB` | `[0.01, 1e3]` (log) | Growth rate of belief with distance from `thr` on the benign side |
| `<feature>_betD` | `[0.01, 1e3]` (log) | Growth rate of disbelief with distance from `thr` on the spoofed side |

`betB`/`betD` are capped at `1e3`, not `1e9`. The features lie in `[0, 1)`,
so larger rates already turn the mapping into a step function: `u` collapses
to 0 and Δu/misclassification AUROC lose their signal.

### Fusion operator (categorical, 1 dim)

| Parameter | Values | Meaning |
|---|---|---|
| `fusion_op` | `cbf`, `avg`, `wbf` | Cumulative Belief Fusion (independent sources), Averaging Belief Fusion (dependent/redundant sources) or Weighted Belief Fusion (lower-uncertainty sources weigh more) |

## Fixed parameters

| Parameter | Value | Why it's fixed |
|---|---|---|
| `decision_thr` | `0.5` | Operating point `p = b + 0.5·u ≥ 0.5 ⇔ b ≥ d`. It is the only threshold with no built-in evidence bias, and it is the same rule the B-Spline/MLP sweeps use. A free threshold could be tuned against the uncertainty objectives. `DECISION_THR_FIXED` in `sweep_optuna.py`. |
| `kl_weight` (λ in Sensoy's `MSE + λ·KL`) | `1.0` | Only used for `evidential_loss`, a diagnostic logged per trial (`trial.user_attrs`). It is **not** one of the 5 objectives, so it has no influence on the search. The single-objective analytic approach (`single-objective_analytic/`) searches it in `[0, 1]` instead. |
| `trust` (per-feature reliability weight) | disabled (`USE_TRUST_DISCOUNT = False`) | Trust weights can raise `u` globally and destroy the misclassification AUROC. In an earlier V2X run with trust enabled, the balanced trial dropped to 0.40. |
| `DELTA_U_STAT` | `"mean"` | Statistic for Δu, used in the objective and in Tables 1/4/5. It is changed in one place, so selection and reporting always agree. |

Sampler: `optuna.samplers.NSGAIISampler` with `population_size=50`,
`UniformCrossover`, `crossover_prob=0.9`, `swapping_prob=0.5` and `seed=42`.

## Objectives

All 5 are computed on the **validation** split and macro-averaged over the
attack groups (see "Attack groups" below):

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` | maximize | F1 per group (spoofed = positive class), then averaged |
| `macro_aurc` | minimize | Area under the risk–coverage curve (confidence = `1 - u`) |
| `macro_misclass_auroc` | maximize | AUROC of `u` as a misclassification detector |
| `macro_delta_u` | maximize | `mean(u │ wrong) − mean(u │ correct)` per group |
| `macro_belief_correctness` | maximize | Class-balanced mean of `b` on correct benign and `d` on correct spoofed predictions |

## Attack groups

`selected_attacks = ["ds3", "ds4", "ds7", "ds8"]`, in this order.

`cleanStatic` is deliberately excluded. It contains only benign windows, so
its F1 is always 0 whatever the model does, and its belief correctness is
undefined. The benign windows *inside* the `ds*` scenarios (before the attack
starts) are kept and carry the false-positive evaluation. The script fails
loudly if a listed group is missing or has only one class.

## Trial selection

1. **Min-max normalize** the 5 objectives to `[0, 1]` (`1` = best) over all
   completed trials.
2. **4 knee-point methods** vote: `closest_to_utopia`, `farthest_from_nadir`,
   `chebyshev`, `weighted_sum`.
3. **Majority vote**. Ties are broken in the order `chebyshev` →
   `closest_to_utopia` → `weighted_sum` → `farthest_from_nadir`.

## Final test evaluation

Runs automatically after the sweep (or with `--analyze-only`) for the
selected trial:

- Forward pass on Test at the fixed threshold `0.5`, with no refit.
- Outputs: Tables 1–5, per-sample CSV, standard and extended
  (12-category) opinion boxplots, results JSON and a model JSON (all
  hyperparameters, fusion operator, threshold).

## Outputs

```
<output-dir>/
├── sweep_summary.csv               # parameters + objectives of all trials, Pareto flag
├── pareto_<obj_x>_vs_<obj_y>.png   # 2-D Pareto projections (10 pairs)
├── balanced_trial.json             # knee votes and the selected trial
└── final_test/
    ├── table_1_method_metrics_<m>.csv
    ├── table_2_per_attack_f1_<m>.csv
    ├── table_3_risk_coverage_<m>.csv
    ├── table_4_opinions_per_attack_<m>.csv
    ├── table_5_delta_u_per_attack_<m>.csv
    ├── boxplot_opinions[_extended]_<m>.png/.pdf
    ├── opinions_per_sample_<m>.csv
    ├── results_paper_<m>.json
    └── sl_hp_trust_model_<m>.json
```

Here `<m>` is `HP_balanced_trial<NNN>_<fusion_op>`.
