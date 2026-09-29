# Multi-Objective Analytic Approach — Prediction-Based MDS

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

The **multi-objective analytic approach** uses the same mapping, but Optuna
(NSGA-II) optimizes five objectives jointly on the validation split: F1
(detection performance), AUROC and AURC (uncertainty ranking), Δu (uncertainty
margin) and belief correctness (belief–disbelief margin). From the resulting
Pareto front, one balanced trial is selected by a knee-point consensus and
evaluated on the test split.

This directory is one of the four quantification approaches in
`quantification-approach/prediction-based_mds/`. All four use the same trust
evidence, metrics and evaluation tables, so their results compare directly:

- `single-objective_analytic/` — single-objective analytic approach (baseline)
- `multi-objective_analytic/` — multi-objective analytic approach (this directory)
- `multi-objective_bspline/` — multi-objective B-Spline approach
- `multi-objective_mlp/` — multi-objective multilayer perceptron (MLP) approach

## Files

| File | Purpose |
|---|---|
| `sweep_optuna.py` | Optuna multi-objective sweep, knee-point selection, final test evaluation. Entry point. |
| `main.py` | Closed-form per-feature SL opinions, fusion operators (CBF/AVG/WBF), trust discount, `compute_opinions`. |
| `metrics.py` | AURC, misclassification-AUROC, Δu, ECE, Brier, Sensoy evidential loss, macro variants. |
| `data_cache.py` | Builds and reads the binary `.npz` feature cache from the raw per-message JSON dataset. |
| `paper_outputs.py` | Renders Tables 1–5 and the opinion boxplot from an evaluated method. |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`) |

## Setup

```bash
pip install -r requirements.txt
```

Python ≥ 3.9 recommended (no other version-specific features are used).

## Data

`data_cache.py` expects the raw dataset in this layout:

```
<data-root>/<scenario>_<density>_<attack>/<Train|Validation|Test>/**/*.json
```

Each JSON file holds a list of messages with 6 raw prediction-error fields
and an `attacker` ground-truth label (`0` = benign).

Features (in column order, `FEATURE_ORDER` in `sweep_optuna.py`, must match
the column order produced by `data_cache.py`):

| `FEATURE_ORDER` entry | `data_cache.py` name | Raw JSON field | Meaning |
|---|---|---|---|
| `pos_dist` | `pos_err` | `relative_position_error` | Deviation between reported and predicted position |
| `speed_dist` | `speed_err` | `sender_speed_error` | Deviation between reported and predicted speed |
| `acc_dist` | `acc_err` | `sender_acceleration_error` | Deviation between reported and predicted acceleration |
| `road_edge` | `road_edge` | `distance_to_road_edge_error` | Implausible distance to the road edge |
| `time_err` | `time_err` | `receiver_time_error` | Timestamp/latency inconsistency |
| `heading` | `heading_err` | `sender_heading_error` | Deviation between reported and predicted heading |

Unlike the `rule-based_mds` variant, these are continuous error magnitudes
with **no N/A sentinel** — every feature is always defined, so there is no
vacuous per-feature opinion case here.

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

# 2. Run the sweep
python sweep_optuna.py --n-trials 2000 --cache-dir /path/to/cache --output-dir ./sweep_output

# Resume an existing study
python sweep_optuna.py --resume --n-trials 5000 --cache-dir /path/to/cache --output-dir ./sweep_output

# Only knee-point selection + final test on an existing study
python sweep_optuna.py --analyze-only --cache-dir /path/to/cache --output-dir ./sweep_output
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

Both defaults are relative to the working directory — override
`--data-root`/`--cache-dir` if your data lives elsewhere. Keep the cache in its
own directory: `--clear`/`--rebuild` delete the whole cache directory.

### `sweep_optuna.py` arguments

| Flag | Default | Meaning |
|---|---|---|
| `--n-trials` | `40000` | Total number of completed trials to reach (cumulative — safe to re-run). This default matches the 40,000 trials used for the paper results. |
| `--n-workers` | `cpu_count() // 2` | Parallel worker processes evaluating trials |
| `--cache-dir` | `dc.CACHE_DIR_DEFAULT` (`../data/cache`) | Feature cache built in step 1 |
| `--output-dir` | `sweep_output` | Where CSVs, Pareto plots, `balanced_trial.json`, `final_test/` are written |
| `--resume` | off | Resume the existing Optuna study instead of failing/recreating |
| `--analyze-only` | off | Skip the sweep; only run knee-point selection + final test on an existing study |

The Optuna study itself is stored in a local SQLite file (`hp_optuna_v2.db`
by default, via `STUDY_DB`/`STUDY_NAME` constants at the top of
`sweep_optuna.py`) — delete it (or bump `STUDY_NAME`) if you change the
search-space schema (feature list, parameter names, or number of
objectives) and want a fresh study rather than a `KeyError` on resume.

## Model

Each of the 6 prediction-error features gets its own closed-form opinion
mapping from its (non-negative, unbounded) error value `X`:

```
X <= thr:  b = alpB * (1 - exp(-betB * (X - thr)^2)),  d = 0
X >  thr:  d = alpD * (1 - exp(-betD * (X - thr)^2)),  b = 0
u = 1 - b - d
```

The 6 per-feature opinions are then combined into one fused opinion via the
searched fusion operator (see below), and the fused opinion drives the
benign/attacker decision at the fixed operating point `decision_thr = 0.5`.

## Search space

### Per-feature hyperparameters (6 features × 5 params = 30 dims)

| Parameter | Search space | Meaning |
|---|---|---|
| `<feature>_thr` | `[0.0, 1.0]` (linear) | Decision boundary on the error scale; `X <= thr` → evidence for benign, `X > thr` → evidence for attacker |
| `<feature>_alpB` | `[0.01, 0.99]` (linear) | Maximum belief mass achievable for this feature (saturation of `b`) |
| `<feature>_alpD` | `[0.01, 0.99]` (linear) | Maximum disbelief mass achievable for this feature (saturation of `d`) |
| `<feature>_betB` | `[0.01, 1e9]` (log) | Growth rate of belief with distance from `thr` on the benign side |
| `<feature>_betD` | `[0.01, 1e9]` (log) | Growth rate of disbelief with distance from `thr` on the attacker side |

### Fusion operator (categorical)

| Parameter | Values | Meaning |
|---|---|---|
| `fusion_op` | `cbf`, `avg`, `wbf` | How the 6 per-feature opinions are combined into one fused opinion: Cumulative Belief Fusion (independent sources), Averaging Belief Fusion (dependent/redundant sources), or Weighted Belief Fusion (lower-uncertainty sources weighted more) |

## Fixed parameters

| Parameter | Value | Why it's fixed |
|---|---|---|
| `decision_thr` | `0.5` | The operating point `p = b + 0.5*u >= decision_thr`. Fixed at 0.5 because `p >= 0.5 <=> b >= d` — the only threshold with no free evidence bias, and the same convention used by the B-Spline/MLP baselines, making all three methods directly comparable. See the `DECISION_THR_FIXED` comment in `sweep_optuna.py` for the full derivation and a worked counter-example of why a refit breaks the uncertainty metrics. |
| `kl_weight` (= λ in Sensoy's `MSE + λ·KL` evidential loss) | `1.0` | Only used to compute `evidential_loss`, an informational/diagnostic metric logged per trial (`trial.user_attrs`) — it is **not** one of the 5 Pareto objectives and therefore has no influence on the search. Kept fixed at 1.0. The single-objective analytic approach (`single-objective_analytic/`) sweeps this instead — see its README. |
| `trust` (per-feature reliability weight) | disabled (`USE_TRUST_DISCOUNT = False`) | A previous run with trust enabled let it discount `u` globally, collapsing `macro_misclass_auroc` to 0.40 (worse than random). Flip `USE_TRUST_DISCOUNT = True` in `sweep_optuna.py` to re-enable the `<feature>_trust ∈ [0.0, 1.0]` search parameter. |

## Pareto objectives

All 5 are computed on the **validation** split only, macro-averaged over the
13 attack types in `selected_attacks` (fixed order, fixed set — see
"Attack whitelist" below):

| Objective | Direction | Meaning |
|---|---|---|
| `macro_f1` | maximize | Macro-averaged F1 (attacker = positive class), one F1 per attack type, then averaged |
| `macro_aurc` | minimize | Macro-averaged Area Under the Risk-Coverage curve (confidence = `1 - u`) |
| `macro_misclass_auroc` | maximize | Macro-averaged AUROC of `u` as a misclassification detector |
| `macro_delta_u` | maximize | Macro-averaged mean-`u` gap between wrong and correct predictions |
| `macro_belief_correctness` | maximize | Macro-averaged, class-balanced mean of `b` on correct benign predictions and `d` on correct attacker predictions |

Sampler: `optuna.samplers.NSGAIISampler` with `population_size=50`,
`UniformCrossover`, `crossover_prob=0.9`, `swapping_prob=0.5`, `seed=42`
(population is 5x larger than in the B-Spline/MLP sweeps because this search
space is 30-dimensional vs. their much smaller ones).

## Attack whitelist

The sweep and the final evaluation both restrict to a fixed set of 13 attack
types (`selected_attacks`), in this order (so group index `k` means the same
attack across every split and every method):

```
constantPositionOffset, randomPositionOffset, positionMirroring,
suddenStop, accelerationMultiplication, feignedBraking,
constantSpeedOffset, randomSpeedOffset, suddenConstantSpeed,
zeroSpeedReport, reversedHeading, dataReplay, dosAttack
```

`timeDelayAttack` and `trafficCongestionSybil` are deliberately excluded:
they cannot be detected by local MDSs.

Any other attack label present in the cache (e.g. from folders that
happened to exist when the cache was built) is dropped before computing
metrics, on both validation and test. This avoids the sweep and the final
tables silently averaging over different sets of attack groups.

## From Pareto front to one model: knee-point selection

After the sweep, one trial is picked from the completed trials as follows:

1. **Min-max normalize** the 5 objectives to `[0, 1]` (`1` = best) over all
   completed trials.
2. **4 knee-point methods** vote on a winner: `closest_to_utopia`,
   `farthest_from_nadir`, `chebyshev`, `weighted_sum` (equal weights).
3. **Majority vote**; ties are broken by `TIEBREAKER_PRIORITY = [chebyshev,
   closest_to_utopia, weighted_sum, farthest_from_nadir]`.

The winning trial's hyperparameters are what gets replayed on the test split
in the final evaluation.

## Final test evaluation

Run automatically after the sweep (or standalone via `--analyze-only`) for
the single balanced trial:

- Forward pass on Test at the fixed operating point (`decision_thr = 0.5`,
  no refit — see rationale above).
- Tables 1–5, per-sample CSV (`opinions_per_sample_*.csv`), 12-category
  extended opinion boxplot, results JSON, model JSON (all searched
  hyperparameters + fusion op + decision threshold).

All final-test artifacts are written under `<output-dir>/final_test/`.

## Reproducing / extending

- Changing `FEATURE_ORDER`, the number of objectives, or any Optuna
  parameter name changes the search-space schema. Either delete
  `hp_optuna_v2.db` or bump `STUDY_NAME` before resuming — otherwise
  `--resume`/`--analyze-only` will fail with a schema-mismatch error or a
  `KeyError` while replaying old trials.
- `evidential_loss` (Sensoy's `MSE + λ·KL`) is computed and logged for every
  trial but is not searched over or optimized against; treat it as a
  post-hoc diagnostic, not a sweep objective.
