# quantification-approach

Trust quantification approaches that map the outputs of the detection
systems in `../detection-systems` (the *trust evidence*) to one Subjective
Logic (SL) opinion `(b, d, u)` per sample: belief `b` that the sample is
benign, disbelief `d` and uncertainty `u`. The approaches are optimized so
that the opinions support an accurate decision **and** carry informative
uncertainty.

## Detection systems

| Directory | Detection system | Domain / data | Trust evidence |
|---|---|---|---|
| [`rule-based_mds/`](rule-based_mds/) | Rule-based Misbehavior Detection System (MDS), after Kamel et al. | V2X (VeReMi NextGen) | 8 probabilistic plausibility-check scores |
| [`prediction-based_mds/`](prediction-based_mds/) | Prediction-based Misbehavior Detection System (MDS), after Hsu et al. | V2X (VeReMi NextGen) | 6 prediction errors (deviation between predicted and reported attributes) |
| [`gds/`](gds/) | GNSS Spoofing Detection System (GDS), features of Iqbal et al. | GNSS (TEXBAT) | 4 anomaly scores (squared Mahalanobis distances) |

## Quantification approaches

Every detection-system directory contains the same four approaches:

| Directory | Approach | Mapping | Objective(s) |
|---|---|---|---|
| `single-objective_analytic/` | Single-objective analytic approach (baseline, Hermann et al.) | Predefined analytic mapping per attribute + fusion operator (no training) | F1 only; shows the *confident-wrong* problem when uncertainty quality is not optimized |
| `multi-objective_analytic/` | Multi-objective analytic approach | Same mapping as the single-objective analytic approach | 5 objectives, NSGA-II |
| `multi-objective_bspline/` | Multi-objective B-Spline approach | Learned per-attribute B-Spline mapping (evidential) | 5 objectives, NSGA-II |
| `multi-objective_mlp/` | Multi-objective multilayer perceptron (MLP) approach | Learned joint mapping of all attributes (evidential MLP) | 5 objectives, NSGA-II |

## Shared setup

- **Objectives** (on the validation split, macro-averaged over attack
  types: the 13 V2X attack types in `rule-based_mds/` and
  `prediction-based_mds/`, the 4 spoofing scenarios in `gds/`):
  `macro_f1` ↑, `macro_aurc` ↓, `macro_misclass_auroc` ↑, `macro_delta_u` ↑,
  `macro_belief_correctness` ↑. The VeReMi NextGen attacks `timeDelayAttack`
  and `trafficCongestionSybil` are excluded because local MDSs cannot detect
  them.
- **Decision rule:** fixed at `p = b + 0.5·u ≥ 0.5` (⇔ `b ≥ d`), never
  fitted.
- **Trial selection:** four knee-point methods vote on one balanced trial
  from the completed trials (multi-objective sweeps only; see below).
- **Final test:** only the selected trial is evaluated on the test split.
  The outputs are paper Tables 1–5, opinion boxplots, a per-sample CSV and
  the saved model.

## Knee-point selection

A multi-objective sweep does not produce one best trial, but a set of
trade-offs between the 5 objectives. The three multi-objective approaches
therefore pick one "balanced" trial from all completed trials with the same
procedure. The single-objective sweep does not need it: it simply takes the
trial with the highest F1 (`study.best_trial`).

### 1. Normalization

Each objective is min-max normalized over the completed trials to `[0, 1]`,
so that `1` is always best:

```
maximize:  z = (x - min) / (max - min)
minimize:  z = 1 - (x - min) / (max - min)     (used for macro_aurc)
```

If an objective has the same value in every trial, it gets `z = 0.5` for all
trials. Every trial is then a point `z ∈ [0, 1]^5`. The ideal point
(**utopia**) is `(1, …, 1)`, the worst point (**nadir**) is `(0, …, 0)`.
Normalization keeps the objective with the largest value range from
dominating the choice.

### 2. The four knee-point methods

Each method selects exactly one trial:

| Method | Rule | Meaning |
|---|---|---|
| `chebyshev` | minimize `max_j │1 − z_j│` | Trial whose **worst** objective is closest to the ideal. Makes sure no single objective falls far behind. |
| `closest_to_utopia` | minimize `‖z − (1, …, 1)‖₂` | Trial with the smallest Euclidean distance to the ideal point. |
| `weighted_sum` | maximize `(1/5) · Σ_j z_j` | Trial with the highest mean normalized score (all objectives weighted equally). |
| `farthest_from_nadir` | maximize `‖z‖₂` | Trial with the largest Euclidean distance from the worst point. |

If several trials have exactly the same score within one method, that method
picks the one that comes first in the list of completed trials.

### 3. Majority vote

The trial chosen by the most methods wins. The vote (votes per trial,
choice of each method, consensus strength = votes / 4) is stored in
`balanced_trial.json`.

### 4. Tie-breaking

With four votes, the vote can end in a tie: 2–2, or 1–1–1–1 when every method
picks a different trial. The tie is then resolved with this fixed priority:

1. `chebyshev`
2. `closest_to_utopia`
3. `weighted_sum`
4. `farthest_from_nadir`

The winner is the trial selected by the first method in this list whose
choice is among the tied trials. Chebyshev comes first because it is the most
robust choice: it guarantees the best worst-case objective, i.e. a balanced
trial without a collapsed objective.

Because the Chebyshev choice is always one of the tied trials, in practice a
tie always goes to the Chebyshev choice. The rest of the list only applies
if it is changed.

Each approach directory is self-contained, with its own README (data
layout, setup, usage, search space, outputs).
