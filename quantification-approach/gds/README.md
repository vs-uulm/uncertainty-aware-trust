# GDS — Trust Quantification Approaches

Four trust quantification approaches for the GNSS Spoofing Detection System
(GDS, `detection-systems/gds`). The GDS uses the features of Iqbal et al.:
GNSS observations are aggregated over 0.5 s windows into four groups, a
one-class model is trained per group on benign data, and the squared
Mahalanobis distance to each group's learned mean yields four anomaly scores
(`d_power`, `d_shape`, `d_track`, `d_dyn`). These scores are the trust
evidence; each approach maps them to one Subjective Logic (SL) opinion
`(b, d, u)` per window. The data are the static TEXBAT scenarios `ds3`, `ds4`,
`ds7` and `ds8`. All four approaches share the objectives, the fixed decision
threshold `p ≥ 0.5` and the evaluation tables, so their results compare
directly.

| Directory | Quantification approach | Mapping | What Optuna searches | Objective(s) |
|---|---|---|---|---|
| [`single-objective_analytic/`](single-objective_analytic/) | Single-objective analytic approach (baseline) | Predefined analytic mapping per attribute + fusion operator | Same 21 dims as the multi-objective analytic approach + `kl_weight` λ ∈ [0, 1] (diagnostic only) | F1 only — shows the *confident-wrong* problem when uncertainty quality is not optimized |
| [`multi-objective_analytic/`](multi-objective_analytic/) | Multi-objective analytic approach | Predefined analytic mapping per attribute + fusion operator (no training) | 21 dims: per-attribute threshold/shape parameters + fusion operator | 5 objectives (F1 + 4 uncertainty-quality metrics), NSGA-II, knee-point consensus |
| [`multi-objective_bspline/`](multi-objective_bspline/) | Multi-objective B-Spline approach | Learned per-attribute B-Spline mapping (evidential) | 4 training-loss hyperparameters (λ weights from 0 + hinge margin) | Same 5 objectives, NSGA-II, knee-point consensus |
| [`multi-objective_mlp/`](multi-objective_mlp/) | Multi-objective multilayer perceptron (MLP) approach | Learned joint mapping of all 4 attributes (evidential MLP) | 8 hyperparameters (λ weights from 0, margin, architecture, learning rate) | Same 5 objectives, NSGA-II, knee-point consensus |

Each directory is self-contained, with its own README (data, setup, search
space, outputs), `requirements.txt` and `.gitignore`. The GDS split CSVs are
shipped with this repository in `dataset/gds/`:

```
dataset/gds/oneclass_sl_train.csv
dataset/gds/oneclass_sl_val.csv
dataset/gds/oneclass_sl_test.csv
```

They are produced by `detection-systems/gds/detector_oneclass.py`. The scripts
default to `gds/data/`; pass `--data-dir ../../../dataset/gds` (`--csv` for the
B-Spline approach) from an approach directory to use the shipped CSVs.

The `rule-based_mds/` and `prediction-based_mds/` sibling trees mirror the
same four-way structure for V2X misbehavior detection.
