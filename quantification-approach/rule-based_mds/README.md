# Rule-Based MDS — Trust Quantification Approaches

Four trust quantification approaches for the rule-based Misbehavior Detection
System (MDS, `detection-systems/rule-based_mds`). The rule-based MDS of Kamel
et al. runs eight plausibility checks on each received V2X message, e.g.,
whether the sender–receiver distance lies within transmission range, each
yielding a probabilistic misbehavior score. These 8 scores are the trust
evidence; each approach maps them to one Subjective Logic (SL) opinion
`(b, d, u)` per V2X message. The data are from VeReMi NextGen. All four
approaches share the objectives, the attack whitelist (13 attack types) and
the evaluation tables for a fair comparison. `timeDelayAttack` and `trafficCongestionSybil`
are excluded because local MDSs cannot detect them.

| Directory | Quantification approach | Mapping | What Optuna searches | Objective(s) |
|---|---|---|---|---|
| [`single-objective_analytic/`](single-objective_analytic/) | Single-objective analytic approach (baseline) | Predefined analytic mapping per attribute + fusion operator | Same parameters as the multi-objective analytic approach | F1 only — shows the *confident-wrong* problem when uncertainty quality is not optimized |
| [`multi-objective_analytic/`](multi-objective_analytic/) | Multi-objective analytic approach | Predefined analytic mapping per attribute + fusion operator (no training) | 40 per-attribute threshold/shape parameters (8 × 5) + fusion operator | 5 objectives (F1 + 4 uncertainty-quality metrics), NSGA-II |
| [`multi-objective_bspline/`](multi-objective_bspline/) | Multi-objective B-Spline approach | Learned per-attribute B-Spline mapping (evidential) | Training-loss hyperparameters | Same 5 objectives, NSGA-II |
| [`multi-objective_mlp/`](multi-objective_mlp/) | Multi-objective multilayer perceptron (MLP) approach | Learned joint mapping of all 8 attributes (evidential MLP) | Loss weights + architecture | Same 5 objectives, NSGA-II |

Each directory is self-contained with its own README (data layout, setup,
search space, outputs).
