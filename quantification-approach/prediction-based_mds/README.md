# Prediction-Based MDS — Trust Quantification Approaches

Four trust quantification approaches for the prediction-based Misbehavior
Detection System (MDS, `detection-systems/prediction-based_mds`). The
prediction-based MDS builds on Hsu et al.: an MLP trained on benign V2X
messages predicts each message's attributes from the vehicle's preceding
messages. The deviations between predicted and reported values for six
attributes (position, speed, acceleration, road-edge distance, time, heading)
are the trust evidence; each approach maps them to one Subjective Logic (SL)
opinion `(b, d, u)` per V2X message. The data are from VeReMi NextGen. All
four approaches share the 5 evaluation metrics, the attack whitelist (13
attack types) and the test-set tables, so they compare directly. See each
subdirectory's README for details.
`timeDelayAttack` and `trafficCongestionSybil`
are excluded because local MDSs cannot detect them.

| Directory | Quantification approach | Mapping | What Optuna searches | Objective(s) |
|---|---|---|---|---|
| [`single-objective_analytic/`](single-objective_analytic/README.md) | Single-objective analytic approach (baseline) | Predefined analytic mapping per attribute + fusion operator | Same 30 dims as the multi-objective analytic approach | **F1 only** — shows the *confident-wrong* problem when uncertainty quality is not optimized |
| [`multi-objective_analytic/`](multi-objective_analytic/README.md) | Multi-objective analytic approach | Predefined analytic mapping per attribute + fusion operator (no training) | 30 dims: per-attribute thresholds/shape parameters + fusion operator | Pareto front + knee-point consensus (5 objectives) |
| [`multi-objective_bspline/`](multi-objective_bspline/README.md) | Multi-objective B-Spline approach | Learned per-attribute B-Spline mapping, trained with Adam | 4 training-loss hyperparameters (the coefficients are learned) | Pareto front + knee-point consensus (5 objectives) |
| [`multi-objective_mlp/`](multi-objective_mlp/README.md) | Multi-objective multilayer perceptron (MLP) approach | Learned joint mapping of all 6 attributes (MLP), trained with Adam | 7 hyperparameters (loss weights + architecture) | Pareto front + knee-point consensus (5 objectives) |

The `rule-based_mds/` and `gds/` directories contain the same four
approaches for the rule-based MDS and the GDS.
