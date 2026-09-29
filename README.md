# Uncertainty-Aware Trust Quantification

Uncertainty-aware trust quantification in Subjective Logic (SL) for
heterogeneous vehicular detection systems.

This repository contains the artifacts (code, configurations, data splits
and detector outputs) for the paper:

> **Uncertainty-Aware Trust Quantification for Heterogeneous Vehicular
> Detection Systems**
>
> Artur Hermann, Dennis Eisermann, Frank Kargl (Ulm University)

The paper is currently under review for IEEE VTC 2027-Spring.

## Overview

Vehicle-to-Everything (V2X) communication improves road safety but is
vulnerable to data manipulation. Trust assessment frameworks therefore fuse
the outputs of several detection systems, for example Misbehavior Detection
Systems (MDSs) for V2X messages and GNSS Spoofing Detection Systems (GDSs).
In Subjective Logic, each detector output is quantified into an **opinion**
`(b, d, u)`:

- `b`: belief that the data are benign
- `d`: disbelief, i.e. belief that the data are manipulated
- `u`: uncertainty, with `b + d + u = 1`

The uncertainty weights each opinion during fusion. Existing quantification
approaches are optimized for detection performance alone. As a result, they
can assign *lower* uncertainty to incorrect than to correct decisions, so
incorrect outputs dominate the fusion. The paper calls this the
**confident-wrong** problem.

This repository implements the full pipeline used in the paper:

1. **Detection systems:** three heterogeneous detection systems produce the
   trust evidence: a rule-based MDS and a prediction-based MDS for V2X
   messages (VeReMi NextGen), and a GDS (GNSS Spoofing Detection System,
   TEXBAT).
2. **Trust quantification:** the trust evidence of each detection system is
   mapped to an SL opinion. Four quantification approaches are compared:
   - the **single-objective analytic approach** (baseline): the existing
     analytic approach, optimized for F1 alone,
   - the **multi-objective analytic approach**: the same predefined mapping,
     optimized for opinion quality as well,
   - the **multi-objective B-Spline approach**: a per-attribute mapping
     learned from data,
   - the **multi-objective multilayer perceptron (MLP) approach**: a mapping
     learned jointly across all attributes.

   The three multi-objective approaches are tuned with a **multi-objective
   optimization** (Optuna NSGA-II) over detection performance (F1) and four
   opinion-quality metrics: AURC, misclassification AUROC, Δu and belief
   correctness.
3. **Fusion:** the opinions of the prediction-based MDS and the GDS are fused with SL fusion
   operators, and the effect of opinion quality on the fused trust decision
   is evaluated.

All quantification approaches use the same objectives, attack groups,
decision rule (`p = b + 0.5·u ≥ 0.5 ⇔ b ≥ d`) and evaluation tables, so the
results compare directly within each detection system.

## Repository structure

The top-level directories build on each other:
`detection-systems/` → `dataset/` → `quantification-approach/` → `fusion-experiment/`.

```
.
├── README.md
├── LICENSE.txt                                # Apache License 2.0
├── NOTICE                                     # copyright and third-party data notice
├── CITATION.cff                               # citation metadata
│
├── dataset/                                   # detector outputs = input of the trust quantification (~3 GB)
│   ├── README.md
│   ├── rule-based_mds/                        # 8 rule-check features per V2X message (Git LFS)
│   │   ├── features_train.npz                 # training split
│   │   ├── features_validation.npz            # validation split
│   │   ├── features_test.npz                  # test split
│   │   └── metadata.json                      # marks the directory as a ready-made feature cache
│   ├── prediction-based_mds/                  # 6 prediction-error features per V2X message (Git LFS)
│   │   ├── features_train.npz                 # training split
│   │   ├── features_validation.npz            # validation split
│   │   ├── features_test.npz                  # test split
│   │   └── metadata.json                      # marks the directory as a ready-made feature cache
│   └── gds/                                   # 4 Mahalanobis distances per GNSS time window
│       ├── oneclass_<scenario>.csv            # per-scenario detector output (cleanStatic, ds3, ds4, ds7, ds8)
│       ├── oneclass_sl_train.csv              # training split
│       ├── oneclass_sl_val.csv                # validation split
│       └── oneclass_sl_test.csv               # test split
│
├── detection-systems/                         # the three detectors that produce dataset/
│   ├── README.md
│   ├── rule-based_mds/                        # CaTch plausibility/consistency checks (VeReMi NextGen)
│   │   ├── main.py, catch_checks.py, mdm_lib.py, data_processing.py, data_structures.py
│   │   └── README.md, requirements.txt
│   ├── prediction-based_mds/                  # context-window MLP, prediction errors as features (VeReMi NextGen)
│   │   ├── generate_feature_output_abs.py, test_train_helper.py
│   │   ├── model/                             # pretrained MLP weights (.pth)
│   │   ├── NN_classes/                        # network definition
│   │   ├── NN_datasets/                       # VeReMi NextGen dataset loaders + feature preparation
│   │   └── README.md, requirements.txt
│   └── gds/                                   # GNSS Spoofing Detection System (GDS), TEXBAT
│       ├── detector.py, detector_oneclass.py
│       └── README.md, requirements.txt
│
├── quantification-approach/                  # SL trust quantification: Optuna sweeps
│   ├── README.md
│   ├── rule-based_mds/                        # sweeps on dataset/rule-based_mds
│   │   ├── multi-objective_analytic/          # multi-objective analytic approach, 5 objectives (NSGA-II)
│   │   ├── multi-objective_bspline/           # multi-objective B-Spline approach, 5 objectives
│   │   ├── multi-objective_mlp/               # multi-objective MLP approach, 5 objectives
│   │   └── single-objective_analytic/         # single-objective analytic approach, F1 only (baseline)
│   ├── prediction-based_mds/                  # same four approaches on dataset/prediction-based_mds
│   │   └── ...
│   └── gds/                                   # same four approaches on dataset/gds
│       ├── multi-objective_analytic/
│       │   ├── sweep_optuna.py                # entry point: sweep, knee-point selection, final test
│       │   ├── sl_model.py                    # SL opinion model + fusion operators
│       │   ├── metrics.py                     # AURC, misclassification AUROC, Δu, ECE, Brier, ...
│       │   ├── data_cache_gnss.py             # data loading
│       │   ├── paper_outputs.py               # paper tables 1–5 + boxplots
│       │   └── README.md, requirements.txt
│       ├── multi-objective_bspline/           # sweep_optuna_bspline_gnss.py, data cache, sl_metrics, paper_outputs
│       ├── multi-objective_mlp/               # sweep_optuna_mlp.py, data cache, sl_metrics, paper_outputs
│       └── single-objective_analytic/         # sweep_optuna_f1only.py + shared analytic modules
│
└── fusion-experiment/                         # SL fusion of a V2X MDS and the GDS
    ├── sl_fusion_eval.py                      # fusion operators (CBF, ABF, BCF, CCF, mult) + evaluation
    ├── sl_fusion_results_summary_v5.csv       # result summary (mlp vs. analytic_f1 opinions)
    └── README.md, requirements.txt
```

In total the repository holds 12 quantification sweeps: 3 detection systems
× 4 approaches. Each sweep directory is self-contained, with its own README
covering data, setup, usage, search space and outputs.

| Directory | Contents |
|---|---|
| [`dataset/`](dataset/) | Detector outputs with their train/validation/test splits. The `*.npz` files are tracked with **Git LFS** (`git lfs install` before cloning). |
| [`detection-systems/`](detection-systems/) | Code of the three detection systems that produce `dataset/`. |
| [`quantification-approach/`](quantification-approach/) | Trust quantification: for every detection system, the single-objective analytic approach (baseline) and the multi-objective analytic, B-Spline and MLP approaches. |
| [`fusion-experiment/`](fusion-experiment/) | Fuses the opinions of a V2X MDS and the GDS with several SL fusion operators and evaluates the fused decision. |

## Data sources

The detector outputs in `dataset/` are derived from:

- **VeReMi NextGen**, for the two V2X misbehavior detection systems:
  <https://veremi-dataset.github.io/veremi-nextgen>
- **TEXBAT** (Texas Spoofing Test Battery), for the GNSS spoofing detection
  system: <https://radionavlab.ae.utexas.edu/texbat/>

If you use the data, please also cite these original datasets.

## License and citation

The code is licensed under the Apache License 2.0 (see [`LICENSE.txt`](LICENSE.txt)
and [`NOTICE`](NOTICE)). The data in `dataset/` are derived from VeReMi NextGen
and TEXBAT and are additionally subject to the terms of these datasets.

## Contact

Artur Hermann, Dennis Eisermann, Frank Kargl — Ulm University, Germany
(`{artur.hermann, dennis.eisermann, frank.kargl}@uni-ulm.de`)
