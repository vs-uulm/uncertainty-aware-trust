# dataset

Trust evidence produced by the three detection systems in `detection-systems/`
(see its README), with train/validation/test splits. These files are the input
of the trust quantification approaches in `quantification-approach/`. The
`*.npz` files are tracked with Git LFS.

| Directory | Detection system | Files | Contents |
|---|---|---|---|
| [`rule-based_mds/`](rule-based_mds/) | Rule-based Misbehavior Detection System (MDS), `detection-systems/rule-based_mds` | `features_{train,validation,test}.npz`, `metadata.json` | `X` (N×8 float32 plausibility-check scores, as implausibility `1 − plausibility`, `-1` = check not applicable), `y` (N×1 int8 label, `1` = benign), `attack`/`density`/`scenario` (string metadata) |
| [`prediction-based_mds/`](prediction-based_mds/) | Prediction-based Misbehavior Detection System (MDS), `detection-systems/prediction-based_mds` | `features_{train,validation,test}.npz`, `metadata.json` | `X` (N×6 float32 prediction errors: position, speed, acceleration, road-edge distance, receive time, heading), `y`, `attack`/`density`/`scenario` |
| [`gds/`](gds/) | GNSS Spoofing Detection System (GDS), `detection-systems/gds` | `oneclass_<scenario>.csv`, `oneclass_sl_{train,val,test}.csv` | Per-scenario Mahalanobis distances and decisions of the GDS, plus the chronological train/val/test split used by the quantification approaches |

The two MDS directories are ready-made feature caches for the quantification
sweeps (`metadata.json` marks them as valid caches); see "Using the shipped
dataset" in the sweep READMEs. The `.npz` files contain all VeReMi NextGen
attack types; the sweeps restrict them to 13 attack types.
`timeDelayAttack` and `trafficCongestionSybil` are excluded because local
MDSs cannot detect them.
