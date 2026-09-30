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

## Attack types

### VeReMi NextGen (rule-based and prediction-based MDS)

The two MDS datasets contain the 15 attack types of VeReMi NextGen [1]. By
default, 20% of the vehicles are attackers, and each attacker manipulates its
own V2X messages. The descriptions and parameter ranges follow the
[attack generator documentation](https://github.com/VeReMi-dataset/VeReMi-NextGen/tree/main/Documentation/Post-Processing/Attack%20Generator)
of VeReMi NextGen. Some attacks are labeled as malicious only when the
manipulation has a practical effect (e.g. a reversed heading of a moving
vehicle); the documentation lists the exact conditions.

| Category | Attack (`attack` value) | Description |
|---|---|---|
| Time | `timeDelayAttack` | Delays the message timestamps (send and receive time) by 2–4 s |
| Position | `constantPositionOffset` | Adds a constant offset of ±20–70 m (in X and Y), chosen once per attacker, to the reported position |
| Position | `randomPositionOffset` | Adds a new random offset of ±20–70 m (in X and Y) to the reported position in each message |
| Position | `positionMirroring` | Mirrors the reported position to the opposite side of the road |
| Speed | `constantSpeedOffset` | Adds a constant offset of ±1–7 m/s to the reported speed |
| Speed | `randomSpeedOffset` | Adds a new random offset of ±1–7 m/s to the reported speed in each message |
| Speed | `zeroSpeedReport` | Reports a speed of 0 while the vehicle is moving |
| Speed | `suddenConstantSpeed` | Freezes the reported speed at some point while the actual speed keeps changing |
| Heading | `reversedHeading` | Reports the heading rotated by 180° (reversed driving direction) |
| Acceleration | `feignedBraking` | Reports braking by negating the acceleration and multiplying it by 2–4 |
| Acceleration | `accelerationMultiplication` | Multiplies the reported acceleration by 2–4 |
| Multi-parameter | `suddenStop` | Reports a sudden stop (frozen position, zero speed) while the vehicle keeps moving |
| Multi-parameter | `dosAttack` | Denial of Service: sends 2–4 duplicates of each message to flood the network |
| Multi-parameter | `trafficCongestionSybil` | Sybil attack: creates 4–6 phantom vehicles around the attacker to fake a traffic congestion |
| Multi-parameter | `dataReplay` | Replays messages of a nearby vehicle (within 400 m) as the attacker's own |

`timeDelayAttack` and `trafficCongestionSybil` are not used in the
quantification sweeps (see above).

### TEXBAT (GDS)

The GDS dataset uses the clean static recording `cleanStatic` and four static
spoofing scenarios of TEXBAT [2, 3]. In all four, a spoofer takes over the
tracking loops of the receiver with power-matched signals, i.e. without an
obvious rise in received power.

| Scenario | Description |
|---|---|
| `ds3` | Matched-power time push: the spoofer gradually shifts the receiver's time solution |
| `ds4` | Matched-power position push: the spoofer gradually shifts the receiver's position solution |
| `ds7` | Matched-power time push like `ds3`, but more subtle because the spoofing signals are carrier-phase aligned with the authentic signals |
| `ds8` | Like `ds7`, but the spoofer additionally estimates the navigation data bits in real time instead of knowing them in advance (zero-delay security code estimation and replay, SCER) |

## References

[1] A. Hermann, J.-N. Remmers, D. Eisermann, B. Erb, and F. Kargl, "VeReMi
NextGen: A Dataset for Evaluating Misbehavior Detection Systems in VANETs,"
in *Proc. 2026 IEEE Vehicular Networking Conference (VNC)*, Montreal, Canada,
2026, pp. 1–8, doi: [10.1109/VNC69225.2026.11629123](https://ieeexplore.ieee.org/document/11629123).
Dataset: <https://veremi-dataset.github.io/veremi-nextgen>

[2] T. E. Humphreys, J. A. Bhatti, D. P. Shepard, and K. D. Wesson, "The Texas
Spoofing Test Battery: Toward a Standard for Evaluating GPS Signal
Authentication Techniques," in *Proc. ION GNSS*, Nashville, TN, USA, 2012.
<https://radionavlab.ae.utexas.edu/texbat/>

[3] A. Lemmenes, P. Corbell, and S. Gunawardena, "Detailed Analysis of the
TEXBAT Datasets Using a High Fidelity Software GPS Receiver," in *Proc. ION
GNSS+*, Portland, OR, USA, 2016.
