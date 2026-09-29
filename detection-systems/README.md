# detection-systems

The three detection systems whose outputs are the trust evidence for the
trust quantification approaches in `quantification-approach/`. Each
subdirectory is self-contained (own `README.md` + `requirements.txt`).

## Overview

| Directory | Detection system | Domain | Data source | Trust evidence |
|---|---|---|---|---|
| [`rule-based_mds/`](rule-based_mds/) | Rule-based Misbehavior Detection System (MDS), after Kamel et al. (CaTch) | V2X | VeReMi NextGen | 8 plausibility-check scores per message |
| [`prediction-based_mds/`](prediction-based_mds/) | Prediction-based Misbehavior Detection System (MDS), after Hsu et al. | V2X | VeReMi NextGen | 6 prediction errors per message: a pretrained MLP predicts each message's attributes from the vehicle's preceding messages |
| [`gds/`](gds/) | GNSS Spoofing Detection System (GDS), features of Iqbal et al. | GNSS | TEXBAT | 4 anomaly scores (squared Mahalanobis distances of four feature groups) per 0.5 s window |

## `rule-based_mds/` — Rule-Based MDS

Rule-based Misbehavior Detection System (MDS) for V2X/VANET message data.
Applies plausibility and consistency checks (range, position, speed, heading,
intersection, sudden appearance) to sender/receiver vehicle messages and
labels each message as benign or attacker. Implements the detection approach
from Kamel et al., "CaTch: A Confidence Range Tolerant Misbehavior Detection
Approach" (IEEE WCNC 2019). Data: **VeReMi NextGen**
(<https://veremi-dataset.github.io/veremi-nextgen>).

## `prediction-based_mds/` — Prediction-Based MDS

Prediction-based Misbehavior Detection System (MDS) for V2X messages. A
pretrained context-window MLP predicts a vehicle's next expected message
(position, speed, acceleration, heading, distance to road edge, receive
time) from its recent message history; the absolute error between
prediction and the actually received message becomes a per-message anomaly
feature (six features in total). Data: **VeReMi NextGen**
(<https://veremi-dataset.github.io/veremi-nextgen>).

## `gds/` — GDS

GNSS Spoofing Detection System (GDS). Builds per-window features (C/N0, code/carrier
lock, early-minus-late/early-plus-late correlator ratios, code error,
Doppler) from GNSS receiver tracking logs and fits a strict one-class
Mahalanobis detector (fit only on clean data, no attack labels used for
fitting or threshold selection) against real spoofing attacks. Data:
**TEXBAT** (Texas Spoofing Test Battery,
<https://radionavlab.ae.utexas.edu/texbat/>).
