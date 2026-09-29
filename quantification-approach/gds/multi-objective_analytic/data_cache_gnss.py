"""
GNSS data loader for the closed-form SL sweep.

Reads the three split CSVs written by the upstream one-class GNSS spoofing
detector (detection-systems/gds/detector_oneclass.py):

    <data-dir>/oneclass_sl_train.csv
    <data-dir>/oneclass_sl_val.csv
    <data-dir>/oneclass_sl_test.csv

load_split(split) returns (X, y, attack, density, scenario):
    X        (N, 4)  float32   the four grouped Mahalanobis distances,
                               normalized to [0, 1) (see NORMALIZE)
    y        (N, 1)  int8      1 = benign, 0 = attacker   <-- INVERTED from the
                               CSV's label (1 = spoofed), to match the
                               convention the SL code assumes.
    attack   (N,)    str       TEXBAT scenario name (used as macro group)
    density  (N,)    str       "na" (no density concept in GNSS)
    scenario (N,)    str       TEXBAT scenario name

FEATURE_ORDER must match the column order in X and sl_model.Attribute.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# SL input features = the four grouped Mahalanobis distances.
FEATURE_ORDER = ["d_power", "d_shape", "d_track", "d_dyn"]

# Mapping of the squared distances: "tanh" = d / (d + scale) per feature
# (monotone, no saturation plateau), "none" = raw distances.
NORMALIZE = "tanh"
_TANH_SCALE = {"d_power": 50.0, "d_shape": 5.0, "d_track": 5.0, "d_dyn": 2.0}

SPLITS = ["Train", "Validation", "Test"]
_CSV = {
    "Train":      "oneclass_sl_train.csv",
    "Validation": "oneclass_sl_val.csv",
    "Test":       "oneclass_sl_test.csv",
}

# Default location of the CSVs, relative to the working directory.
DATA_DIR_DEFAULT = "../data"


def _base(data_dir):
    return Path(data_dir) if data_dir else Path(DATA_DIR_DEFAULT)


def data_exists(data_dir=None):
    """True if all three GNSS split CSVs are present in data_dir."""
    base = _base(data_dir)
    return all((base / _CSV[s]).exists() for s in SPLITS)


def load_split(split, data_dir=None):
    """Return (X, y, attack, density, scenario) for one split."""
    if split not in _CSV:
        raise KeyError(f"unknown split '{split}', expected one of {SPLITS}")
    path = _base(data_dir) / _CSV[split]
    if not path.exists():
        raise FileNotFoundError(
            f"GNSS split CSV missing: {path}\n"
            f"  Generate it with detection-systems/gds/detector_oneclass.py first.")

    df = pd.read_csv(path)
    missing = [c for c in FEATURE_ORDER + ["label", "scenario"] if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    X = df[FEATURE_ORDER].to_numpy(dtype=np.float32)
    if NORMALIZE == "tanh":
        # d / (d + scale): monotone, maps [0, inf) -> [0, 1), no saturation plateau
        for j, feat in enumerate(FEATURE_ORDER):
            s = _TANH_SCALE[feat]
            X[:, j] = (X[:, j] / (X[:, j] + s)).astype(np.float32)
    # NORMALIZE == "none": raw distances passed through

    # CSV: label 1 = spoofed, 0 = clean.  Internal: y 1 = benign, 0 = attacker.
    spoofed = df["label"].to_numpy().astype(np.int8)
    y = (1 - spoofed).astype(np.int8).reshape(-1, 1)

    scenario = df["scenario"].astype(str).to_numpy()
    # Macro-metric grouping: the scenario name of every row (clean and
    # spoofed), so each group contains both classes.
    attack = scenario.copy().astype(str)
    density = np.full(len(df), "na", dtype=object).astype(str)

    return X, y, attack, density, scenario


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print a summary of the GNSS split CSVs.")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT,
                        help="Directory containing oneclass_sl_{train,val,test}.csv")
    args = parser.parse_args()
    for sp in SPLITS:
        try:
            X, y, attack, density, scenario = load_split(sp, data_dir=args.data_dir)
            n_benign = int((y == 1).sum())
            n_attack = int((y == 0).sum())
            print(f"{sp:11s}: X={X.shape} y={y.shape}  "
                  f"benign={n_benign} attacker={n_attack}  "
                  f"scenarios={sorted(set(scenario))}")
        except FileNotFoundError as e:
            print(f"{sp:11s}: {e}")
