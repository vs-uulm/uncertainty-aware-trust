"""
GNSS data loader for the evidential MLP sweep.

Reads the three split CSVs written by the upstream one-class GNSS spoofing
detector (detection-systems/gds/detector_oneclass.py):

    <data-dir>/oneclass_sl_train.csv
    <data-dir>/oneclass_sl_val.csv
    <data-dir>/oneclass_sl_test.csv

load_split(split) returns (X, y, attack, density, scenario):
    X        (N, 4)  float32   the four grouped Mahalanobis distances
                               (chi2-CDF normalized to [0, 1] if NORMALIZE_CHI2)
    y        (N, 1)  int8      1 = benign, 0 = attacker   <-- INVERTED from the
                               CSV's label (1 = spoofed), to match the
                               convention the SL code assumes.
    attack   (N,)    str       TEXBAT scenario name (used as macro group)
    density  (N,)    str       "na" (no density concept in GNSS)
    scenario (N,)    str       TEXBAT scenario name

FEATURE_NAMES must match the column order in X.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

# SL input features = the four grouped Mahalanobis distances.
FEATURE_NAMES = ["d_power", "d_shape", "d_track", "d_dyn"]

# Degrees of freedom per group = number of raw features in that group (see
# GROUPS in detector_oneclass.py). Used to map each squared distance to a
# calibrated probability in [0, 1] via the chi-square CDF. Small p -> close to
# clean (benign); large p -> anomalous (attack).
_GROUP_DF = {"d_power": 2, "d_shape": 3, "d_track": 3, "d_dyn": 1}

# Set False to feed the raw distances instead (the MLP standardizes its
# inputs with the Train mean/std either way).
NORMALIZE_CHI2 = True

SPLITS = ["Train", "Validation", "Test"]
_CSV = {
    "Train":      "oneclass_sl_train.csv",
    "Validation": "oneclass_sl_val.csv",
    "Test":       "oneclass_sl_test.csv",
}

# Default location of the CSVs, relative to the working directory.
# Override via set_data_dir() or the --data-dir CLI arg of sweep_optuna_mlp.py.
DATA_DIR_DEFAULT = "../data"
_DATA_DIR = Path(DATA_DIR_DEFAULT)


def set_data_dir(path):
    global _DATA_DIR
    _DATA_DIR = Path(path)


def _base(data_dir):
    return Path(data_dir) if data_dir else _DATA_DIR


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
    missing = [c for c in FEATURE_NAMES + ["label", "scenario"] if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    if NORMALIZE_CHI2:
        # Map each squared Mahalanobis distance to [0, 1] via its chi2 CDF
        for j, feat in enumerate(FEATURE_NAMES):
            X[:, j] = chi2.cdf(X[:, j], df=_GROUP_DF[feat]).astype(np.float32)

    # CSV: label 1 = spoofed, 0 = clean.  Internal: y 1 = benign, 0 = attacker.
    spoofed = df["label"].to_numpy().astype(np.int8)
    y = (1 - spoofed).astype(np.int8).reshape(-1, 1)

    scenario = df["scenario"].astype(str).to_numpy()
    # Macro-metric grouping: use the SCENARIO name for every row (both clean
    # and spoofed), NOT a separate "benign" bucket. The macro metrics average
    # per group and need BOTH classes present in each group; a standalone
    # "benign" group would leave every attack group single-class.
    attack = scenario.copy().astype(str)
    density = np.full(len(df), "na", dtype=object).astype(str)

    return X, y, attack, density, scenario


def ensure_cache(*args, verbose=True, **kwargs):
    """Checks that the CSVs exist (they are produced by the detector, not built here)."""
    if not data_exists():
        raise FileNotFoundError(
            f"GNSS split CSVs missing in {_DATA_DIR}/ "
            f"(expected {', '.join(_CSV.values())}).")
    if verbose:
        print(f"[Data] OK: {_DATA_DIR}/")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print a summary of the GNSS split CSVs.")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT,
                        help="Directory containing oneclass_sl_{train,val,test}.csv")
    args = parser.parse_args()
    set_data_dir(args.data_dir)
    for sp in SPLITS:
        try:
            X, y, attack, density, scenario = load_split(sp)
            n_benign = int((y == 1).sum())
            n_attack = int((y == 0).sum())
            print(f"{sp:11s}: X={X.shape} y={y.shape}  "
                  f"benign={n_benign} attacker={n_attack}  "
                  f"scenarios={sorted(set(scenario))}")
        except FileNotFoundError as e:
            print(f"{sp:11s}: {e}")
