"""
CSV cache adapter for the GNSS B-spline Subjective-Logic model.

Expected input columns
----------------------
Required:
    scenario, split, label, d_power, d_shape, d_track, d_dyn
Optional metadata (not used as model input):
    prn, t_s, captured, d_total, p_spoof, pred

Only these four columns are used as B-spline inputs:
    d_power, d_shape, d_track, d_dyn

Label conversion
----------------
Input CSV:
    label = 0 -> clean / benign
    label = 1 -> spoofed / attacker
Internal convention used by sweep_optuna_bspline_gnss.py:
    y = 1 -> benign
    y = 0 -> spoofed / attacker

The adapter accepts either:
  1. one combined CSV containing a `split` column, or
  2. a directory containing oneclass_sl_train.csv, oneclass_sl_val.csv,
     and oneclass_sl_test.csv.

Usage
-----
    python data_cache_gnss_bspline.py --csv ../data --build
    python data_cache_gnss_bspline.py --csv ../data --info
    python data_cache_gnss_bspline.py --csv oneclass_sl_all.csv --rebuild

Optional feature transformations:
    raw       : use distances unchanged (default)
    log1p     : log(1+x), useful for very large Mahalanobis distances
    chi2_cdf  : map squared Mahalanobis distances to [0,1]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


FEATURE_NAMES = ["d_power", "d_shape", "d_track", "d_dyn"]
REQUIRED_COLUMNS = ["scenario", "label", *FEATURE_NAMES]
SPLITS = ["Train", "Validation", "Test"]

CACHE_DIR_DEFAULT = "cache"
CSV_PATH_DEFAULT = "../data"
SCHEMA_VERSION = 3

# Number of raw features in each Mahalanobis group. Only used for chi2_cdf.
_GROUP_DF = {"d_power": 2, "d_shape": 3, "d_track": 3, "d_dyn": 1}

_CSV_PATH = Path(CSV_PATH_DEFAULT)
_CACHE_DIR = Path(CACHE_DIR_DEFAULT)
_FEATURE_TRANSFORM = "raw"

_SPLIT_ALIASES = {
    "train": "Train",
    "training": "Train",
    "val": "Validation",
    "valid": "Validation",
    "validation": "Validation",
    "test": "Test",
    "testing": "Test",
}

_SEPARATE_FILES = {
    "Train": "oneclass_sl_train.csv",
    "Validation": "oneclass_sl_val.csv",
    "Test": "oneclass_sl_test.csv",
}


def configure(csv_path: str | os.PathLike | None = None,
              cache_dir: str | os.PathLike | None = None,
              feature_transform: str | None = None) -> None:
    """Configure source CSV/directory, cache directory and feature transform."""
    global _CSV_PATH, _CACHE_DIR, _FEATURE_TRANSFORM, CACHE_DIR_DEFAULT
    if csv_path is not None:
        _CSV_PATH = Path(csv_path)
    if cache_dir is not None:
        _CACHE_DIR = Path(cache_dir)
        CACHE_DIR_DEFAULT = str(_CACHE_DIR)
    if feature_transform is not None:
        feature_transform = feature_transform.lower()
        if feature_transform not in {"raw", "log1p", "chi2_cdf"}:
            raise ValueError(
                "feature_transform must be one of: raw, log1p, chi2_cdf"
            )
        _FEATURE_TRANSFORM = feature_transform


def split_filename(cache_dir: str | os.PathLike, split: str) -> str:
    return str(Path(cache_dir) / f"features_{split.lower()}.npz")


def metadata_filename(cache_dir: str | os.PathLike) -> str:
    return str(Path(cache_dir) / "metadata.json")


def _source_signature(path: Path) -> Dict[str, object]:
    if path.is_file():
        st = path.stat()
        return {
            "kind": "combined_csv",
            "path": str(path.resolve()),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    if path.is_dir():
        files = []
        for split, name in _SEPARATE_FILES.items():
            p = path / name
            if p.exists():
                st = p.stat()
                files.append({
                    "split": split,
                    "path": str(p.resolve()),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                })
        return {"kind": "split_csv_directory", "path": str(path.resolve()), "files": files}
    return {"kind": "missing", "path": str(path)}


def _normalize_split(value: object) -> str | None:
    key = str(value).strip().lower()
    return _SPLIT_ALIASES.get(key)


def _read_source(csv_path: Path) -> pd.DataFrame:
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            raise KeyError(
                f"{csv_path} has no 'split' column. Either add it or pass a directory "
                "with oneclass_sl_train.csv, oneclass_sl_val.csv and oneclass_sl_test.csv."
            )
        return df

    if csv_path.is_dir():
        frames = []
        missing = []
        for split, name in _SEPARATE_FILES.items():
            p = csv_path / name
            if not p.exists():
                missing.append(str(p))
                continue
            part = pd.read_csv(p)
            part = part.copy()
            part["split"] = split
            frames.append(part)
        if missing:
            raise FileNotFoundError(
                "Missing GNSS split CSV files:\n  " + "\n  ".join(missing)
            )
        return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError(f"GNSS CSV source not found: {csv_path}")


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS + ["split"] if c not in df.columns]
    if missing:
        raise KeyError(f"GNSS CSV is missing required columns: {missing}")

    out = df.copy()
    out["split"] = out["split"].map(_normalize_split)
    if out["split"].isna().any():
        bad = sorted(set(df.loc[out["split"].isna(), "split"].astype(str)))
        raise ValueError(
            f"Unknown split values: {bad}. Supported aliases: "
            f"{sorted(_SPLIT_ALIASES)}"
        )

    out["scenario"] = out["scenario"].astype(str).str.strip()
    if (out["scenario"] == "").any():
        raise ValueError("Column 'scenario' contains empty values.")

    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    if out["label"].isna().any():
        raise ValueError("Column 'label' contains non-numeric values.")
    labels = set(out["label"].astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected label values 0/1, found: {sorted(labels)}")
    out["label"] = out["label"].astype(np.int8)

    for col in FEATURE_NAMES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    finite_mask = np.isfinite(out[FEATURE_NAMES].to_numpy(dtype=np.float64)).all(axis=1)
    if not finite_mask.all():
        n_drop = int((~finite_mask).sum())
        print(f"[Cache] Drop {n_drop} rows with NaN/inf in GNSS input features.")
        out = out.loc[finite_mask].copy()

    if (out[FEATURE_NAMES] < 0).any().any():
        raise ValueError(
            "Mahalanobis-distance inputs must be non-negative; negative values were found."
        )
    return out


def _transform_features(X: np.ndarray, transform: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if transform == "raw":
        out = X
    elif transform == "log1p":
        out = np.log1p(X)
    elif transform == "chi2_cdf":
        from scipy.stats import chi2
        out = np.empty_like(X)
        for j, feature in enumerate(FEATURE_NAMES):
            out[:, j] = chi2.cdf(X[:, j], df=_GROUP_DF[feature])
    else:
        raise ValueError(f"Unknown feature transform: {transform}")
    return out.astype(np.float32)


def _class_counts(y: np.ndarray) -> Dict[str, int]:
    flat = y.reshape(-1)
    return {
        "benign_internal_y1": int((flat == 1).sum()),
        "spoofed_internal_y0": int((flat == 0).sum()),
    }


def build_cache(csv_path: str | os.PathLike | None = None,
                cache_dir: str | os.PathLike | None = None,
                feature_transform: str | None = None,
                verbose: bool = True) -> None:
    configure(csv_path, cache_dir, feature_transform)
    source = _CSV_PATH
    target = _CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = _validate_and_clean(_read_source(source))

    meta: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": _source_signature(source),
        "feature_names": FEATURE_NAMES,
        "feature_transform": _FEATURE_TRANSFORM,
        "label_input": "0=benign, 1=spoofed",
        "label_internal": "1=benign, 0=spoofed",
        "ignored_columns": ["d_total", "p_spoof", "pred", "captured", "prn", "t_s"],
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {},
    }

    for split in SPLITS:
        part = df.loc[df["split"] == split].copy()
        if part.empty:
            raise ValueError(f"Split '{split}' contains no rows in {source}.")

        X = _transform_features(
            part[FEATURE_NAMES].to_numpy(dtype=np.float64), _FEATURE_TRANSFORM
        )
        # Input label: 0 benign, 1 spoofed. Internal: 1 benign, 0 attacker.
        y = (1 - part["label"].to_numpy(dtype=np.int8)).reshape(-1, 1)
        scenario = np.asarray(part["scenario"].astype(str).tolist(), dtype=str)
        attack = scenario.copy()  # group macro metrics by TEXBAT scenario
        density = np.full(len(part), "na", dtype="<U2")

        out_path = split_filename(target, split)
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            attack=attack,
            density=density,
            scenario=scenario,
        )
        info = {
            "n_samples": int(len(part)),
            "shape": list(X.shape),
            "class_counts": _class_counts(y),
            "scenarios": sorted(set(scenario.tolist())),
            "path": out_path,
            "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 3),
        }
        meta["splits"][split] = info  # type: ignore[index]
        if verbose:
            cc = info["class_counts"]
            print(
                f"[Cache] {split:10s}: N={len(part):7d}, X={X.shape}, "
                f"benign={cc['benign_internal_y1']}, spoofed={cc['spoofed_internal_y0']}, "
                f"scenarios={info['scenarios']}"
            )

    with open(metadata_filename(target), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(
            f"[Cache] Built in {time.time() - t0:.2f}s: {target}/ "
            f"(transform={_FEATURE_TRANSFORM})"
        )


def cache_exists(cache_dir: str | os.PathLike | None = None) -> bool:
    target = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    meta_path = Path(metadata_filename(target))
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False
    if meta.get("feature_names") != FEATURE_NAMES:
        return False
    if meta.get("feature_transform") != _FEATURE_TRANSFORM:
        return False
    if meta.get("source") != _source_signature(_CSV_PATH):
        return False
    return all(Path(split_filename(target, split)).exists() for split in SPLITS)


def ensure_cache(csv_path: str | os.PathLike | None = None,
                 cache_dir: str | os.PathLike | None = None,
                 feature_transform: str | None = None,
                 verbose: bool = True) -> None:
    configure(csv_path, cache_dir, feature_transform)
    if cache_exists(_CACHE_DIR):
        if verbose:
            print(f"[Cache] OK: {_CACHE_DIR}/")
        return
    if verbose:
        print(f"[Cache] Missing/stale; rebuilding from {_CSV_PATH}")
    build_cache(verbose=verbose)


def load_split(split: str,
               cache_dir: str | os.PathLike | None = None) -> Tuple[np.ndarray, ...]:
    if split not in SPLITS:
        raise KeyError(f"Unknown split '{split}', expected one of {SPLITS}")
    target = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    path = Path(split_filename(target, split))
    if not path.exists():
        raise FileNotFoundError(
            f"Cache missing: {path}\nBuild it with data_cache_gnss_bspline.py --build."
        )
    with np.load(path, allow_pickle=False) as z:
        return (
            z["X"].copy(),
            z["y"].copy(),
            z["attack"].copy(),
            z["density"].copy(),
            z["scenario"].copy(),
        )


def scenarios_with_spoofed(y: np.ndarray, scenario: np.ndarray) -> list[str]:
    """Return scenarios containing at least one spoofed sample (internal y=0)."""
    y_flat = np.asarray(y).reshape(-1)
    scenario = np.asarray(scenario).astype(str)
    return sorted({str(s) for s in np.unique(scenario) if np.any((scenario == s) & (y_flat == 0))})


def scenarios_with_both_classes(y: np.ndarray, scenario: np.ndarray) -> list[str]:
    """Return scenarios containing benign and spoofed samples."""
    y_flat = np.asarray(y).reshape(-1)
    scenario = np.asarray(scenario).astype(str)
    result = []
    for s in np.unique(scenario):
        labels = set(y_flat[scenario == s].tolist())
        if labels == {0, 1}:
            result.append(str(s))
    return sorted(result)


def print_info(cache_dir: str | os.PathLike | None = None) -> None:
    target = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    path = Path(metadata_filename(target))
    if not path.exists():
        print(f"No cache metadata at {path}")
        return
    print(path.read_text(encoding="utf-8"))


def clear_cache(cache_dir: str | os.PathLike | None = None) -> None:
    target = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    if target.exists():
        shutil.rmtree(target)
        print(f"[Cache] Deleted: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=CSV_PATH_DEFAULT,
                        help="Combined GNSS CSV or directory with three split CSVs")
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--feature-transform", choices=["raw", "log1p", "chi2_cdf"],
                        default="raw")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    configure(args.csv, args.cache_dir, args.feature_transform)
    if args.clear:
        clear_cache()
        return
    if args.rebuild:
        clear_cache()
        build_cache(verbose=True)
        return
    if args.info:
        if not cache_exists():
            print("Cache is missing or stale.")
        print_info()
        return
    if args.build:
        ensure_cache(verbose=True)
        return
    ensure_cache(verbose=True)


if __name__ == "__main__":
    main()
