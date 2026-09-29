"""
Cached binary storage for V2X MBD features.

Parses the raw JSON detector outputs ONCE (in parallel) and stores them as
.npz. Later runs read directly from the cache, which is orders of magnitude
faster than parsing all JSON files again.

Expected raw layout:
    <data_root>/<scenario>_<density>_<attack>/<Train|Validation|Test>/**/*.json

Paths can be set via CLI flags or the environment variables
MBD_DATA_ROOT and MBD_CACHE_DIR.

CLI usage:
    python data_cache.py --build       # build the cache (if missing)
    python data_cache.py --info        # show cache status
    python data_cache.py --rebuild     # delete and rebuild the cache
    python data_cache.py --clear       # delete the cache

Programmatic usage:
    from data_cache import ensure_cache, load_split
    ensure_cache(verbose=True)                              # builds if missing
    X, y, attack, density, scenario = load_split("Train")   # from .npz
"""
import argparse
import glob
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT_DEFAULT = os.environ.get("MBD_DATA_ROOT", "../data")
CACHE_DIR_DEFAULT = os.environ.get("MBD_CACHE_DIR", "../data/cache")
NUM_WORKERS_DEFAULT = os.cpu_count() or 1
PATTERN = "*.json"

# Order of the feature keys; read from obj["check"][<key>].
# sudden_appearance_check is intentionally ignored.
FEATURE_CHECK_KEYS = [
    "range_plausibility_check",
    "position_plausibility_check",
    "speed_plausibility_check",
    "position_consistency_check",
    "speed_consistency_check",
    "position_speed_consistency_check",
    "position_heading_consistency_check",
    "intersection_check",
]
FEATURE_ORDER = [
    "range_plaus", "pos_plaus", "speed_plaus", "pos_cons",
    "speed_cons", "pos_speed_cons", "pos_head_cons", "intersection",
]
FEATURE_NAMES = list(FEATURE_ORDER)

# Detector output -> implausibility (x = 1 - plausibility). Values that are -1,
# missing or non-finite mean "detector not applicable" and are encoded with
# this sentinel.
NA_SENTINEL = -1.0

SPLITS = ["Train", "Validation", "Test"]

# Bump on any change to FEATURE_CHECK_KEYS or the label logic -> automatic rebuild
SCHEMA_VERSION = 2


# ============================================================================
# Parser (runs in worker processes)
# ============================================================================

def _parse_dataset_folder_name(folder_name):
    parts = folder_name.split("_", 2)
    if len(parts) == 3:
        return tuple(parts)  # scenario, density, attack
    return "unknown", "unknown", folder_name


def _parse_file_worker(args):
    """Parse one JSON file and return (X, y, attack, density, scenario) lists."""
    file_path, root = args
    root_path = Path(root)
    file_path_p = Path(file_path)
    rel_parts = file_path_p.relative_to(root_path).parts
    dataset_folder = rel_parts[0]
    scenario, density, attack = _parse_dataset_folder_name(dataset_folder)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            s = f.read().strip()
    except OSError:
        return [], [], [], [], []
    if not s:
        return [], [], [], [], []
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        return [], [], [], [], []
    if not isinstance(arr, list):
        arr = [arr]

    X, y, attacks, densities, scenarios = [], [], [], [], []
    for obj in arr:
        check = obj.get("check")
        if not isinstance(check, dict):
            check = {}
        row = []
        for k in FEATURE_CHECK_KEYS:
            try:
                v = float(check.get(k, -1.0))
            except (ValueError, TypeError):
                v = NA_SENTINEL
            if v < 0.0 or not np.isfinite(v):
                v = NA_SENTINEL          # detector not applicable
            else:
                v = 1.0 - v              # plausibility -> implausibility
            row.append(v)
        X.append(row)
        try:
            label = int(obj.get("attacker"))
        except (ValueError, TypeError):
            label = 1
        # Convention: 1 = benign (attacker == 0), 0 = attacker (attacker != 0)
        y.append(1 if label == 0 else 0)
        attacks.append(attack)
        densities.append(density)
        scenarios.append(scenario)

    return X, y, attacks, densities, scenarios


def _gather_files(root, split):
    pattern = os.path.join(root, "*", split, "**", PATTERN)
    return sorted(glob.glob(pattern, recursive=True))


# ============================================================================
# Cache paths
# ============================================================================

def split_filename(cache_dir, split):
    return os.path.join(cache_dir, f"features_{split.lower()}.npz")


def metadata_filename(cache_dir):
    return os.path.join(cache_dir, "metadata.json")


def cache_exists(cache_dir=CACHE_DIR_DEFAULT):
    """True if all three splits and the metadata exist and the schema matches."""
    meta_path = metadata_filename(cache_dir)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("schema_version") != SCHEMA_VERSION:
        return False
    for split in SPLITS:
        if not os.path.exists(split_filename(cache_dir, split)):
            return False
    return True


# ============================================================================
# Build + Load
# ============================================================================

def build_cache(data_root=DATA_ROOT_DEFAULT,
                cache_dir=CACHE_DIR_DEFAULT,
                num_workers=NUM_WORKERS_DEFAULT,
                verbose=True):
    """Parse all JSON files and write one .npz per split."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "data_root":      data_root,
        "feature_names":  FEATURE_NAMES,
        "feature_check_keys": FEATURE_CHECK_KEYS,
        "splits": {},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for split in SPLITS:
        if verbose:
            print(f"\n[Cache build] {split} ...", flush=True)
        files = _gather_files(data_root, split)
        if verbose:
            print(f"  {len(files)} JSON files found", flush=True)

        if not files:
            print(f"  [WARN] {split}: no files in {data_root}/*/{split}/**/")
            meta["splits"][split] = {
                "n_samples": 0, "n_files": 0, "load_time_sec": 0.0, "path": None,
            }
            continue

        tasks = [(f, data_root) for f in files]
        t0 = time.time()
        X, y, attacks, densities, scenarios = [], [], [], [], []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for X_c, y_c, a_c, d_c, s_c in ex.map(_parse_file_worker, tasks, chunksize=8):
                if X_c:
                    X.extend(X_c); y.extend(y_c)
                    attacks.extend(a_c); densities.extend(d_c); scenarios.extend(s_c)
        dt = time.time() - t0

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int8).reshape(-1, 1)
        attacks_arr = np.asarray(attacks)
        densities_arr = np.asarray(densities)
        scenarios_arr = np.asarray(scenarios)

        out_path = split_filename(cache_dir, split)
        np.savez_compressed(
            out_path,
            X=X_arr, y=y_arr,
            attack=attacks_arr, density=densities_arr, scenario=scenarios_arr,
        )
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        meta["splits"][split] = {
            "n_samples":     int(X_arr.shape[0]),
            "n_files":       len(files),
            "load_time_sec": round(dt, 2),
            "size_mb":       round(size_mb, 2),
            "path":          out_path,
        }
        if verbose:
            print(f"  → {X_arr.shape[0]} samples in {dt:.1f}s → {out_path} ({size_mb:.1f} MB)")

    with open(metadata_filename(cache_dir), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        total = sum(s.get("size_mb", 0) for s in meta["splits"].values())
        print(f"\n[Cache] done — total size: {total:.1f} MB")
        print(f"[Cache] directory: {cache_dir}/")


def load_split(split, cache_dir=CACHE_DIR_DEFAULT):
    """
    Load one split from the cache.

    Returns (X, y, attack, density, scenario) as np.ndarrays:
      - X:        (N, 8)  float32   (implausibility per detector, -1 = N/A)
      - y:        (N, 1)  int8     (1 = benign, 0 = attacker)
      - attack:   (N,)    str
      - density:  (N,)    str
      - scenario: (N,)    str
    """
    p = split_filename(cache_dir, split)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Cache missing: {p}\n"
            f"  Run 'python data_cache.py --build' to build it."
        )
    with np.load(p, allow_pickle=False) as z:
        return (
            z["X"].copy(),
            z["y"].copy(),
            z["attack"].copy(),
            z["density"].copy(),
            z["scenario"].copy(),
        )


def ensure_cache(data_root=DATA_ROOT_DEFAULT,
                 cache_dir=CACHE_DIR_DEFAULT,
                 num_workers=NUM_WORKERS_DEFAULT,
                 verbose=True):
    """Build the cache if it is missing; otherwise do nothing."""
    if cache_exists(cache_dir):
        if verbose:
            with open(metadata_filename(cache_dir)) as f:
                meta = json.load(f)
            total = sum(s.get("n_samples", 0) for s in meta["splits"].values())
            print(f"[Cache] OK — {total} samples in total in {cache_dir}/")
        return
    if verbose:
        print(f"[Cache] missing — building from {data_root} ...")
    build_cache(data_root=data_root, cache_dir=cache_dir,
                num_workers=num_workers, verbose=verbose)


def print_info(cache_dir=CACHE_DIR_DEFAULT):
    meta_path = metadata_filename(cache_dir)
    if not os.path.exists(meta_path):
        print(f"Cache directory: {cache_dir}/  — empty / missing")
        return
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Cache directory:   {cache_dir}/")
    print(f"Schema version:    {meta['schema_version']}")
    print(f"Data root:         {meta['data_root']}")
    print(f"Built at:          {meta.get('built_at', 'unknown')}")
    print(f"Features:          {', '.join(meta['feature_names'])}")
    print()
    print(f"  {'Split':12s}  {'Samples':>12s}  {'Files':>8s}  {'Size':>8s}  {'LoadTime':>10s}")
    for split, info in meta.get("splits", {}).items():
        sz = info.get("size_mb", 0)
        print(f"  {split:12s}  {info['n_samples']:>12d}  {info['n_files']:>8d}  "
              f"{sz:>6.1f} MB  {info['load_time_sec']:>9.1f}s")


def clear_cache(cache_dir=CACHE_DIR_DEFAULT):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[Cache] deleted: {cache_dir}/")
    else:
        print(f"[Cache] does not exist: {cache_dir}/")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--build", action="store_true",
                   help="Build the cache (skipped if it exists)")
    p.add_argument("--rebuild", action="store_true",
                   help="Force a rebuild (deletes the cache first)")
    p.add_argument("--info", action="store_true", help="Show cache status")
    p.add_argument("--clear", action="store_true", help="Delete the cache")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                   help=f"Raw JSON root (default: $MBD_DATA_ROOT or {DATA_ROOT_DEFAULT})")
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT,
                   help=f"Cache directory (default: $MBD_CACHE_DIR or {CACHE_DIR_DEFAULT})")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS_DEFAULT,
                   help="Parallel parser processes (default: CPU count)")
    args = p.parse_args()

    if args.info:
        print_info(cache_dir=args.cache_dir)
        return
    if args.clear:
        clear_cache(cache_dir=args.cache_dir)
        return
    if args.rebuild:
        clear_cache(cache_dir=args.cache_dir)
        build_cache(data_root=args.data_root, cache_dir=args.cache_dir,
                    num_workers=args.num_workers)
        return
    if args.build:
        if cache_exists(args.cache_dir):
            print("[Cache] already exists — skipping. Use --rebuild to rebuild.")
            print_info(cache_dir=args.cache_dir)
            return
        build_cache(data_root=args.data_root, cache_dir=args.cache_dir,
                    num_workers=args.num_workers)
        return

    # Default without flags: build only if missing
    ensure_cache(data_root=args.data_root, cache_dir=args.cache_dir,
                 num_workers=args.num_workers)


if __name__ == "__main__":
    main()
