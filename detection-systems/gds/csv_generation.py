"""
Convert GNSS-SDR tracking dumps into the per-channel CSVs read by detector.py.

With `Tracking_1C.dump=true`, GNSS-SDR writes one binary dump per tracking
channel (`trk_ch_<N>.dat`) and, when the receiver stops, the same data as a
MATLAB file (`trk_ch_<N>.mat`, v7.3/HDF5). This script reads the `.mat`
files and writes one CSV per channel with one column per tracking variable
(`CN0_SNV_dB_Hz`, `PRN`, `PRN_start_sample_count`, `abs_E`, `abs_P`, ...).
The auxiliary fields `aux1`/`aux2` are dropped.

Expected input layout (one GNSS-SDR run per TEXBAT scenario):

    <input-dir>/<scenario>/tracking/trk_ch_<N>.mat

Output layout (what detector.py / detector_oneclass.py read):

    <output-dir>/<scenario>/tracking/trk_ch_<N>.csv

Usage:
    python csv_generation.py --input-dir outputs --output-dir csv_outputs
    python csv_generation.py --input-dir outputs --scenarios cleanStatic ds3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

DROP_COLUMNS = ["aux1", "aux2"]


def read_mat(path):
    """Reads a .mat file (classic or v7.3/HDF5) into {name: squeezed array}."""
    try:
        data = scipy.io.loadmat(path)
        return {k: np.asarray(v).squeeze() for k, v in data.items()
                if not k.startswith("__")}
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5 containers
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.asarray(f[k][()]).squeeze() for k in f.keys()}


def mat_to_df(path):
    """
    Converts one tracking .mat file into a DataFrame. Only 1-D variables with
    the most common length (= number of tracking epochs) become columns.
    """
    data = read_mat(path)
    lengths = [v.size for v in data.values()
               if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1]
    if not lengths:
        return pd.DataFrame()
    n_epochs = max(set(lengths), key=lengths.count)
    cols = {k: v for k, v in data.items()
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == n_epochs}
    return pd.DataFrame(cols).drop(columns=DROP_COLUMNS, errors="ignore")


def convert_scenario(scenario_dir, out_dir):
    """Converts all tracking/*.mat files of one scenario. Returns #files written."""
    mat_files = sorted((scenario_dir / "tracking").glob("*.mat"))
    if not mat_files:
        print(f"  [{scenario_dir.name}] no tracking/*.mat files found, skipping")
        return 0
    target = out_dir / scenario_dir.name / "tracking"
    target.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for path in mat_files:
        df = mat_to_df(path)
        if df.empty:
            print(f"  [{scenario_dir.name}] {path.name}: no tracking data, skipping")
            continue
        out = target / (path.stem + ".csv")
        df.to_csv(out, index=False)
        n_written += 1
        print(f"  [{scenario_dir.name}] {path.name} -> {out}  ({len(df):,} rows)")
    return n_written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="outputs",
                        help="GNSS-SDR output root containing <scenario>/tracking/*.mat")
    parser.add_argument("--output-dir", default="csv_outputs",
                        help="Where <scenario>/tracking/trk_ch_<N>.csv are written")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario subfolders to convert (default: all)")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if args.scenarios:
        scenario_dirs = [in_dir / s for s in args.scenarios]
    else:
        scenario_dirs = sorted(p for p in in_dir.iterdir()
                               if p.is_dir() and (p / "tracking").is_dir())
    if not scenario_dirs:
        raise SystemExit(f"No <scenario>/tracking/ folders found in {in_dir}")

    total = 0
    for scenario_dir in scenario_dirs:
        total += convert_scenario(scenario_dir, out_dir)
    print(f"\nWrote {total} CSV files to {out_dir}/")


if __name__ == "__main__":
    main()
