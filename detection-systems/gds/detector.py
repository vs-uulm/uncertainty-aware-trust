"""
Shared data loading and feature windowing for TEXBAT tracking logs.

Used by detector_oneclass.py to build the per-window feature table it fits
its one-class model on and evaluates.

INPUT LAYOUT  (one gnss-sdr run per scenario, converted to CSV)
    csv_outputs/cleanStatic/tracking/trk_ch_*.csv
    csv_outputs/ds3/tracking/trk_ch_*.csv
    csv_outputs/ds4/tracking/trk_ch_*.csv
    csv_outputs/ds7/tracking/trk_ch_*.csv
    csv_outputs/ds8/tracking/trk_ch_*.csv
  A channel CSV holds whatever PRN that receiver slot tracked; the PRN is in a
  column, not the filename. Time comes from PRN_start_sample_count / 25e6.

FEATURES  (CONVENTIONAL class: what a standard DLL/PLL receiver exposes).
Artefact-poisoned quantities excluded, following Lemmenes et al.:
    absolute power / Prompt magnitude  -> 8.6 dB scenario fingerprint
    aux1, aux2                         -> aux2 == sample count == time leak
    code-minus-carrier                 -> 0.369 Hz VSG offset in ds1-4
    raw carrier_doppler value          -> identifies the PRN, not the attack
"""

from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd

FS = 25_000_000.0

ONSETS = {
    "cleanStatic": (None, None),
    "ds3": (114.89, 195.79),
    "ds4": (110.12, 225.22),
    "ds7": (113.00, 139.00),
    "ds8": (113.00, 139.00),
}
CAPTURED = {"ds3": {7}, "ds4": {3, 10, 23}, "ds7": "all", "ds8": "all"}

LABEL_CLEAN, LABEL_GUARD, LABEL_SPOOFED = 0, -1, 1


def load_scenario(csv_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(Path(csv_dir) / "trk_ch_*.csv"))):
        d = pd.read_csv(f)
        if "PRN" not in d or "CN0_SNV_dB_Hz" not in d or len(d) < 100:
            continue
        modal = d["PRN"].mode().iloc[0]
        d = d[d["PRN"] == modal].copy()
        d["t_s"] = d["PRN_start_sample_count"] / FS
        d = d[d["CN0_SNV_dB_Hz"] > 0]
        if len(d) > 100:
            frames.append(d)
    if not frames:
        raise RuntimeError(
            f"no usable channels in {csv_dir}. Check the path points at the "
            f"folder containing trk_ch_*.csv (often .../tracking), and that "
            f"Prompt_Q is non-zero (not an old real-samples run).")
    return pd.concat(frames, ignore_index=True)


def label(scenario: str, t: float) -> int:
    onset, takeover = ONSETS[scenario]
    if onset is None or t < onset:
        return LABEL_CLEAN
    if t < takeover:
        return LABEL_GUARD
    return LABEL_SPOOFED


def windowize(df: pd.DataFrame, scenario: str, w_s: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df["eml"] = (df["abs_E"] - df["abs_L"]) / (df["abs_P"] + 1e-9)
    df["epl"] = (df["abs_E"] + df["abs_L"]) / (2 * df["abs_P"] + 1e-9)
    df["win"] = (df["t_s"] // w_s).astype(int)

    rows = []
    for (prn, win), g in df.groupby(["PRN", "win"]):
        if len(g) < 5:
            continue
        t_c = win * w_s + w_s / 2
        lab = label(scenario, t_c)
        if lab == LABEL_GUARD:
            continue
        rows.append({
            "scenario": scenario, "prn": int(prn), "t_s": t_c, "label": lab,
            "captured": int(CAPTURED.get(scenario, set()) == "all"
                            or prn in CAPTURED.get(scenario, set())
                            if CAPTURED.get(scenario) else 0),
            "cn0_raw": g["CN0_SNV_dB_Hz"].mean(),
            "cn0_std": g["CN0_SNV_dB_Hz"].std(),
            "lock_mean": g["carrier_lock_test"].mean(),
            "lock_min": g["carrier_lock_test"].min(),
            "eml_mean": g["eml"].mean(), "eml_std": g["eml"].std(),
            "epl_mean": g["epl"].mean(),
            "cerr_std": g["code_error_chips"].std(),
            "dopp_std": g["carrier_doppler_hz"].std(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    onset = ONSETS[scenario][0]
    for raw, delta in [("cn0_raw", "cn0_delta"),
                       ("epl_mean", "epl_delta"),
                       ("eml_mean", "eml_delta")]:
        out[delta] = 0.0
        for prn in out.prn.unique():
            m = out.prn == prn
            ref = out.loc[m & (out.t_s < onset), raw] if onset else out.loc[m, raw]
            base = ref.median() if len(ref) else out.loc[m, raw].median()
            out.loc[m, delta] = out.loc[m, raw] - base
    return out.fillna(0.0)
