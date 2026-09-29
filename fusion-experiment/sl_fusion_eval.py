#!/usr/bin/env python3
"""
Subjective Logic fusion of two detection systems: the prediction-based MDS
(ai_mbd) and the GDS (gnss).

For each detection system, a quantification approach outputs per message a
binomial Subjective Logic opinion
(b, d, u) about the proposition "the message is benign" with base rate
a = 0.5. This script pairs gnss messages with ai_mbd messages, fuses each
pair's two opinions with several SL fusion operators and reports F1 scores
for detecting malicious messages.

Input files (see INPUT_PATHS), one CSV per (source, variant), columns:
    attack, y_true, y_pred, correct, p, b, d, u
  - y_true == 1 -> benign, y_true == 0 -> malicious (attack)
  - b = belief(benign), d = belief(malicious), u = uncertainty
  - p = b + a*u (a = 0.5) is the projected probability of benign;
    y_pred == 1 iff p >= 0.5

Pairing (gnss-driven, one group per distinct gnss `attack` value):
  - Draw N_SAMPLES gnss rows for the attack type (with replacement):
    ATTACK_RATE of them from its malicious pool, the rest from its benign
    pool, so every group has an attack rate of ATTACK_RATE by construction.
  - Benign gnss row    -> partner is a random benign ai_mbd row (any attack type).
  - Malicious gnss row -> partner is a random malicious ai_mbd row whose attack
    type is in AI_MBD_MALICIOUS_ATTACKS.
  The ground truth of a pair is therefore the label of its gnss row.

The pairing is built once per seed from the "mlp" files and reused by row
position for every other variant, so differences between variants/operators
within a seed are due to the quantification method and fusion operator only.
This requires all variants' files to be row-aligned per source, which
assert_row_aligned() checks at startup. Across seeds the pairing is redrawn
independently to measure sampling variability.

Evaluation per (variant, operator): a pair is predicted malicious iff the
fused projected probability < 0.5. Reported (malicious = positive class):
  - F1 per gnss attack type
  - macro-F1   = mean of the per-attack-type F1 scores
  - overall F1 = F1 over all pairs pooled together

Usage:
    python3 sl_fusion_eval.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
BASE_RATE = 0.5
N_SAMPLES = 1_000_000  # pairs per gnss attack-type group
ATTACK_RATE = 0.20     # fraction of malicious pairs per group
SEEDS = [42, 43, 44]   # one independent random pairing per seed

# ai_mbd attack types eligible as malicious partner for a malicious gnss row.
# Types missing from the ai_mbd files are simply absent from the pool.
AI_MBD_MALICIOUS_ATTACKS = ["constantPositionOffset", "randomPositionOffset", "suddenStop"]

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"
VARIANTS = ["mlp", "analytic_f1"]

# (ai_mbd path, gnss path) per variant. Files of the same source must be
# row-aligned across variants (same message, same order).
INPUT_PATHS = {
    "mlp": (
        DATA_DIR / "ai_mbd_mlp.csv",
        DATA_DIR / "gnss_mlp.csv",
    ),
    "analytic_f1": (
        DATA_DIR / "ai_mbd_analytic_f1.csv",
        DATA_DIR / "gnss_analytic_f1.csv",
    ),
}


# --------------------------------------------------------------- SL operators
# Signature (b1, d1, u1, a1, b2, d2, u2, a2) -> (b, d, u, a), numpy-array-safe.

def cbf(b1, d1, u1, a1, b2, d2, u2, a2):
    """Cumulative Belief Fusion: weights by inverse uncertainty."""
    k = u1 + u2 - u1 * u2
    ks = np.where(k <= EPS, 1.0, k)
    b = np.where(k <= EPS, 0.5 * (b1 + b2), (b1 * u2 + b2 * u1) / ks)
    d = np.where(k <= EPS, 0.5 * (d1 + d2), (d1 * u2 + d2 * u1) / ks)
    u = np.where(k <= EPS, 0.0, (u1 * u2) / ks)
    return b, d, u, a1


def abf(b1, d1, u1, a1, b2, d2, u2, a2):
    """Averaging Belief Fusion, for dependent/correlated sources."""
    s = u1 + u2
    ss = np.where(s <= EPS, 1.0, s)
    b = np.where(s <= EPS, 0.5 * (b1 + b2), (b1 * u2 + b2 * u1) / ss)
    d = np.where(s <= EPS, 0.5 * (d1 + d2), (d1 * u2 + d2 * u1) / ss)
    u = np.where(s <= EPS, 0.0, (2 * u1 * u2) / ss)
    return b, d, u, a1


def bcf(b1, d1, u1, a1, b2, d2, u2, a2):
    """Belief Constraint Fusion (Dempster's rule), Josang (2016) Eq. (12.20)."""
    conflict = b1 * d2 + d1 * b2
    denom = 1.0 - conflict
    ds = np.where(denom <= EPS, 1.0, denom)
    b = np.where(denom <= EPS, 0.5 * (b1 + b2), (b1 * b2 + b1 * u2 + b2 * u1) / ds)
    d = np.where(denom <= EPS, 0.5 * (d1 + d2), (d1 * d2 + d1 * u2 + d2 * u1) / ds)
    u = np.where(denom <= EPS, 0.0, (u1 * u2) / ds)
    return b, d, u, a1


def ccf(b1, d1, u1, a1, b2, d2, u2, a2):
    """Consensus & Compromise Fusion, Josang (2016) Ch. 12.6."""
    b_cons, d_cons = np.minimum(b1, b2), np.minimum(d1, d2)
    b1_res, b2_res = np.maximum(0.0, b1 - b_cons), np.maximum(0.0, b2 - b_cons)
    d1_res, d2_res = np.maximum(0.0, d1 - d_cons), np.maximum(0.0, d2 - d_cons)
    consensus_mass = b_cons + d_cons
    b_comp = b1_res * u2 + b2_res * u1 + b1_res * b2_res
    d_comp = d1_res * u2 + d2_res * u1 + d1_res * d2_res
    x_comp = b1_res * d2_res + d1_res * b2_res
    u_pre = u1 * u2
    compromise_mass = b_comp + d_comp + x_comp
    cms = np.where(compromise_mass > EPS, compromise_mass, 1.0)
    eta = np.where(compromise_mass > EPS, (1.0 - consensus_mass - u_pre) / cms, 1.0)
    b = np.clip(b_cons + eta * b_comp, 0.0, 1.0)
    d = np.clip(d_cons + eta * d_comp, 0.0, 1.0)
    u = np.clip(1.0 - b - d, 0.0, 1.0)
    return b, d, u, a1


def mult(b1, d1, u1, a1, b2, d2, u2, a2):
    """Binomial multiplication (AND), Josang (2016) Eq. (7.1)."""
    denom = 1.0 - a1 * a2
    d = d1 + d2 - d1 * d2
    a = a1 * a2
    if denom <= EPS:
        return b1 * b2, d, u1 * u2, a
    b = b1 * b2 + ((1 - a1) * a2 * b1 * u2 + a1 * (1 - a2) * u1 * b2) / denom
    u = u1 * u2 + ((1 - a2) * b1 * u2 + (1 - a1) * u1 * b2) / denom
    return b, d, u, a


OPERATORS = {"cbf": cbf, "abf": abf, "bcf": bcf, "ccf": ccf, "mult": mult}


# ------------------------------------------------------------------- pairing
def build_pairs(ai_df: pd.DataFrame, gnss_df: pd.DataFrame, rng: np.random.Generator):
    """Returns {(source, attack): dict(ai_idx, gnss_idx, malicious)}."""
    ai_benign_idx = ai_df.index[ai_df.y_true == 1].to_numpy()
    ai_malicious_idx = ai_df.index[
        (ai_df.y_true == 0) & ai_df.attack.isin(AI_MBD_MALICIOUS_ATTACKS)].to_numpy()
    if len(ai_malicious_idx) == 0:
        raise ValueError(
            f"No ai_mbd rows with y_true==0 found for attack types "
            f"{AI_MBD_MALICIOUS_ATTACKS}. Available attack types with y_true==0: "
            f"{sorted(ai_df.loc[ai_df.y_true == 0, 'attack'].unique())}")

    n_malicious = round(N_SAMPLES * ATTACK_RATE)
    n_benign = N_SAMPLES - n_malicious

    pairs = {}
    for attack in sorted(gnss_df["attack"].unique()):
        gnss_benign_pool = gnss_df.index[(gnss_df.attack == attack) & (gnss_df.y_true == 1)].to_numpy()
        gnss_malicious_pool = gnss_df.index[(gnss_df.attack == attack) & (gnss_df.y_true == 0)].to_numpy()
        if len(gnss_benign_pool) == 0 or len(gnss_malicious_pool) == 0:
            raise ValueError(
                f"gnss attack '{attack}' has no rows for one of the two classes "
                f"(benign={len(gnss_benign_pool)}, malicious={len(gnss_malicious_pool)}) "
                f"-- cannot draw a {ATTACK_RATE:.0%} attack rate for this group.")

        primary = np.concatenate([
            rng.choice(gnss_benign_pool, size=n_benign, replace=True),
            rng.choice(gnss_malicious_pool, size=n_malicious, replace=True),
        ])
        partner = np.concatenate([
            rng.choice(ai_benign_idx, size=n_benign, replace=True),
            rng.choice(ai_malicious_idx, size=n_malicious, replace=True),
        ])
        malicious = np.concatenate([np.zeros(n_benign, dtype=bool), np.ones(n_malicious, dtype=bool)])

        pairs[("gnss", attack)] = dict(ai_idx=partner, gnss_idx=primary, malicious=malicious)

    return pairs


# ---------------------------------------------------------------- evaluation
def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return precision, recall, np.nan
    return precision, recall, 2 * precision * recall / (precision + recall)


def evaluate(pairs, ai_df, gnss_df, fuse):
    """Fuse every pair with `fuse`; return (per-attack F1, macro-F1, overall F1)."""
    per_attack = {}
    pooled_tp = pooled_fp = pooled_fn = 0
    for (source, attack), pair in pairs.items():
        b1, d1, u1 = (ai_df.loc[pair["ai_idx"], c].to_numpy() for c in ("b", "d", "u"))
        b2, d2, u2 = (gnss_df.loc[pair["gnss_idx"], c].to_numpy() for c in ("b", "d", "u"))

        bf, _, uf, af = fuse(b1, d1, u1, BASE_RATE, b2, d2, u2, BASE_RATE)
        pred_malicious = (bf + af * uf) < 0.5

        malicious = pair["malicious"]
        tp = int(np.sum(malicious & pred_malicious))
        fp = int(np.sum(~malicious & pred_malicious))
        fn = int(np.sum(malicious & ~pred_malicious))
        per_attack[f"{source}:{attack}"] = precision_recall_f1(tp, fp, fn)[2]
        pooled_tp += tp
        pooled_fp += fp
        pooled_fn += fn

    macro_f1 = float(np.nanmean(list(per_attack.values())))
    overall_f1 = precision_recall_f1(pooled_tp, pooled_fp, pooled_fn)[2]
    return per_attack, macro_f1, overall_f1


def assert_row_aligned(ref_df: pd.DataFrame, other_df: pd.DataFrame, source: str, other_variant: str):
    """Abort if `other_df` is not row-aligned with the mlp reference file, since
    the mlp-based pairing indices would otherwise point at the wrong messages."""
    if len(ref_df) != len(other_df):
        raise ValueError(
            f"{source}: mlp file has {len(ref_df)} rows but {other_variant} file "
            f"has {len(other_df)} rows -- row order/count must match per source.")
    mismatch = (ref_df["attack"].to_numpy() != other_df["attack"].to_numpy()) | \
               (ref_df["y_true"].to_numpy() != other_df["y_true"].to_numpy())
    if mismatch.any():
        first = int(np.argmax(mismatch))
        raise ValueError(
            f"{source}: mlp and {other_variant} files are not row-aligned "
            f"({int(mismatch.sum())} mismatching rows, first at row {first}: "
            f"mlp={ref_df.loc[first, ['attack', 'y_true']].to_dict()} vs "
            f"{other_variant}={other_df.loc[first, ['attack', 'y_true']].to_dict()}).")


def main():
    print(f"[info] N_SAMPLES={N_SAMPLES} per gnss attack-type group, "
          f"attack rate {ATTACK_RATE:.0%} per group")
    print(f"[info] malicious ai_mbd partner drawn from {AI_MBD_MALICIOUS_ATTACKS}")
    print(f"[info] seeds={SEEDS} (each seed = one independent random pairing)\n")

    variant_dfs = {variant: (pd.read_csv(ai_path), pd.read_csv(gnss_path))
                   for variant, (ai_path, gnss_path) in INPUT_PATHS.items()}

    ai_mlp, gnss_mlp = variant_dfs["mlp"]
    for variant, (ai_df, gnss_df) in variant_dfs.items():
        if variant != "mlp":
            assert_row_aligned(ai_mlp, ai_df, "ai_mbd", variant)
            assert_row_aligned(gnss_mlp, gnss_df, "gnss", variant)

    detail_rows = []
    for seed in SEEDS:
        pairs = build_pairs(ai_mlp, gnss_mlp, np.random.default_rng(seed))
        for variant in VARIANTS:
            ai_df, gnss_df = variant_dfs[variant]
            for op_name, fuse in OPERATORS.items():
                per_attack, macro_f1, overall_f1 = evaluate(pairs, ai_df, gnss_df, fuse)
                row = {"seed": seed, "variant": variant, "operator": op_name}
                row.update({f"f1[{key}]": f1 for key, f1 in per_attack.items()})
                row["macro_f1"] = macro_f1
                row["overall_f1"] = overall_f1
                detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    detail_path = OUTPUT_DIR / "sl_fusion_results_v5_per_seed.csv"
    detail_df.to_csv(detail_path, index=False, float_format="%.4f")
    print(f"[wrote] {detail_path} ({len(SEEDS)} seeds x {len(VARIANTS)} variants x "
          f"{len(OPERATORS)} operators)")

    agg = (detail_df.groupby(["variant", "operator"])
           .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                overall_f1_mean=("overall_f1", "mean"), overall_f1_std=("overall_f1", "std"))
           .reset_index())

    for variant in VARIANTS:
        sub = (agg[agg["variant"] == variant].drop(columns="variant")
               .sort_values("overall_f1_mean", ascending=False))
        out_path = OUTPUT_DIR / f"sl_fusion_results_{variant}_v5.csv"
        sub.to_csv(out_path, index=False, float_format="%.4f")
        print(f"\n=== {variant} (mean +/- std over {len(SEEDS)} seeds) ===")
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"[wrote] {out_path}")

    pivot = agg.pivot(index="operator", columns="variant",
                      values=["macro_f1_mean", "macro_f1_std", "overall_f1_mean", "overall_f1_std"])
    pivot = pivot.sort_values(("overall_f1_mean", "analytic_f1"), ascending=False)
    summary_path = OUTPUT_DIR / "sl_fusion_results_summary_v5.csv"
    pivot.to_csv(summary_path, float_format="%.4f")
    print("\n=== mlp vs analytic_f1 (mean/std over seeds, per operator) ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
