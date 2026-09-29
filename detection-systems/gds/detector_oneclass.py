"""
Strict one-class GNSS spoofing detector for TEXBAT.

DATA SPLIT
    Train       = first 60 % of cleanStatic
    Validation  = next 20 % of cleanStatic
    Test        = final 20 % of cleanStatic + all attack scenarios

The cleanStatic split is chronological and based on window timestamps, so all
PRNs from the same time interval stay in the same split. The Mahalanobis model
is fitted only on the clean training interval. The decision threshold is chosen
only from the clean validation interval as the (1 - FPR_TARGET) quantile. No
attack labels or test data are used for fitting or threshold selection.

FOUR SL-INPUT SCORES PER WINDOW
    d_power   cn0_delta, cn0_std
    d_shape   eml_delta, epl_delta, eml_std
    d_track   lock_mean, lock_min, cerr_std
    d_dyn     dopp_std

The combined baseline decision uses d_shape + d_power. All four group distances
are still exported as inputs for the Subjective Logic quantification layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from detector import load_scenario, windowize


GROUPS = {
    "power": ["cn0_delta", "cn0_std"],
    "shape": ["eml_delta", "epl_delta", "eml_std"],
    "track": ["lock_mean", "lock_min", "cerr_std"],
    "dyn": ["dopp_std"],
}
ALL_FEATS = [feature for columns in GROUPS.values() for feature in columns]

# Only the groups that showed useful attack discrimination are combined for the
# conventional one-class decision. All groups remain available in the output.
DECISION_GROUPS = ["shape", "power"]
FPR_TARGET = 0.05
SL_ALL_SCENARIOS = ["cleanStatic", "ds3", "ds4", "ds7", "ds8"]
SL_TRAIN_FRAC = 0.60
SL_VAL_FRAC = 0.20


class GroupMahalanobis:
    """Fit one covariance model per physically motivated feature group."""

    def __init__(self, robust: bool = True):
        self.robust = robust
        self.models: dict[str, tuple[object, list[str]]] = {}

    def fit(self, df_clean: pd.DataFrame) -> "GroupMahalanobis":
        if df_clean.empty:
            raise ValueError("The clean training split is empty.")

        missing = [column for column in ALL_FEATS if column not in df_clean]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        for group, columns in GROUPS.items():
            X = df_clean[columns].to_numpy(dtype=float)
            if not np.isfinite(X).all():
                raise ValueError(f"Non-finite values found in feature group '{group}'.")

            estimator = (
                MinCovDet(support_fraction=0.9, random_state=0)
                if self.robust and X.shape[1] > 1
                else EmpiricalCovariance()
            )
            estimator.fit(X)
            self.models[group] = (estimator, columns)
        return self

    def distances(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("GroupMahalanobis must be fitted before use.")

        output = pd.DataFrame(index=df.index)
        for group, (estimator, columns) in self.models.items():
            X = df[columns].to_numpy(dtype=float)
            output[f"d_{group}"] = estimator.mahalanobis(X)
        return output


def _clean_time_masks(
    clean: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, float]]:
    """Create chronological 60/20/20-style masks using unique window times.

    Splitting by unique timestamps, rather than row positions, ensures that all
    PRNs observed in one time window remain in the same data split.
    """
    if clean.empty:
        raise ValueError("cleanStatic contains no windows.")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be smaller than 1.")

    times = np.sort(clean["t_s"].dropna().unique())
    if len(times) < 3:
        raise ValueError("At least three distinct clean window timestamps are required.")

    n_train = max(1, int(np.floor(len(times) * train_frac)))
    n_val = max(1, int(np.floor(len(times) * val_frac)))

    # Guarantee at least one timestamp for the final test split.
    if n_train + n_val >= len(times):
        n_val = len(times) - n_train - 1
    if n_val < 1:
        raise ValueError("The selected fractions leave no validation timestamps.")

    train_end = float(times[n_train - 1])
    val_end = float(times[n_train + n_val - 1])

    train_mask = clean["t_s"] <= train_end
    val_mask = (clean["t_s"] > train_end) & (clean["t_s"] <= val_end)
    test_mask = clean["t_s"] > val_end

    bounds = {
        "min_time": float(times[0]),
        "train_end": train_end,
        "val_start": float(times[n_train]),
        "val_end": val_end,
        "test_start": float(times[n_train + n_val]),
        "max_time": float(times[-1]),
    }
    return train_mask, val_mask, test_mask, bounds


def _renormalize_clean_from_training(
    clean: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """Recompute cleanStatic deltas using training data only.

    detector.windowize() normally centers cleanStatic with the median of the
    complete clean recording. That would leak validation/test information into
    the training features. Here, the per-PRN baselines are therefore estimated
    only from the first 60 % training interval and then applied to all three
    clean splits.
    """
    output = clean.copy()
    raw_delta_pairs = [
        ("cn0_raw", "cn0_delta"),
        ("epl_mean", "epl_delta"),
        ("eml_mean", "eml_delta"),
    ]

    for raw_column, delta_column in raw_delta_pairs:
        if raw_column not in output:
            raise ValueError(
                f"Column '{raw_column}' is required to recompute '{delta_column}'."
            )

        output[delta_column] = 0.0
        for prn in output["prn"].unique():
            prn_mask = output["prn"] == prn
            reference = output.loc[prn_mask & train_mask, raw_column]
            if reference.empty:
                raise ValueError(
                    f"No clean training samples available for PRN {prn} "
                    f"while normalizing '{raw_column}'."
                )
            baseline = float(reference.median())
            output.loc[prn_mask, delta_column] = (
                output.loc[prn_mask, raw_column] - baseline
            )

    return output.fillna(0.0)


def run(
    scenario_dirs: dict[str, str],
    out_prefix: str = "oneclass",
    clean_name: str = "cleanStatic",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    fpr_target: float = FPR_TARGET,
    window_seconds: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Fit, validate, and evaluate the strict one-class detector.

    Writes one CSV per test scenario: {out_prefix}_{scenario}.csv"""
    if clean_name not in scenario_dirs:
        raise KeyError(f"Missing clean scenario '{clean_name}' in scenario_dirs.")
    if not 0.0 < fpr_target < 1.0:
        raise ValueError("fpr_target must be between 0 and 1.")

    # ---- Windowed features per scenario -------------------------------------
    parts: dict[str, pd.DataFrame] = {}
    for scenario, directory in scenario_dirs.items():
        windows = windowize(load_scenario(directory), scenario, w_s=window_seconds)
        if windows.empty:
            raise RuntimeError(f"No windows generated for scenario '{scenario}'.")
        parts[scenario] = windows
        print(
            f"{scenario:12s}: {len(windows):5d} windows ({window_seconds}s)  "
            f"labels={windows.label.value_counts().to_dict()}"
        )

    # ---- Chronological clean split: 60 % / 20 % / 20 % ----------------------
    clean_raw = parts[clean_name].sort_values(["t_s", "prn"]).reset_index(drop=True)
    train_mask, val_mask, test_mask, bounds = _clean_time_masks(
        clean_raw, train_frac=train_frac, val_frac=val_frac
    )

    # Avoid normalization leakage from clean validation/test into training.
    clean = _renormalize_clean_from_training(clean_raw, train_mask)
    clean_train = clean.loc[train_mask].copy()
    clean_val = clean.loc[val_mask].copy()
    clean_test = clean.loc[test_mask].copy()

    test_frac = 1.0 - train_frac - val_frac
    print("\n=== cleanStatic chronological split ===")
    print(
        f"  train {train_frac:5.1%}: {len(clean_train):5d} windows, "
        f"t={bounds['min_time']:.2f}..{bounds['train_end']:.2f} s"
    )
    print(
        f"  val   {val_frac:5.1%}: {len(clean_val):5d} windows, "
        f"t={bounds['val_start']:.2f}..{bounds['val_end']:.2f} s"
    )
    print(
        f"  test  {test_frac:5.1%}: {len(clean_test):5d} windows, "
        f"t={bounds['test_start']:.2f}..{bounds['max_time']:.2f} s"
    )

    # ---- Fit exclusively on the first 60 % of cleanStatic ------------------
    model = GroupMahalanobis(robust=True).fit(clean_train)

    def add_distances(df: pd.DataFrame) -> pd.DataFrame:
        distances = model.distances(df)
        result = pd.concat(
            [df.reset_index(drop=True), distances.reset_index(drop=True)], axis=1
        )
        decision_columns = [f"d_{group}" for group in DECISION_GROUPS]
        result["d_total"] = result[decision_columns].sum(axis=1)

        degrees_of_freedom = sum(len(GROUPS[group]) for group in DECISION_GROUPS)
        result["p_spoof"] = chi2.cdf(result["d_total"], df=degrees_of_freedom)
        return result

    # ---- Validation: threshold from clean data only -------------------------
    validation = add_distances(clean_val)
    threshold = float(np.quantile(validation["d_total"], 1.0 - fpr_target))
    validation["pred"] = (validation["d_total"] >= threshold).astype(int)
    validation_fpr = float(validation["pred"].mean())

    print("\n=== validation on cleanStatic only ===")
    print(
        f"  threshold = quantile({1.0 - fpr_target:.3f}) = {threshold:.3f}"
    )
    print(
        f"  requested FPR={fpr_target:.3f}, observed validation FPR={validation_fpr:.3f} "
        f"({int(validation['pred'].sum())}/{len(validation)})"
    )

    # ---- Test: final 20 % cleanStatic + all complete attack scenarios -------
    test_parts: dict[str, pd.DataFrame] = {clean_name: clean_test}
    for scenario, windows in parts.items():
        if scenario != clean_name:
            test_parts[scenario] = windows

    keep = (
        ["scenario", "prn", "t_s", "label", "captured", "split"]
        + [f"d_{group}" for group in GROUPS]
        + ["d_total", "p_spoof", "pred"]
    )

    print("\n=== per-scenario test results ===")
    outputs: dict[str, pd.DataFrame] = {}
    for scenario, windows in test_parts.items():
        evaluated = add_distances(windows)
        evaluated["pred"] = (evaluated["d_total"] >= threshold).astype(int)
        evaluated["split"] = "test"

        y_true = evaluated["label"].to_numpy()
        y_pred = evaluated["pred"].to_numpy()

        if len(np.unique(y_true)) > 1:
            print(
                f"  {scenario:12s} "
                f"F1={f1_score(y_true, y_pred, zero_division=0):.3f}  "
                f"AUROC={roc_auc_score(y_true, evaluated['d_total']):.3f}  "
                f"prec={precision_score(y_true, y_pred, zero_division=0):.3f}  "
                f"rec={recall_score(y_true, y_pred, zero_division=0):.3f}  "
                f"n={len(evaluated)}"
            )
        else:
            fp = int(y_pred.sum())
            # cleanDynamic (or any non-fitted clean scenario) is out-of-
            # distribution: the model was fitted on the STATIC platform, so a
            # high FPR here reflects platform mobility, not detector quality.
            ood = ("  [OUT-OF-DISTRIBUTION: dynamic platform vs static-fitted "
                   "model]" if scenario != clean_name else "")
            print(
                f"  {scenario:12s} FPR={float(y_pred.mean()):.3f} "
                f"({fp}/{len(evaluated)} false positives)  "
                f"[F1 undefined: no positives]{ood}"
            )

        out_csv = f"{out_prefix}_{scenario}.csv"
        evaluated[keep].to_csv(out_csv, index=False)
        outputs[scenario] = evaluated[keep]

    # ---- combined CSV: all scenarios in one file ----------------------------
    # cleanStatic appears with its TEST portion only (train/val excluded), so no
    # fitting data leaks into the combined evaluation file.
    combined = pd.concat(outputs.values(), ignore_index=True)
    combined_csv = f"{out_prefix}_all.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"\ncombined output ({len(combined)} windows, "
          f"{combined.scenario.nunique()} scenarios) -> {combined_csv}")

    # ---- overall metrics on the combined set --------------------------------
    y = combined["label"].to_numpy()
    p = combined["pred"].to_numpy()
    if len(np.unique(y)) > 1:
        print("\n=== overall (oneclass_all.csv) ===")
        print(
            f"  F1={f1_score(y, p, zero_division=0):.3f}  "
            f"AUROC={roc_auc_score(y, combined['d_total']):.3f}  "
            f"prec={precision_score(y, p, zero_division=0):.3f}  "
            f"rec={recall_score(y, p, zero_division=0):.3f}  n={len(combined)}"
        )
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        print(f"  (class balance: {n_pos} spoofed / {n_neg} clean "
              f"= {n_pos/len(y):.0%} positive -- overall F1 is optimistically "
              f"biased by this imbalance)")

        # balanced F1: all positives vs an equal-sized random clean sample
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if len(neg_idx) >= 10:
            rng = np.random.default_rng(0)
            k = min(len(pos_idx), len(neg_idx))
            sel = np.concatenate([rng.choice(pos_idx, k, replace=False),
                                  rng.choice(neg_idx, k, replace=False)])
            f1_bal = f1_score(y[sel], p[sel], zero_division=0)
            print(f"  balanced F1 (50/50 positives vs clean sample) = {f1_bal:.3f}")

    # ---- per-group diagnosis: attack scenarios only (exclude clean OOD) ------
    attack = pd.concat(
        [df for sc, df in outputs.items()
         if df["label"].nunique() > 1], ignore_index=True
    )
    if len(attack):
        y_all = attack["label"].to_numpy()
        print("\n=== per-group AUROC (attack scenarios pooled) ===")
        for group in GROUPS:
            print(f"  {group:6s} AUROC={roc_auc_score(y_all, attack[f'd_{group}']):.3f}")

    print(f"\nper-scenario outputs -> {out_prefix}_<scenario>.csv")
    print(f"SL-input columns: {', '.join('d_' + group for group in GROUPS)}")

    # Every scenario contributes to train, val AND test with disjoint windows,
    # so the SL layer sees all attack types during training -- matching the V2X
    # domain's regime. Split is CHRONOLOGICAL per scenario (first 60% train, next
    # 20% val, last 20% test) so adjacent correlated windows don't leak across
    # splits. The detector itself stays one-class (fitted on cleanStatic only);
    # only the SL layer above sees all attacks.
    sl_scenarios = [s for s in SL_ALL_SCENARIOS if s in outputs]
    if not sl_scenarios:
        print(f"\n[SL split skipped] none of {SL_ALL_SCENARIOS} present")
        return outputs

    sl_parts = {"sl_train": [], "sl_val": [], "sl_test": []}
    for sc in sl_scenarios:
        df = outputs[sc].sort_values("t_s").reset_index(drop=True)
        # Stratify by label: split clean and spoofed windows SEPARATELY, each
        # chronologically 60/20/20, then recombine. Without this, a scenario's
        # clean prefix (all at the start) lands entirely in train, leaving val
        # and test almost purely spoofed.
        for lab in (0, 1):
            sub = df[df["label"] == lab]
            if len(sub) == 0:
                continue
            n = len(sub)
            i_tr = int(n * SL_TRAIN_FRAC)
            i_va = int(n * (SL_TRAIN_FRAC + SL_VAL_FRAC))
            sl_parts["sl_train"].append(sub.iloc[:i_tr])
            sl_parts["sl_val"].append(sub.iloc[i_tr:i_va])
            sl_parts["sl_test"].append(sub.iloc[i_va:])

    for name, parts_list in sl_parts.items():
        merged = pd.concat(parts_list, ignore_index=True)
        path = f"{out_prefix}_{name}.csv"
        merged.to_csv(path, index=False)
        bal = merged["label"].value_counts().to_dict()
        scns = merged["scenario"].value_counts().to_dict()
        print(f"  {name:9s} -> {path}  ({len(merged)} windows, "
              f"labels={bal}, scenarios={scns})")

    print(f"  SL split: every scenario {int(SL_TRAIN_FRAC*100)}/"
          f"{int(SL_VAL_FRAC*100)}/{int((1-SL_TRAIN_FRAC-SL_VAL_FRAC)*100)} "
          f"chronological (all attacks in train, val and test)")
    return outputs


if __name__ == "__main__":
    directories = {
        "cleanStatic": "csv_outputs/cleanStatic/tracking",
        "ds3": "csv_outputs/ds3/tracking",
        "ds4": "csv_outputs/ds4/tracking",
        "ds7": "csv_outputs/ds7/tracking",
        "ds8": "csv_outputs/ds8/tracking",
    }
    run(directories)