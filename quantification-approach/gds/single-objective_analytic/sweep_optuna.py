"""
Optuna multi-objective sweep for the closed-form Subjective Logic (SL)
GNSS spoofing detector.

Each of the four grouped Mahalanobis distances (d_power, d_shape, d_track,
d_dyn) is mapped to a per-feature SL opinion (b, d, u) with a closed-form
function, the four opinions are fused (CBF/AVG/WBF), and the fused opinion
drives the benign/spoofed decision at the fixed operating point 0.5.

Workflow:
  - Data are loaded ONCE per worker process (no re-parsing per trial).
  - Each trial is evaluated ONLY on the validation split (no test set, no
    paper outputs during the sweep itself).
  - 5 Pareto objectives: macro_f1, macro_aurc, macro_misclass_auroc,
    macro_delta_u, macro_belief_correctness.
  - After all trials: one "balanced" trial is selected via 4 knee-point
    methods + majority consensus.
  - Final test run for the balanced trial ONLY: Tables 1-5 + per-sample CSV
    + extended boxplot (12 categories).

Sampler: NSGA-II with population_size=50 (the B-spline/MLP sweeps use a
smaller population because their search spaces are much smaller). The
search space here has 21 dimensions (4 features x 5 HPs + fusion_op).

Usage:
    python sweep_optuna.py --n-trials 2000 --data-dir ../data
    python sweep_optuna.py --resume --n-trials 5000
    python sweep_optuna.py --analyze-only
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna

import sl_model as sl_main
import metrics
import paper_outputs
import data_cache_gnss as dc


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_OUTPUT_DIR = "results/analytic_sweep"

# Aggregation statistic for macro_delta_u. ONE switch for all places that use
# it (objective, final-test macro, tables 1/4/5), so selection and reporting
# cannot diverge.
#   "median" - median within each attack, mean across attacks. Robust, but
#              with a 50% breakdown point it is blind to the confident-wrong
#              tail.
#   "mean"   - mean within each attack. Sees the tail, but can make
#              r(macro_f1, macro_delta_u) negative; on the V2X data NSGA-II
#              then lost the feasible region (F1 collapse). With "mean", run a
#              small first sweep and check that the Pareto front still keeps
#              macro_f1 high.
DELTA_U_STAT = "mean"

STUDY_DB = "sqlite:///hp_optuna_gnss_v1.db"
# STUDY_NAME encodes the objective set AND the attack group set. Whenever
# OBJECTIVE_NAMES, selected_attacks, DECISION_THR_FIXED or the search space
# change, the name MUST be bumped - otherwise trials are selected on values
# computed under one definition and reported under another.
STUDY_NAME = f"gnss_hp_pareto_v1_5obj_8scn_{DELTA_U_STAT}du"

# Five Pareto objectives — ALL computed on VALIDATION during the sweep
OBJECTIVE_NAMES = [
    "macro_f1", "macro_aurc", "macro_misclass_auroc",
    "macro_delta_u", "macro_belief_correctness",
]
OBJECTIVE_DIRECTION = ["maximize", "minimize", "maximize", "maximize", "maximize"]

# Attack groups for the macro metrics. Must match the `attack` values from
# data_cache_gnss.load_split exactly (= TEXBAT scenario name).
#
# cleanStatic is DELIBERATELY NOT included. The group is single-class (benign
# only) and yields a constant F1 = 0.0 in _trial_task regardless of the model:
#   perfectly classified -> tp=fp=fn=0 -> denom=0 -> fallback 0.0
#   with false positives -> tp=0       -> 2*tp/denom = 0.0
# It would cap macro_f1 at (n_groups-1)/n_groups without saying anything about
# model quality, and it yields NaN for _macro_belief_correctness (no
# truth==0). The benign samples of the clean sections INSIDE the ds scenarios
# are kept and carry the false-positive evaluation.
selected_attacks = [
    "ds3", "ds4", "ds7", "ds8",
]

FEATURE_ORDER = dc.FEATURE_ORDER  # ["d_power", "d_shape", "d_track", "d_dyn"]

# Fixed order of the attack groups. The ONLY place where groups are formed -
# sweep and final test cannot diverge.
_ATTACK_TO_IDX = {a: i for i, a in enumerate(selected_attacks)}
N_ATTACKS = len(selected_attacks)


def _validate_attack_groups(attack_arr, y, where):
    """
    Fails loudly if selected_attacks does not match the data.

    Without this check a typo or a naming mismatch ("DS3" instead of "ds3")
    is silent: np.isin returns an empty keep mask, the split shrinks to 0
    rows and the sweep keeps running with degenerate metrics instead of
    aborting.
    """
    a = np.asarray(attack_arr).astype(str)
    y = np.asarray(y).reshape(-1)
    present = set(a.tolist())
    missing = [s for s in selected_attacks if s not in present]
    if missing:
        raise ValueError(
            f"[{where}] selected_attacks not found in the data: {missing}. "
            f"Available scenarios: {sorted(present)}. "
            f"Fix selected_attacks in sweep_optuna.py.")
    extra = sorted(present - set(selected_attacks))
    if extra:
        print(f"  [{where}] scenarios not evaluated (dropped): {extra}")
    for grp in selected_attacks:
        m = (a == grp)
        nb = int((y[m] == 1).sum()); na = int((y[m] == 0).sum())
        if nb == 0 or na == 0:
            raise ValueError(
                f"[{where}] group '{grp}' is single-class (benign={nb}, "
                f"attacker={na}). macro_f1 would be constant 0.0 for it and "
                f"macro_belief_correctness NaN. Remove the group from "
                f"selected_attacks or check how the split was created.")


def _attack_filter_and_idx(attack_arr):
    """
    attack strings -> (keep_mask, idx, n_attacks) with the FIXED order from
    selected_attacks (never derived from the data, so the sweep and the paper
    tables always average over the same groups).
    """
    a = np.asarray(attack_arr).astype(str)
    keep = np.isin(a, selected_attacks)
    a_keep = a[keep]
    idx = np.zeros(len(a_keep), dtype=np.int64)
    for i, name in enumerate(selected_attacks):
        idx[a_keep == name] = i
    return keep, idx, N_ATTACKS


FUSION_OPTIONS = ["cbf", "avg", "wbf"]

# Trust discount: one reliability weight in [0, 1] per feature.
# DISABLED: trust parameters can raise u globally and destroy MAUROC. In a
# previous V2X run with trust enabled, the balanced trial ended up at
# macro_misclass_auroc = 0.40 (< 0.5, i.e. worse than random).
USE_TRUST_DISCOUNT = False

TIMEOUT_PER_TRIAL_S = 900


# ============================================================================
# Worker pool with one-time data loading
# ============================================================================
# Global worker state (each process has its own copy)

_DATA_BY_SPLIT = None     # dict: split -> (msgs_np, attack_idx, n_attacks)


def load_filtered_split(split, data_dir, where=None):
    """
    Loads one split, validates and restricts it to selected_attacks (fixed
    group order) and returns (msgs, attack, attack_idx, n_attacks, n_dropped).

    msgs = [X (N,4), truth (N,1)] -> (N,5); compute_opinions reads truth from
    column Attribute.truth.value == 4.
    """
    X, y, attack, _, _ = dc.load_split(split, data_dir=data_dir)
    _validate_attack_groups(attack, y, where=where or split)
    keep, attack_idx, n_attacks = _attack_filter_and_idx(attack)
    X, y = X[keep], y[keep]
    attack = np.asarray(attack).astype(str)[keep]
    msgs = np.concatenate(
        [X.astype(np.float64), y.astype(np.float64).reshape(-1, 1)], axis=1,
    )
    return msgs, attack, attack_idx, n_attacks, int((~keep).sum())


def _init_worker(data_dir, splits_to_load):
    """
    Runs ONCE per worker process.
    Loads the requested splits from the GNSS CSVs.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    global _DATA_BY_SPLIT
    _DATA_BY_SPLIT = {}
    for split in splits_to_load:
        msgs, _attack, attack_idx, n_attacks, _ = load_filtered_split(
            split, data_dir, where=f"Worker/{split}")
        _DATA_BY_SPLIT[split] = (msgs, attack_idx, n_attacks)


def _belief_correctness_balanced(b, d, truth, y_pred):
    """
    Class-balanced belief correctness:
        (mean(b | y_true=1, correct) + mean(d | y_true=0, correct)) / 2

    Analogous to the balanced loss in the B-spline/MLP sweeps: averages PER
    CLASS instead of PER SAMPLE.
    """
    correct = (y_pred == truth)
    mask_b = (truth == 1) & correct
    mask_a = (truth == 0) & correct
    if not mask_b.any() or not mask_a.any():
        return float("nan")
    bcb = float(b[mask_b].mean())
    dca = float(d[mask_a].mean())
    return (bcb + dca) / 2.0


def _macro_belief_correctness(b, d, truth, y_pred, attack_idx, n_attacks):
    """Macro average of the balanced belief correctness per attack."""
    scores = []
    for k in range(n_attacks):
        mask = attack_idx == k
        if mask.sum() < 2:
            continue
        score = _belief_correctness_balanced(
            b[mask], d[mask], truth[mask], y_pred[mask]
        )
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


def _trial_task(alpB, betB, alpD, betD, thr, decision_thr,
                fusion_op, trust, kl_weight, split):
    """
    Per trial: forward pass on the requested split, computation of the 5
    macro objectives + micro stats + confusion matrix.
    """
    if _DATA_BY_SPLIT is None:
        raise RuntimeError("Worker dataset not initialized")
    if split not in _DATA_BY_SPLIT:
        raise RuntimeError(f"Split '{split}' not loaded in worker")

    msgs, attack_idx, n_attacks = _DATA_BY_SPLIT[split]

    b, d, u, truth = sl_main.compute_opinions(
        msgs, alpB, betB, alpD, betD, thr, fusion_op=fusion_op, trust=trust
    )

    p_benign = b + 0.5 * u
    # '>=' - identical to run_final_test and to the MLP/B-spline sweeps.
    # Critical, because p = b + 0.5*u puts a vacuous opinion (u=1) EXACTLY at
    # 0.5 and CBF/WBF fusion pushes a lot of mass there: '>' vs '>=' flips
    # these samples wholesale from benign to attacker.
    y_pred_benign = (p_benign >= decision_thr).astype(np.int64)
    correct = (y_pred_benign == truth).astype(np.int64)

    # --- Micro confusion (attacker = positive class) ---
    is_atk   = (truth == 0)
    pred_atk = (y_pred_benign == 0)
    is_ben   = ~is_atk
    tp_arr = is_atk & pred_atk
    fp_arr = is_ben & pred_atk
    fn_arr = is_atk & ~pred_atk
    tp = int(tp_arr.sum())
    fp = int(fp_arr.sum())
    tn = int((is_ben & ~pred_atk).sum())
    fn = int(fn_arr.sum())
    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom > 0 else 0.0

    # --- Macro F1 via vectorized bincount ---
    tp_per = np.bincount(attack_idx, weights=tp_arr.astype(np.float64), minlength=n_attacks)
    fp_per = np.bincount(attack_idx, weights=fp_arr.astype(np.float64), minlength=n_attacks)
    fn_per = np.bincount(attack_idx, weights=fn_arr.astype(np.float64), minlength=n_attacks)
    denom_per = 2.0 * tp_per + fp_per + fn_per
    f1_per_atk = np.where(denom_per > 0, (2.0 * tp_per) / np.maximum(denom_per, 1e-9), 0.0)
    macro_f1 = float(f1_per_atk.mean())

    confidence = 1.0 - u
    aurc = metrics._aurc(confidence, correct)
    misclass_auroc = metrics._misclass_auroc(u, correct)
    if np.isnan(misclass_auroc):
        misclass_auroc = 0.5
    delta_u_micro = metrics._delta_u(u, correct, statistic=DELTA_U_STAT)
    if np.isnan(delta_u_micro):
        delta_u_micro = 0.0

    macro_aurc = metrics._macro_aurc(confidence, correct, attack_idx, n_attacks)
    macro_misclass_auroc = metrics._macro_misclass_auroc(u, correct, attack_idx, n_attacks)
    if np.isnan(macro_misclass_auroc):
        macro_misclass_auroc = 0.5
    macro_delta_u = metrics._macro_delta_u(u, correct, attack_idx, n_attacks,
                                           statistic=DELTA_U_STAT)
    if np.isnan(macro_delta_u):
        macro_delta_u = 0.0

    # --- belief_correctness (5th objective) ---
    macro_belief_correctness = _macro_belief_correctness(
        b, d, truth, y_pred_benign, attack_idx, n_attacks
    )
    if np.isnan(macro_belief_correctness):
        macro_belief_correctness = 0.0

    evidential_loss = metrics.evidential_loss_from_opinion(
        b, d, u, truth, kl_weight=kl_weight
    )

    return {
        "f1": float(f1),
        "macro_f1": macro_f1,
        "aurc": float(aurc),
        "macro_aurc": float(macro_aurc),
        "misclass_auroc": float(misclass_auroc),
        "macro_misclass_auroc": float(macro_misclass_auroc),
        "delta_u_median": float(delta_u_micro),   # key name kept for compatibility
        "macro_delta_u": float(macro_delta_u),
        "macro_belief_correctness": float(macro_belief_correctness),
        "evidential_loss": float(evidential_loss),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "f1_per_attack": f1_per_atk.tolist(),
    }


class MetricsPoolHP:
    """Worker pool with one-time data loading + automatic rebuild on BrokenProcessPool."""

    def __init__(self, max_workers, data_dir, splits_to_load, start_method="spawn"):
        self._max_workers = max_workers
        self._data_dir = data_dir
        self._splits = splits_to_load
        self._start_method = start_method
        self._lock = threading.Lock()
        self._pool_version = 0
        self._build_pool()

    def _build_pool(self):
        ctx = mp.get_context(self._start_method)
        self.pool = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self._data_dir, self._splits),
        )
        self._pool_version += 1

    def _rebuild_if_needed(self, version_seen):
        with self._lock:
            if self._pool_version == version_seen:
                try:
                    self.pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._build_pool()
                print(f"[Pool] Rebuilt (version {self._pool_version}).", flush=True)

    def shutdown(self):
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

    def eval(self, alpB, betB, alpD, betD, thr, decision_thr,
             fusion_op, trust, kl_weight, split, timeout_s=TIMEOUT_PER_TRIAL_S):
        version_seen = self._pool_version
        try:
            fut = self.pool.submit(
                _trial_task, alpB, betB, alpD, betD, thr, decision_thr,
                fusion_op, trust, kl_weight, split
            )
        except (BrokenProcessPool, RuntimeError):
            self._rebuild_if_needed(version_seen)
            raise RuntimeError("Pool broken on submit.")
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            fut.cancel()
            raise TimeoutError(f"Trial timed out after {timeout_s}s")
        except BrokenProcessPool:
            self._rebuild_if_needed(version_seen)
            raise RuntimeError("Worker crashed.")


# ============================================================================
# Hyperparameter search space
# ============================================================================

def _suggest_feature(trial, prefix, thr, alpB, alpD, betB, betD, trust,
                     use_trust=USE_TRUST_DISCOUNT):
    thr.append(trial.suggest_float(f"{prefix}_thr", 0.0, 1.0))
    alpB.append(trial.suggest_float(f"{prefix}_alpB", 0.01, 0.99))
    alpD.append(trial.suggest_float(f"{prefix}_alpD", 0.01, 0.99))
    # Upper bound 1e3 instead of 1e9 (GNSS-specific):
    # data_cache_gnss normalizes all four features to [0, 1), so
    # delta_sq = (X-thr)^2 is in [0, 1]. For betB >= 1e3, 1-exp(-bet*delta_sq)
    # already exceeds 0.9999 at |X-thr| = 0.1 - the evidence mapping becomes a
    # step function, u collapses toward 0 and delta_u / misclass_auroc lose
    # their signal. In a log-uniform range [1e-2, 1e9], 6 of 11 decades (55%
    # of the prior mass) would lie entirely in this saturated regime.
    betB.append(trial.suggest_float(f"{prefix}_betB", 0.01, 1_000.0, log=True))
    betD.append(trial.suggest_float(f"{prefix}_betD", 0.01, 1_000.0, log=True))
    if use_trust:
        trust.append(trial.suggest_float(f"{prefix}_trust", 0.0, 1.0))


# Operating point: FIXED at 0.5, NOT an Optuna search parameter.
#  (a) p >= 0.5  <=>  b >= d  is the canonical SL decision rule.
#  (b) A free decision_thr would let NSGA-II tune the operating point against
#      exactly the uncertainty objectives the paper compares - a degree of
#      freedom the MLP/B-spline models do not have.
DECISION_THR_FIXED = sl_main.DECISION_THR

# Lambda in Sensoy's evidential loss L = MSE + lambda*KL. Only feeds the
# diagnostic user_attr "evidential_loss"; it has no effect on predictions or
# on any of the 5 objectives. The single-objective ablation sweeps it.
KL_WEIGHT_FIXED = 1.0


def _build_trial_params(trial):
    """Returns alpB, betB, alpD, betD, thr, trust, kl_weight, decision_thr, fusion_op."""
    alpB, alpD, betB, betD, thr, trust = [], [], [], [], [], []
    for prefix in FEATURE_ORDER:
        _suggest_feature(trial, prefix, thr, alpB, alpD, betB, betD, trust)
    decision_thr = DECISION_THR_FIXED
    fusion_op = trial.suggest_categorical("fusion_op", FUSION_OPTIONS)
    kl_weight = KL_WEIGHT_FIXED
    trust_out = trust if USE_TRUST_DISCOUNT else None
    return alpB, betB, alpD, betD, thr, trust_out, kl_weight, decision_thr, fusion_op


def _params_from_trial_params(trial_params_dict):
    """
    Reconstructs alpB, betB, alpD, betD, thr, trust, decision_thr, fusion_op
    from a trial.params dict (to replay the selected trial in the final test).
    The decision threshold is always DECISION_THR_FIXED.
    """
    alpB, alpD, betB, betD, thr, trust = [], [], [], [], [], []
    for prefix in FEATURE_ORDER:
        thr.append(trial_params_dict[f"{prefix}_thr"])
        alpB.append(trial_params_dict[f"{prefix}_alpB"])
        alpD.append(trial_params_dict[f"{prefix}_alpD"])
        betB.append(trial_params_dict[f"{prefix}_betB"])
        betD.append(trial_params_dict[f"{prefix}_betD"])
        if USE_TRUST_DISCOUNT:
            trust.append(trial_params_dict[f"{prefix}_trust"])
    decision_thr = DECISION_THR_FIXED
    fusion_op = trial_params_dict["fusion_op"]
    trust_out = trust if USE_TRUST_DISCOUNT else None
    return alpB, betB, alpD, betD, thr, trust_out, decision_thr, fusion_op


# ============================================================================
# Knee-point selection (identical to B-spline/MLP)
# ============================================================================

METRICS_FOR_KNEE = [
    ("macro_f1",                 "maximize"),
    ("macro_aurc",               "minimize"),
    ("macro_misclass_auroc",     "maximize"),
    ("macro_delta_u",            "maximize"),
    ("macro_belief_correctness", "maximize"),
]
TIEBREAKER_PRIORITY = ["chebyshev", "closest_to_utopia",
                       "weighted_sum", "farthest_from_nadir"]

# Optional constraint filter applied BEFORE the knee-point selection, as a
# list of (metric, op, threshold) tuples, e.g. ("macro_f1", ">=", 0.5).
# Empty: the knee-point selection runs on all completed trials.
KNEE_CONSTRAINTS = []
# True: an unreachable constraint aborts instead of silently falling back to
# the unfiltered trials (which would disable ALL constraints, not just the
# violated one).
KNEE_FALLBACK_HARD = True


def normalize_trials_for_knee(trials_metrics):
    """Min-max normalizes every objective to [0, 1] with 1 = best."""
    n = len(trials_metrics)
    m = len(METRICS_FOR_KNEE)
    Z = np.zeros((n, m), dtype=np.float64)
    for j, (key, direction) in enumerate(METRICS_FOR_KNEE):
        vals = np.array([p[key] for p in trials_metrics], dtype=float)
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-12:
            Z[:, j] = 0.5
        else:
            x = (vals - lo) / (hi - lo)
            Z[:, j] = x if direction == "maximize" else 1.0 - x
    return Z


def knee_closest_to_utopia(Z):
    return int(np.argmin(np.linalg.norm(Z - np.ones(Z.shape[1]), axis=1)))


def knee_farthest_from_nadir(Z):
    return int(np.argmax(np.linalg.norm(Z, axis=1)))


def knee_chebyshev(Z):
    return int(np.argmin(np.max(np.abs(Z - np.ones(Z.shape[1])), axis=1)))


def knee_weighted_sum(Z):
    w = np.ones(Z.shape[1]) / Z.shape[1]
    return int(np.argmax(Z @ w))


KNEE_METHODS = {
    "closest_to_utopia":   knee_closest_to_utopia,
    "farthest_from_nadir": knee_farthest_from_nadir,
    "chebyshev":           knee_chebyshev,
    "weighted_sum":        knee_weighted_sum,
}


def _apply_knee_constraints(trials_data, constraints, verbose=True):
    """
    Filters trials_data by the constraints. If no trial survives: abort
    (KNEE_FALLBACK_HARD) or fall back to the unfiltered trials with a warning.
    """
    if not constraints:
        return trials_data, False
    keep = []
    for t in trials_data:
        ok = True
        for metric, op, threshold in constraints:
            val = t.get(metric)
            if val is None:
                ok = False; break
            if op == ">=" and val < threshold: ok = False; break
            if op == ">"  and val <= threshold: ok = False; break
            if op == "<=" and val > threshold: ok = False; break
            if op == "<"  and val >= threshold: ok = False; break
            if op == "==" and val != threshold: ok = False; break
        if ok:
            keep.append(t)

    n_before = len(trials_data); n_after = len(keep)
    if verbose:
        print("\n[Knee constraints]")
        for metric, op, threshold in constraints:
            print(f"  {metric} {op} {threshold}")
        print(f"  Trials before filter: {n_before}")
        print(f"  Trials after filter:  {n_after}")

    if n_after == 0:
        msg = ("Knee constraints too strict - no trial left. A silent fallback "
               "would disable ALL constraints, not just the violated one. "
               "Re-set KNEE_CONSTRAINTS based on the observed Pareto front "
               "(sweep_summary.csv) and re-run with --analyze-only.")
        if KNEE_FALLBACK_HARD:
            raise RuntimeError(msg)
        if verbose:
            print(f"  [WARN] {msg}")
            print("  [WARN] Fallback: ignoring constraints, using unfiltered trials.")
        return trials_data, True   # True = fallback used
    return keep, False


def select_balanced_trial(study, verbose=True):
    """
    Applies the constraint filter + 4 knee methods + majority vote to all
    COMPLETED trials. Returns (winner_trial_number, info_dict).
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if not completed:
        raise RuntimeError("No completed trials.")
    trials_data = []
    for t in completed:
        md = {name: t.values[j] for j, name in enumerate(OBJECTIVE_NAMES)}
        md["trial_number"] = t.number
        trials_data.append(md)

    trials_filtered, fallback_used = _apply_knee_constraints(
        trials_data, KNEE_CONSTRAINTS, verbose=verbose
    )

    Z = normalize_trials_for_knee(trials_filtered)
    knee_results = {}
    if verbose:
        print(f"\n{'='*70}")
        print(f"KNEE-POINT SELECTION ({len(trials_filtered)} trials)")
        print(f"{'='*70}")
    for name, fn in KNEE_METHODS.items():
        idx = fn(Z)
        tnum = trials_filtered[idx]["trial_number"]
        knee_results[name] = {
            "trial_number": tnum,
            "metrics": {k: trials_filtered[idx][k] for k, _ in METRICS_FOR_KNEE},
        }
        if verbose:
            print(f"\n[{name}]  → Trial {tnum}")
            for k, _ in METRICS_FOR_KNEE:
                print(f"    {k:30s} = {trials_filtered[idx][k]:.4f}")

    votes = Counter(r["trial_number"] for r in knee_results.values())
    methods_per_trial = defaultdict(list)
    for mname, r in knee_results.items():
        methods_per_trial[r["trial_number"]].append(mname)
    sorted_votes = votes.most_common()
    top_count = sorted_votes[0][1]
    top_trials = [tn for tn, c in sorted_votes if c == top_count]
    info = {
        "constraints": [{"metric": m, "op": op, "threshold": th}
                        for m, op, th in KNEE_CONSTRAINTS],
        "constraint_fallback_used": fallback_used,
        "n_trials_before_filter":   len(trials_data),
        "n_trials_after_filter":    len(trials_filtered),
        "votes_per_trial":    dict(votes),
        "methods_per_trial":  {k: v for k, v in methods_per_trial.items()},
        "n_methods_total":    len(knee_results),
        "tie_breaker_used":   False,
        "tie_breaker_method": None,
    }
    if len(top_trials) == 1:
        winner = top_trials[0]
    else:
        info["tie_breaker_used"] = True
        info["tied_trials"] = top_trials
        winner = None
        for tb in TIEBREAKER_PRIORITY:
            if tb in knee_results and knee_results[tb]["trial_number"] in top_trials:
                winner = knee_results[tb]["trial_number"]
                info["tie_breaker_method"] = tb
                break
        if winner is None:
            winner = top_trials[0]
    info["winner_trial"] = winner
    info["n_votes"] = top_count
    info["consensus_strength"] = top_count / len(knee_results)
    info["knee_results"] = knee_results
    if verbose:
        print(f"\n{'='*70}")
        print(f"CONSENSUS WINNER: Trial {winner}  "
              f"({top_count}/{len(knee_results)} = {100*info['consensus_strength']:.0f}%)")
        if info["tie_breaker_used"]:
            print(f"  Tiebreaker: {info['tie_breaker_method']}")
        print(f"{'='*70}")
    return winner, info


# ============================================================================
# Extended boxplot renderer (identical to B-spline/MLP)
# ============================================================================

def _render_extended_boxplot(boxplot_samples, out_path, title_suffix=""):
    """
    Boxplot with 12 categories, grouped by SL component:
        Group U:  u_benign | u_malicious | u_correct | u_wrong
        Group B:  b_benign | b_malicious | b_correct | b_wrong
        Group D:  d_benign | d_malicious | d_correct | d_wrong
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = {
        "u_benign":    "#5DA0CB", "u_malicious": "#1C4E80",
        "u_correct":   "#3CB371", "u_wrong":     "#A5292A",
        "b_benign":    "#A07ABF", "b_malicious": "#5E3A8F",
        "b_correct":   "#3CB371", "b_wrong":     "#A5292A",
        "d_benign":    "#D4A24C", "d_malicious": "#7A5400",
        "d_correct":   "#3CB371", "d_wrong":     "#A5292A",
    }
    order = [
        ("u_benign",    "u — benign"),
        ("u_malicious", "u — malicious"),
        ("u_correct",   "u — correct"),
        ("u_wrong",     "u — misclassified"),
        ("b_benign",    "b — benign"),
        ("b_malicious", "b — malicious"),
        ("b_correct",   "b — correct"),
        ("b_wrong",     "b — misclassified"),
        ("d_benign",    "d — benign"),
        ("d_malicious", "d — malicious"),
        ("d_correct",   "d — correct"),
        ("d_wrong",     "d — misclassified"),
    ]
    BOX_SPACING = 1.0
    GROUP_GAP = 1.2
    positions, data, box_colors, labels = [], [], [], []
    cursor = 1.0
    for grp_start in [0, 4, 8]:
        for k in range(4):
            key, lbl = order[grp_start + k]
            arr = np.asarray(boxplot_samples.get(key, []), dtype=np.float64)
            if arr.size == 0:
                arr = np.array([np.nan])
            data.append(arr); positions.append(cursor)
            box_colors.append(colors[key]); labels.append(lbl)
            cursor += BOX_SPACING
        cursor += GROUP_GAP

    v_data, v_pos, v_col, v_lbl = [], [], [], []
    for arr, pos, col, lbl in zip(data, positions, box_colors, labels):
        if np.isnan(arr).all():
            continue
        v_data.append(arr); v_pos.append(pos); v_col.append(col); v_lbl.append(lbl)

    bp = ax.boxplot(
        v_data, positions=v_pos, widths=0.7,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    for patch, color in zip(bp["boxes"], v_col):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    ax.set_xticks(v_pos)
    ax.set_xticklabels(v_lbl, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("SL opinion value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Extended SL-Opinion Distribution {title_suffix}".strip())
    for sep_x in [v_pos[3] + 0.6, v_pos[7] + 0.6]:
        ax.axvline(sep_x, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(out_path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close()


# ============================================================================
# Final test run: Tables 1-5 + per-sample CSV + extended boxplot + TS
# ============================================================================

def _f1_attacker(y_true, y_pred):
    tp = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp) / denom if denom > 0 else 0.0


def _f1_attacker_macro(y_true, y_pred, attack_idx, n_attacks):
    """
    Macro F1 (attacker = positive class), averaged per attack.

    DELIBERATELY bit-identical to the worker logic in _trial_task: bincount,
    denom == 0 -> 0.0, mean over ALL n_attacks (missing groups are not
    skipped). Only this way is the reported macro_f1 exactly the sweep
    objective.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    is_atk = (y_true == 0); pred_atk = (y_pred == 0); is_ben = ~is_atk
    tp_arr = is_atk & pred_atk
    fp_arr = is_ben & pred_atk
    fn_arr = is_atk & ~pred_atk
    tp = np.bincount(attack_idx, weights=tp_arr.astype(np.float64), minlength=n_attacks)
    fp = np.bincount(attack_idx, weights=fp_arr.astype(np.float64), minlength=n_attacks)
    fn = np.bincount(attack_idx, weights=fn_arr.astype(np.float64), minlength=n_attacks)
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, (2.0 * tp) / np.maximum(denom, 1e-9), 0.0)
    return float(f1.mean())


def build_paper_outputs_final(p_all, u_all, b_all, d_all,
                              y_pred_all, y_true_all,
                              attack_all, method_name, out_dir,
                              attack_idx_all=None, boxplot_n=50_000):
    """
    Same structure as in the B-spline/MLP sweeps:
    per-sample CSV + Tables 1-5 + boxplot + extended boxplot + results JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    per_attack_results = {}
    pooled_idx_list = []

    csv_path = os.path.join(out_dir, f"opinions_per_sample_{method_name}.csv")
    with open(csv_path, "w", encoding="utf-8") as fcsv:
        fcsv.write("attack,y_true,y_pred,correct,p,b,d,u\n")
        for atk in selected_attacks:
            mask = (attack_all == atk)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            y_t = y_true_all[idx]
            yp_t = y_pred_all[idx]
            correct_t = (yp_t == y_t)
            p_t = p_all[idx]; b_t = b_all[idx]; d_t = d_all[idx]; u_t = u_all[idx]

            for i in range(len(idx)):
                fcsv.write(f"{atk},{int(y_t[i])},{int(yp_t[i])},{int(correct_t[i])},"
                           f"{p_t[i]:.6f},{b_t[i]:.6f},{d_t[i]:.6f},{u_t[i]:.6f}\n")

            f1 = _f1_attacker(y_t, yp_t)
            m = metrics.evaluate_all(y_true=y_t, y_prob=p_t, y_pred=yp_t, uncertainty=u_t)

            mask_benign    = (y_t == 1)
            mask_malicious = (y_t == 0)
            u_c = u_t[correct_t]; u_w = u_t[~correct_t]
            b_c = b_t[correct_t]; b_w = b_t[~correct_t]
            d_c = d_t[correct_t]; d_w = d_t[~correct_t]
            u_bn = u_t[mask_benign]; u_ml = u_t[mask_malicious]
            b_bn = b_t[mask_benign]; b_ml = b_t[mask_malicious]
            d_bn = d_t[mask_benign]; d_ml = d_t[mask_malicious]

            mask_bc = mask_benign & correct_t
            mask_ac = mask_malicious & correct_t
            bcb = float(b_t[mask_bc].mean()) if mask_bc.any() else float("nan")
            dca = float(d_t[mask_ac].mean()) if mask_ac.any() else float("nan")
            belief_corr = (bcb + dca) / 2.0 if not (np.isnan(bcb) or np.isnan(dca)) else float("nan")

            def _mm(a):
                return (float(a.mean()), float(np.median(a))) if len(a) else (float("nan"), float("nan"))
            def _stat(a):
                if len(a) == 0:
                    return float("nan"), float("nan"), float("nan")
                return float(a.mean()), float(np.median(a)), float(a.std())

            uc_m, uc_md, uc_sd = _stat(u_c)
            uw_m, uw_md, uw_sd = _stat(u_w)
            ubn_m, ubn_md = _mm(u_bn);  uml_m, uml_md = _mm(u_ml)
            bc_m, bc_md = _mm(b_c);     bw_m, bw_md = _mm(b_w)
            bbn_m, bbn_md = _mm(b_bn);  bml_m, bml_md = _mm(b_ml)
            dc_m, dc_md = _mm(d_c);     dw_m, dw_md = _mm(d_w)
            dbn_m, dbn_md = _mm(d_bn);  dml_m, dml_md = _mm(d_ml)

            per_attack_results[atk] = {
                "f1": f1,
                "aurc": m.get("aurc", float("nan")),
                "misclass_auroc": m.get("misclass_auroc", float("nan")),
                "ece": m.get("ece", float("nan")),
                "brier": m.get("brier", float("nan")),
                "u_correct_mean": uc_m, "u_correct_median": uc_md, "u_correct_std": uc_sd,
                "u_wrong_mean":   uw_m, "u_wrong_median":   uw_md, "u_wrong_std":   uw_sd,
                "u_benign_mean":    ubn_m, "u_benign_median":    ubn_md,
                "u_malicious_mean": uml_m, "u_malicious_median": uml_md,
                "b_correct_mean": bc_m, "b_correct_median": bc_md,
                "b_wrong_mean":   bw_m, "b_wrong_median":   bw_md,
                "b_benign_mean":    bbn_m, "b_benign_median":    bbn_md,
                "b_malicious_mean": bml_m, "b_malicious_median": bml_md,
                "d_correct_mean": dc_m, "d_correct_median": dc_md,
                "d_wrong_mean":   dw_m, "d_wrong_median":   dw_md,
                "d_benign_mean":    dbn_m, "d_benign_median":    dbn_md,
                "d_malicious_mean": dml_m, "d_malicious_median": dml_md,
                "b_correct_benign_mean":   bcb,
                "d_correct_attacker_mean": dca,
                "belief_correctness":      belief_corr,
                "n_correct": int(correct_t.sum()),
                "n_wrong":   int((~correct_t).sum()),
                "n_benign":    int(mask_benign.sum()),
                "n_malicious": int(mask_malicious.sum()),
            }
            pooled_idx_list.append(idx)
    print(f"  per-sample CSV: {csv_path}")

    if not pooled_idx_list:
        return None

    pooled_idx = np.concatenate(pooled_idx_list)
    y_pool = y_true_all[pooled_idx]
    p_pool = p_all[pooled_idx]
    u_pool = u_all[pooled_idx]
    b_pool = b_all[pooled_idx]
    d_pool = d_all[pooled_idx]
    yp_pool = y_pred_all[pooled_idx]
    correct_pool = (yp_pool == y_pool)
    benign_pool = (y_pool == 1)
    malicious_pool = (y_pool == 0)
    m_pool = metrics.evaluate_all(y_true=y_pool, y_prob=p_pool, y_pred=yp_pool, uncertainty=u_pool)
    f1_micro = _f1_attacker(y_pool, yp_pool)

    u_c_pool = u_pool[correct_pool]; u_w_pool = u_pool[~correct_pool]
    b_c_pool = b_pool[correct_pool]; b_w_pool = b_pool[~correct_pool]
    d_c_pool = d_pool[correct_pool]; d_w_pool = d_pool[~correct_pool]
    u_bn_pool = u_pool[benign_pool];    u_ml_pool = u_pool[malicious_pool]
    b_bn_pool = b_pool[benign_pool];    b_ml_pool = b_pool[malicious_pool]
    d_bn_pool = d_pool[benign_pool];    d_ml_pool = d_pool[malicious_pool]

    def _ric(arr_u, arr_correct, cov):
        n = len(arr_u)
        if n == 0: return float("nan")
        k = max(1, int(cov * n))
        order = np.argsort(arr_u)
        errors = (~arr_correct[order]).astype(float)
        return float(errors[:k].mean())

    overall_micro = {
        "f1": f1_micro,
        "aurc": m_pool.get("aurc", float("nan")),
        "misclass_auroc": m_pool.get("misclass_auroc", float("nan")),
        "ece": m_pool.get("ece", float("nan")),
        "brier": m_pool.get("brier", float("nan")),
        "risk_at_70":  _ric(u_pool, correct_pool, 0.70),
        "risk_at_80":  _ric(u_pool, correct_pool, 0.80),
        "risk_at_90":  _ric(u_pool, correct_pool, 0.90),
        "risk_at_100": float(1.0 - correct_pool.mean()) if len(correct_pool) else float("nan"),
        "delta_u_mean":   float(u_w_pool.mean() - u_c_pool.mean())
                          if len(u_c_pool) and len(u_w_pool) else float("nan"),
        "delta_u_median": float(np.median(u_w_pool) - np.median(u_c_pool))
                          if len(u_c_pool) and len(u_w_pool) else float("nan"),
    }

    def _macro(key):
        vs = [pa[key] for pa in per_attack_results.values()
              if not (isinstance(pa.get(key), float) and np.isnan(pa[key]))]
        return float(np.mean(vs)) if vs else float("nan")

    # Must use the same statistic as the sweep objective; otherwise selection
    # and reporting would use different quantities.
    _wk, _ck = f"u_wrong_{DELTA_U_STAT}", f"u_correct_{DELTA_U_STAT}"
    delta_u_per_attack = [
        pa[_wk] - pa[_ck]
        for pa in per_attack_results.values()
        if not np.isnan(pa.get(_wk, float("nan")))
        and not np.isnan(pa.get(_ck, float("nan")))
    ]
    # Compute macro_f1 bit-identically to the sweep objective (mean over ALL
    # N_ATTACKS, a missing group counts as 0.0). _macro("f1") would only
    # average over present groups -> a different value from the one selected on.
    if attack_idx_all is not None:
        macro_f1_val = _f1_attacker_macro(
            y_true_all[pooled_idx], y_pred_all[pooled_idx],
            np.asarray(attack_idx_all)[pooled_idx], N_ATTACKS,
        )
    else:
        macro_f1_val = _macro("f1")
    macro = {
        "macro_f1":                 macro_f1_val,
        "macro_aurc":               _macro("aurc"),
        "macro_misclass_auroc":     _macro("misclass_auroc"),
        "macro_ece":                _macro("ece"),
        "macro_brier":              _macro("brier"),
        "macro_delta_u":            float(np.mean(delta_u_per_attack)) if delta_u_per_attack else float("nan"),
        "macro_belief_correctness": _macro("belief_correctness"),
    }

    def _sample(arr, n):
        if len(arr) <= n: return arr
        return rng.choice(arr, size=n, replace=False)

    method_entry = {
        "method_name":   method_name,
        "macro":         macro,
        "overall_micro": overall_micro,
        "per_attack":    per_attack_results,
        "boxplot_samples": {
            "u_correct":   np.asarray(_sample(u_c_pool, boxplot_n)),
            "u_wrong":     np.asarray(_sample(u_w_pool, boxplot_n)),
            "u_benign":    np.asarray(_sample(u_bn_pool, boxplot_n)),
            "u_malicious": np.asarray(_sample(u_ml_pool, boxplot_n)),
            "b_correct":   np.asarray(_sample(b_c_pool, boxplot_n)),
            "b_wrong":     np.asarray(_sample(b_w_pool, boxplot_n)),
            "b_benign":    np.asarray(_sample(b_bn_pool, boxplot_n)),
            "b_malicious": np.asarray(_sample(b_ml_pool, boxplot_n)),
            "d_correct":   np.asarray(_sample(d_c_pool, boxplot_n)),
            "d_wrong":     np.asarray(_sample(d_w_pool, boxplot_n)),
            "d_benign":    np.asarray(_sample(d_bn_pool, boxplot_n)),
            "d_malicious": np.asarray(_sample(d_ml_pool, boxplot_n)),
        },
    }

    json_out = {
        **{k: v for k, v in method_entry.items() if k != "boxplot_samples"},
        "boxplot_samples": {k: v.tolist() for k, v in method_entry["boxplot_samples"].items()},
    }
    json_path = os.path.join(out_dir, f"results_paper_{method_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"  results JSON:   {json_path}")

    paper_outputs.write_all_tables_and_boxplot(
        methods=[method_entry],
        attacks=selected_attacks,
        out_dir=out_dir,
        suffix=method_name,
        statistic=DELTA_U_STAT,
    )

    ext_box_path = os.path.join(out_dir, f"boxplot_opinions_extended_{method_name}.png")
    _render_extended_boxplot(
        boxplot_samples=method_entry["boxplot_samples"],
        out_path=ext_box_path,
        title_suffix=f"({method_name})",
    )
    print(f"  extended boxplot: {ext_box_path}")
    print(f"\n  Macro F1: {macro['macro_f1']:.4f}   AURC: {macro['macro_aurc']:.4f}   "
          f"MAUROC: {macro['macro_misclass_auroc']:.4f}   Δu: {macro['macro_delta_u']:.4f}   "
          f"BeliefCorr: {macro['macro_belief_correctness']:.4f}")
    return method_entry


def run_final_test(study, winner_trial, data_dir, out_dir,
                   method_prefix="HP_balanced"):
    """
    Forward pass on the test split with the HPs of the selected trial at the
    fixed threshold 0.5, then paper outputs + model JSON.
    """
    print(f"\n{'='*70}")
    print(f"FINAL TEST RUN (SL-HP, {method_prefix}): Trial {winner_trial.number}")
    print(f"{'='*70}")

    params = winner_trial.params
    alpB, betB, alpD, betD, thr, trust, _decision_thr, fusion_op = \
        _params_from_trial_params(params)

    # Load Val + Test in the main process, restricted to selected_attacks
    # EXACTLY as in the sweep (_init_worker), so the diagnostics below run on
    # the same sample set as the sweep objectives.
    print("\n[Data] Loading Validation + Test ...")
    msgs_va, _attack_va, attack_idx_va, _, n_drop_va = load_filtered_split(
        "Validation", data_dir)
    msgs_te, attack_te, attack_idx_te, _, n_drop_te = load_filtered_split(
        "Test", data_dir)
    print(f"  Val:  {msgs_va.shape[0]:,} samples ({n_drop_va:,} dropped)")
    print(f"  Test: {msgs_te.shape[0]:,} samples ({n_drop_te:,} dropped)")
    print(f"  {N_ATTACKS} attack groups: {selected_attacks}")

    # Forward on Val
    b_va, d_va, u_va, truth_va = sl_main.compute_opinions(
        msgs_va, alpB, betB, alpD, betD, thr, fusion_op=fusion_op, trust=trust
    )
    p_va = b_va + 0.5 * u_va

    # NO threshold refit. The sweep evaluated all five objectives at 0.5; a
    # refit would shift the correct/wrong partition and collapse exactly the
    # uncertainty metrics that were selected on.
    y_va_flat = truth_va.astype(np.int64)
    t_star = float(DECISION_THR_FIXED)
    macro_f1_va = _f1_attacker_macro(y_va_flat, (p_va >= t_star).astype(int),
                                     attack_idx_va, N_ATTACKS)
    print(f"  Fixed threshold: t* = {t_star:.3f}  "
          f"(Val macro-F1 = {macro_f1_va:.4f}, micro-F1 = "
          f"{_f1_attacker(y_va_flat, (p_va >= t_star).astype(int)):.4f})")
    # Diagnostic: how much mass lies directly on the decision boundary?
    near = float(np.mean(np.abs(p_va - t_star) < 1e-6))
    band = float(np.mean((p_va >= 0.48) & (p_va < 0.52)))
    print(f"  Val mass exactly at t* (vacuous opinions): {100*near:.2f}%  |  "
          f"in [0.48, 0.52): {100*band:.2f}%")
    if band > 0.10:
        print("  [WARN] >10% of the samples lie within the 0.04 band around t*. "
              "The operating point is a knife edge - small changes to "
              "features/HPs flip many predictions at once.")

    # Forward on Test
    b_te, d_te, u_te, truth_te = sl_main.compute_opinions(
        msgs_te, alpB, betB, alpD, betD, thr, fusion_op=fusion_op, trust=trust
    )
    p_te = (b_te + 0.5 * u_te).astype(np.float32)
    b_te_f = b_te.astype(np.float32)
    d_te_f = d_te.astype(np.float32)
    u_te_f = u_te.astype(np.float32)
    y_true_te = truth_te.astype(np.int64)
    y_pred_te = (p_te >= t_star).astype(int)

    method_name = f"{method_prefix}_trial{winner_trial.number:03d}_{fusion_op}"
    build_paper_outputs_final(
        p_all=p_te, u_all=u_te_f, b_all=b_te_f, d_all=d_te_f,
        y_pred_all=y_pred_te, y_true_all=y_true_te,
        attack_all=attack_te, attack_idx_all=attack_idx_te,
        method_name=method_name, out_dir=out_dir,
    )

    # Save the "model" — for the closed-form SL these are the per-feature
    # parameters + fusion operator + decision_thr
    model = {
        "variant": "sl_hp_per_feature",
        "feature_order": FEATURE_ORDER,
        "feature_normalization": dc.NORMALIZE,
        "alpB": list(map(float, alpB)),
        "betB": list(map(float, betB)),
        "alpD": list(map(float, alpD)),
        "betD": list(map(float, betD)),
        "thr":  list(map(float, thr)),
        "trust": list(map(float, trust)) if trust is not None else None,
        "fusion_op": fusion_op,
        "decision_thr": t_star,
    }
    model_path = os.path.join(out_dir, f"sl_hp_trust_model_{method_name}.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"\n  Model:           {model_path}")


# ============================================================================
# Optuna study + CSV/plots
# ============================================================================

def create_or_load_study(resume=False):
    # Search space: 4 features x 5 parameters (thr, alpB, alpD, betB, betD)
    # + fusion_op = 21 dimensions (decision_thr fixed, trust disabled).
    sampler = optuna.samplers.NSGAIISampler(
        population_size=50,
        crossover=optuna.samplers.nsgaii.UniformCrossover(),
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=42,
    )
    if resume:
        try:
            study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB,
                                      sampler=sampler)
            if len(study.directions) != len(OBJECTIVE_DIRECTION):
                raise RuntimeError(
                    f"Schema mismatch! Study has {len(study.directions)} objectives, "
                    f"expected {len(OBJECTIVE_DIRECTION)}.")
            print(f"[Optuna] Study loaded — {len(study.trials)} trials so far.")
            return study
        except KeyError:
            print("[Optuna] No existing study, creating a new one.")
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB,
        directions=OBJECTIVE_DIRECTION, sampler=sampler,
        load_if_exists=True,
    )
    if len(study.directions) != len(OBJECTIVE_DIRECTION):
        raise RuntimeError(
            f"Schema mismatch on creation! "
            f"Fix: delete the DB file from {STUDY_DB} or change STUDY_NAME.")
    return study


def write_summary_csv(study, path):
    pareto_set = {t.number for t in study.best_trials}
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": t.number,
            "is_pareto":    t.number in pareto_set,
            "duration_min": t.user_attrs.get("duration_min", 0),
        }
        for k, v in t.params.items():
            row[k] = v
        for i, name in enumerate(OBJECTIVE_NAMES):
            row[name] = t.values[i] if t.values else float("nan")
        rows.append(row)
    if not rows:
        return
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  CSV: {path}")


def plot_pareto_2d(study, obj_x, obj_y, out_path):
    idx_x = OBJECTIVE_NAMES.index(obj_x)
    idx_y = OBJECTIVE_NAMES.index(obj_y)
    dir_x = OBJECTIVE_DIRECTION[idx_x]
    dir_y = OBJECTIVE_DIRECTION[idx_y]
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    pareto_set = {t.number for t in study.best_trials}
    fig, ax = plt.subplots(figsize=(9, 6))
    np_x = [t.values[idx_x] for t in completed if t.number not in pareto_set]
    np_y = [t.values[idx_y] for t in completed if t.number not in pareto_set]
    p_x = [t.values[idx_x] for t in completed if t.number in pareto_set]
    p_y = [t.values[idx_y] for t in completed if t.number in pareto_set]
    if np_x:
        ax.scatter(np_x, np_y, c="#888888", s=40, alpha=0.4,
                   label=f"dominated (n={len(np_x)})")
    if p_x:
        sorted_p = sorted(zip(p_x, p_y), key=lambda xy: xy[0])
        sx = [xy[0] for xy in sorted_p]; sy = [xy[1] for xy in sorted_p]
        ax.scatter(sx, sy, c="#D85A30", s=80, alpha=0.95, edgecolors="black",
                   linewidths=1.0, label=f"Pareto (n={len(p_x)})", zorder=3)
        ax.plot(sx, sy, c="#D85A30", alpha=0.4, linestyle="--", zorder=2)
    ax.set_xlabel(f"{obj_x}  ({'↑' if dir_x=='maximize' else '↓'})")
    ax.set_ylabel(f"{obj_y}  ({'↑' if dir_y=='maximize' else '↓'})")
    ax.set_title(f"SL-HP Pareto: {obj_x} vs {obj_y}")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=40000,
                        help="Total number of completed trials to reach (cumulative)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count // 2)")
    parser.add_argument("--data-dir", "--cache-dir", dest="data_dir",
                        default=dc.DATA_DIR_DEFAULT,
                        help="Directory with oneclass_sl_{train,val,test}.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where CSVs, Pareto plots, balanced_trial.json and "
                             "final_test/ are written")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the existing study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip the sweep; only knee selection + final test")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Sweep SL-HP] Objectives: {OBJECTIVE_NAMES}")
    print(f"[Sweep SL-HP] Directions: {OBJECTIVE_DIRECTION}")
    print(f"[Sweep SL-HP] Output dir: {args.output_dir}/")
    print(f"[Sweep SL-HP] Data dir:   {args.data_dir}")

    if not dc.data_exists(data_dir=args.data_dir):
        print(f"\n[ERROR] GNSS split CSVs missing in {args.data_dir}/")
        print("        Expected oneclass_sl_{train,val,test}.csv from "
              "detection-systems/gds/detector_oneclass.py")
        sys.exit(1)

    # The workers only load Validation during the sweep (the closed-form model
    # has no training step — the HPs define the function directly).
    splits_for_worker = ["Validation"]

    n_workers = args.n_workers if args.n_workers is not None \
                else max(1, (os.cpu_count() or 8) // 2)
    print(f"[Sweep SL-HP] N workers:  {n_workers}")

    study = create_or_load_study(resume=args.resume or args.analyze_only)

    # ---------- Phase 1: sweep ----------
    if not args.analyze_only:
        n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_remaining = args.n_trials - n_completed
        if n_remaining <= 0:
            print(f"\n[Sweep] Target ({args.n_trials}) already reached ({n_completed} trials).")
        else:
            print(f"\n[Sweep] {n_completed} done, running {n_remaining} more trials")
            pool = MetricsPoolHP(
                max_workers=n_workers, data_dir=args.data_dir,
                splits_to_load=splits_for_worker,
            )

            def _safe(value, direction):
                if value is None:
                    return 1.0 if direction == "minimize" else -1.0
                v = float(value)
                if np.isnan(v):
                    return 1.0 if direction == "minimize" else -1.0
                return v

            def objective(trial):
                alpB, betB, alpD, betD, thr, trust, kl_weight, decision_thr, fusion_op = \
                    _build_trial_params(trial)
                t_start = time.time()
                try:
                    result = pool.eval(
                        alpB, betB, alpD, betD, thr, decision_thr,
                        fusion_op=fusion_op, trust=trust, kl_weight=kl_weight,
                        split="Validation",
                    )
                except (RuntimeError, MemoryError, TimeoutError) as e:
                    print(f"  [WARN] Trial {trial.number} failed: {e}")
                    raise optuna.TrialPruned()
                dt = time.time() - t_start

                for key in ["f1", "macro_f1", "aurc", "macro_aurc",
                            "misclass_auroc", "macro_misclass_auroc",
                            "delta_u_median", "macro_delta_u",
                            "macro_belief_correctness", "evidential_loss"]:
                    if key in result:
                        trial.set_user_attr(key, result[key])
                trial.set_user_attr("fusion_op", fusion_op)
                trial.set_user_attr("duration_min", dt / 60)

                return tuple(
                    _safe(result.get(name), direction)
                    for name, direction in zip(OBJECTIVE_NAMES, OBJECTIVE_DIRECTION)
                )

            t_sweep = time.time()
            try:
                study.optimize(
                    objective, n_trials=n_remaining,
                    n_jobs=n_workers,
                    catch=(RuntimeError, MemoryError, TimeoutError),
                )
            finally:
                pool.shutdown()
            print(f"\n[Sweep] Total sweep time: {(time.time()-t_sweep)/60:.1f} min")

    # ---------- Phase 2: analysis ----------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[ERROR] No successful trials — aborting.")
        return
    print(f"\n{'='*70}")
    print(f"SWEEP ANALYSIS  ({len(completed)} completed trials)")
    print(f"{'='*70}")
    print(f"  Pareto front: {len(study.best_trials)} trials")
    write_summary_csv(study, os.path.join(args.output_dir, "sweep_summary.csv"))
    for i in range(len(OBJECTIVE_NAMES)):
        for j in range(i + 1, len(OBJECTIVE_NAMES)):
            plot_pareto_2d(
                study, OBJECTIVE_NAMES[i], OBJECTIVE_NAMES[j],
                os.path.join(args.output_dir,
                             f"pareto_{OBJECTIVE_NAMES[i]}_vs_{OBJECTIVE_NAMES[j]}.png"),
            )

    # ---------- Phase 3: knee selection ----------
    winner, info = select_balanced_trial(study, verbose=True)
    balanced_path = os.path.join(args.output_dir, "balanced_trial.json")
    with open(balanced_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"  balanced JSON: {balanced_path}")

    # ---------- Phase 4: final test ----------
    winner_trial = next(t for t in study.trials if t.number == winner)
    final_dir = os.path.join(args.output_dir, "final_test")
    run_final_test(study, winner_trial, data_dir=args.data_dir, out_dir=final_dir)
    print(f"\n[Sweep SL-HP] DONE. Final outputs in {final_dir}/")


if __name__ == "__main__":
    main()
