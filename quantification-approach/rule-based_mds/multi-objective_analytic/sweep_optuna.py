"""
Optuna-based multi-objective sweep for SL-HP (rule-based Subjective Logic
hyperparameters) — V2 with a binary data cache (analogous to
the B-spline / MLP sweeps in ../multi-objective_bspline/ and ../multi-objective_mlp/).

Design:
  - Data is loaded ONCE per worker via data_cache.load_split (.npz) instead of
    re-parsing JSON on every trial.
  - 5th Pareto objective macro_belief_correctness (analogous to the
    B-Spline/MLP-V2 sweeps).
  - Each trial evaluates ONLY on the validation split (no test set, no paper
    outputs are produced during the sweep itself).
  - After all trials complete: a single "balanced" trial is selected via
    4 knee-point methods + majority consensus.
  - Final test run for the balanced trial ONLY: Tables 1-5 + per-sample CSV
    + extended boxplot (12 categories).

Sampler: NSGA-II (as in the B-spline / MLP sweeps),
with population_size=50 instead of 10 — the HP search space has 40
dimensions (8 features x 5 HPs) and needs more diversity in the population.
Kept methodologically consistent across all three V2 sweeps for a fair
comparison between methods.

Usage:
    python data_cache.py --build
    python sweep_optuna.py --n-trials 2000
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

# Local project modules
import main as sl_main
import metrics
import paper_outputs
import data_cache as dc


# ============================================================================
# Configuration
# ============================================================================

# Default output directory, relative to the working directory. Override with
# --output-dir.
DEFAULT_OUTPUT_DIR = "sweep_output"
STUDY_DB = "sqlite:///hp_optuna_v2.db"
# NOTE: bump STUDY_NAME whenever selected_attacks changes. The macro
# objectives are averaged over the attack groups, so trials computed over a
# different attack set are NOT comparable: macro_f1 scales differently, and
# mauroc/delta_u/aurc don't change proportionally either. Running
# --analyze-only on an old DB would select on one attack set and report on
# another — the selection-!=-report bug that DECISION_THR_FIXED fixes, just
# one level up.
#
# The same applies to the feature schema: this study uses the 8 plausibility
# checks in data_cache.FEATURE_ORDER (Optuna parameters range_plaus_thr, ...).
# A --resume/--analyze-only against a study with a different feature schema
# would die with a KeyError inside _params_from_trial_params.
STUDY_NAME = "hp_pareto_v4_5obj_13atk_8feat"

# Five Pareto objectives — ALL computed on the VALIDATION split during the sweep
OBJECTIVE_NAMES = [
    "macro_f1", "macro_aurc", "macro_misclass_auroc",
    "macro_delta_u", "macro_belief_correctness",
]
OBJECTIVE_DIRECTION = ["maximize", "minimize", "maximize", "maximize", "maximize"]

# Attack whitelist (identical across all sweeps)
# timeDelayAttack and trafficCongestionSybil are deliberately excluded: they
# cannot be detected by local MDSs.
selected_attacks = [
    "constantPositionOffset", "randomPositionOffset", "positionMirroring",
    "suddenStop", "accelerationMultiplication", "feignedBraking",
    "constantSpeedOffset", "randomSpeedOffset", "suddenConstantSpeed",
    "zeroSpeedReport", "reversedHeading", "dataReplay", "dosAttack",
]

# MUST match data_cache.FEATURE_ORDER and main.FEATURE_NAMES exactly (names
# AND order): the cache delivers X with 8 columns in precisely this order,
# and parameter index i is applied directly to cache column i.
FEATURE_ORDER = [
    "range_plaus", "pos_plaus", "speed_plaus", "pos_cons",
    "speed_cons", "pos_speed_cons", "pos_head_cons", "intersection",
]

# ---------------------------------------------------------------------------
# Attack grouping — ONE single source of truth for both the sweep AND the
# final test.
#
# Deliberately NOT sorted(set(attack)) taken from the data: n_attacks would
# then depend on which folders happen to be present in the cache, and the
# sweep and the final tables could average over different sets. Note also that
# _f1_attacker_macro counts groups with no attacker samples (tp=0, fn=0) as
# f1 = 0.0 in the mean (no skip): each such group costs 1/n_attacks macro_f1
# for something the model has no control over.
#
# The order follows selected_attacks, not alphabetical order — so that group
# index k means the same thing across all splits and all three methods.
# ---------------------------------------------------------------------------
_ATTACK_TO_IDX = {a: i for i, a in enumerate(selected_attacks)}
N_ATTACKS = len(selected_attacks)


def _attack_filter_and_idx(attack_arr):
    """
    Restrict to selected_attacks.

    Returns
    -------
    keep : (N,) bool   — True for rows whose attack is in selected_attacks
    idx  : (M,) int64  — group index of the retained rows, M = keep.sum()
    n    : int         — always len(selected_attacks), independent of the data
    """
    a = np.asarray(attack_arr)
    keep = np.isin(a, selected_attacks)
    a_keep = a[keep]
    idx = np.zeros(len(a_keep), dtype=np.int64)
    for i, name in enumerate(selected_attacks):
        idx[a_keep == name] = i
    return keep, idx, N_ATTACKS

FUSION_OPTIONS = ["cbf", "avg", "wbf"]

# Trust discount: one reliability weight in [0, 1] per feature.
# DISABLED: trust parameters can drive u up globally and destroy MAUROC. In a
# previous run with Trust=True, the balanced trial landed at
# macro_misclass_auroc = 0.40 (< 0.5, i.e. worse than random).
USE_TRUST_DISCOUNT = False

# Trial defaults
TIMEOUT_PER_TRIAL_S = 900


# ============================================================================
# Worker pool with cache loading
# ============================================================================
# Global worker state (each process gets its own copy)

_DATA_BY_SPLIT = None     # dict: split -> (msgs_np, attack_idx, attack_names)
_CACHE_DIR = None


def _init_worker(cache_dir, splits_to_load):
    """
    Runs ONCE per worker process.
    Loads the requested splits from the .npz cache.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    global _DATA_BY_SPLIT, _CACHE_DIR
    _CACHE_DIR = cache_dir
    _DATA_BY_SPLIT = {}
    for split in splits_to_load:
        X, y, attack, _, _ = dc.load_split(split, cache_dir=cache_dir)

        # Restrict to selected_attacks — identical to the final test.
        keep, attack_idx, _n = _attack_filter_and_idx(attack)
        n_drop = int((~keep).sum())
        if n_drop:
            dropped = sorted(set(np.asarray(attack)[~keep].tolist()))
            print(f"[Worker] {split}: dropped {n_drop:,} of {len(keep):,} samples "
                  f"(not in selected_attacks): {dropped}", flush=True)
        X = X[keep]
        y = y[keep]

        # Reconstruct msgs = [X (N,8), truth (N,1)] as expected by compute_opinions
        msgs = np.concatenate(
            [X.astype(np.float64),
             y.astype(np.float64).reshape(-1, 1)],
            axis=1,
        )
        _DATA_BY_SPLIT[split] = (msgs, attack_idx.astype(np.int32),
                                 list(selected_attacks))


def _belief_correctness_balanced(b, d, truth, y_pred):
    """
    Class-balanced belief_correctness:
        (mean(b | y_true=1, correct) + mean(d | y_true=0, correct)) / 2

    Analogous to the balanced loss in B-Spline/MLP-V2: averages PER CLASS
    instead of PER SAMPLE.
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
    """Macro-average of the balanced belief_correctness, per attack."""
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
    Per trial: forward pass on the requested split (Train/Validation/Test),
    computing 5 macro metrics + micro stats + confusion counts.
    """
    if _DATA_BY_SPLIT is None:
        raise RuntimeError("Worker dataset not initialized")
    if split not in _DATA_BY_SPLIT:
        raise RuntimeError(f"Split '{split}' not loaded in worker")

    msgs, attack_idx, attack_names = _DATA_BY_SPLIT[split]
    n_attacks = len(attack_names)

    b, d, u, truth = sl_main.compute_opinions(
        msgs, alpB, betB, alpD, betD, thr, fusion_op=fusion_op, trust=trust
    )

    p_benign = b + 0.5 * u
    y_pred_benign = (p_benign >= decision_thr).astype(np.int64)   # '>=' as in the MLP baseline
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
    delta_u_mean = metrics._delta_u_mean(u, correct)
    if np.isnan(delta_u_mean):
        delta_u_mean = 0.0

    macro_aurc = metrics._macro_aurc(confidence, correct, attack_idx, n_attacks)
    macro_misclass_auroc = metrics._macro_misclass_auroc(u, correct, attack_idx, n_attacks)
    if np.isnan(macro_misclass_auroc):
        macro_misclass_auroc = 0.5
    macro_delta_u_mean = metrics._macro_delta_u_mean(u, correct, attack_idx, n_attacks)
    if np.isnan(macro_delta_u_mean):
        macro_delta_u_mean = 0.0

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
        "delta_u_mean": float(delta_u_mean),
        "macro_delta_u": float(macro_delta_u_mean),
        "macro_belief_correctness": float(macro_belief_correctness),
        "evidential_loss": float(evidential_loss),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "f1_per_attack": f1_per_atk.tolist(),
    }


class MetricsPoolHP:
    """Worker pool with cache loading + automatic rebuild on BrokenProcessPool."""

    def __init__(self, max_workers, cache_dir, splits_to_load, start_method="spawn"):
        self._max_workers = max_workers
        self._cache_dir = cache_dir
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
            initargs=(self._cache_dir, self._splits),
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
            raise RuntimeError("Pool broke while submitting the task.")
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            fut.cancel()
            raise TimeoutError(f"Trial timed out after {timeout_s}s")
        except BrokenProcessPool:
            self._rebuild_if_needed(version_seen)
            raise RuntimeError("Worker process crashed.")


# ============================================================================
# Hyperparameter suggestion
# ============================================================================

def _suggest_feature(trial, prefix, thr, alpB, alpD, betB, betD, trust,
                     use_trust=USE_TRUST_DISCOUNT):
    thr.append(trial.suggest_float(f"{prefix}_thr", 0.0, 1.0))
    alpB.append(trial.suggest_float(f"{prefix}_alpB", 0.01, 0.99))
    alpD.append(trial.suggest_float(f"{prefix}_alpD", 0.01, 0.99))
    betB.append(trial.suggest_float(f"{prefix}_betB", 0.01, 1_000_000_000.0, log=True))
    betD.append(trial.suggest_float(f"{prefix}_betD", 0.01, 1_000_000_000.0, log=True))
    if use_trust:
        trust.append(trial.suggest_float(f"{prefix}_trust", 0.0, 1.0))


# Operating point of the sweep: FIXED at 0.5, NOT an Optuna search parameter.
# Reason: making decision_thr a free variable would let it be tuned against
# the uncertainty objectives (macro_misclass_auroc, macro_delta_u) — a degree
# of freedom that the MLP/B-Spline baselines don't have (they are evaluated
# at a fixed 0.5). Fixing it at 0.5 establishes symmetry across all three
# methods.
DECISION_THR_FIXED = 0.5


def _build_trial_params(trial):
    """Returns alpB, betB, alpD, betD, thr, trust, kl_weight, decision_thr, fusion_op."""
    alpB, alpD, betB, betD, thr, trust = [], [], [], [], [], []
    for prefix in FEATURE_ORDER:
        _suggest_feature(trial, prefix, thr, alpB, alpD, betB, betD, trust)
    decision_thr = DECISION_THR_FIXED   # used to be: trial.suggest_float(...)
    fusion_op = trial.suggest_categorical("fusion_op", FUSION_OPTIONS)
    # kl_weight ("lambda" in Sensoy's MSE + lambda*KL evidential loss) is kept
    # fixed here. evidential_loss is not one of the 5
    # Pareto objectives — it is only computed and logged as an informational
    # trial user_attr (see _trial_task), so it has no influence on the sweep.
    kl_weight = 1.0
    trust_out = trust if USE_TRUST_DISCOUNT else None
    return alpB, betB, alpD, betD, thr, trust_out, kl_weight, decision_thr, fusion_op


def _params_from_trial_params(trial_params_dict):
    """
    Reconstructs alpB, betB, alpD, betD, thr, trust, decision_thr, fusion_op
    from the trial.params dict (used to replay the balanced trial in the
    final test).
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
    decision_thr = trial_params_dict.get("decision_thr", DECISION_THR_FIXED)
    fusion_op = trial_params_dict["fusion_op"]
    trust_out = trust if USE_TRUST_DISCOUNT else None
    return alpB, betB, alpD, betD, thr, trust_out, decision_thr, fusion_op


# ============================================================================
# Knee-point logic (identical to B-Spline/MLP-V2)
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


def normalize_trials_for_knee(trials_metrics):
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
    Filters trials_data by the given constraints. If 0 trials remain: falls
    back to the unfiltered set with a warning.
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
        print("\n[Knee-Constraints]")
        for metric, op, threshold in constraints:
            print(f"  {metric} {op} {threshold}")
        print(f"  Trials before filter: {n_before}")
        print(f"  Trials after filter:  {n_after}")

    if n_after == 0:
        if verbose:
            print("  [WARN] Constraints too strict — no trials remain.")
            print("  [WARN] Fallback: ignoring constraints, using unfiltered trials.")
        return trials_data, True   # True = fallback used
    return keep, False


def select_balanced_trial(study, verbose=True):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if not completed:
        raise RuntimeError("No completed trials.")
    trials_data = []
    for t in completed:
        md = {name: t.values[j] for j, name in enumerate(OBJECTIVE_NAMES)}
        md["trial_number"] = t.number
        trials_data.append(md)

    # Constraint filter BEFORE knee-point selection
    trials_filtered, fallback_used = _apply_knee_constraints(
        trials_data, KNEE_CONSTRAINTS, verbose=verbose
    )

    Z = normalize_trials_for_knee(trials_filtered)
    knee_results = {}
    if verbose:
        print(f"\n{'='*70}")
        print(f"KNEE-POINT SELECTION ({len(trials_data)} trials)")
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
# Extended boxplot renderer (identical to B-Spline/MLP-V2)
# ============================================================================

def _render_extended_boxplot(boxplot_samples, out_path, title_suffix=""):
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
# Final test run: Tables 1-5 + per-sample CSV + extended boxplot
# ============================================================================

def _f1_attacker(y_true, y_pred):
    tp = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp) / denom if denom > 0 else 0.0



def _f1_attacker_macro(y_true, y_pred, attack_idx, n_attacks):
    """
    Macro-F1 (attacker = positive class), averaged per attack.

    Deliberately BIT-IDENTICAL to the macro-F1 computation in _trial_task:
    bincount, denom==0 -> 0.0, averaged over ALL
    n_attacks (no skipping). Only this way is the threshold criterion exactly
    the same as the sweep objective macro_f1.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    is_atk   = (y_true == 0)
    pred_atk = (y_pred == 0)
    tp_arr = is_atk & pred_atk
    fp_arr = (~is_atk) & pred_atk
    fn_arr = is_atk & (~pred_atk)
    tp_per = np.bincount(attack_idx, weights=tp_arr.astype(np.float64), minlength=n_attacks)
    fp_per = np.bincount(attack_idx, weights=fp_arr.astype(np.float64), minlength=n_attacks)
    fn_per = np.bincount(attack_idx, weights=fn_arr.astype(np.float64), minlength=n_attacks)
    denom_per = 2.0 * tp_per + fp_per + fn_per
    f1_per = np.where(denom_per > 0, (2.0 * tp_per) / np.maximum(denom_per, 1e-9), 0.0)
    return float(f1_per.mean())


def build_paper_outputs_final(p_all, u_all, b_all, d_all,
                               y_pred_all, y_true_all,
                               attack_all, method_name, out_dir,
                               boxplot_n=50_000):
    """
    Same structure as in the B-spline / MLP sweeps:
    per-sample CSV + Tables 1-5 + boxplot + extended boxplot.
    """
    from sl_metrics_compat import evaluate_all as _eval_all
    # (fallback module registered further below in this file)
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
            m = _eval_all(y_true=y_t, y_prob=p_t, y_pred=yp_t, uncertainty=u_t)

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
    m_pool = _eval_all(y_true=y_pool, y_prob=p_pool, y_pred=yp_pool, uncertainty=u_pool)
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

    delta_u_per_attack = [
        pa["u_wrong_mean"] - pa["u_correct_mean"]
        for pa in per_attack_results.values()
        if not np.isnan(pa.get("u_wrong_mean", float("nan")))
        and not np.isnan(pa.get("u_correct_mean", float("nan")))
    ]
    macro = {
        "macro_f1":                 _macro("f1"),
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


# Adapter for evaluate_all from metrics.py (signature-compatible with sl_metrics)
class _SLMetricsCompat:
    @staticmethod
    def evaluate_all(y_true, y_prob, y_pred, uncertainty):
        return metrics.evaluate_all(y_true, y_prob, y_pred, uncertainty)


# Register sl_metrics_compat as an importable module name
sys.modules["sl_metrics_compat"] = _SLMetricsCompat


def run_final_test(study, winner_trial, cache_dir, out_dir, method_prefix="HP_balanced"):
    """
    Forward pass on the test set with the HP values of the balanced trial,
    at the fixed operating point from the sweep, then paper outputs on Test.

    method_prefix: label prefix for method_name in the paper outputs
                   (e.g. "HP_balanced" for the multi-objective sweep, or
                   "F1only" for the single-objective negative-example
                   variant).
    """
    print(f"\n{'='*70}")
    print(f"FINAL TEST RUN (SL-HP): Trial {winner_trial.number}")
    print(f"{'='*70}")

    params = winner_trial.params
    alpB, betB, alpD, betD, thr, trust, decision_thr_init, fusion_op = \
        _params_from_trial_params(params)

    # Load Val + Test directly in the main process (no worker involved)
    print("\n[Data] Loading Val + Test from cache ...")
    X_va, y_va, attack_va, _, _ = dc.load_split("Validation", cache_dir=cache_dir)
    X_te, y_te, attack_te, _, _ = dc.load_split("Test", cache_dir=cache_dir)
    print(f"  Val:  {X_va.shape[0]:,} samples")
    print(f"  Test: {X_te.shape[0]:,} samples")

    # Restrict to selected_attacks — EXACTLY as in the sweep (_init_worker).
    # build_paper_outputs_final only iterates over selected_attacks anyway;
    # filtering here ensures that the t_star diagnostics also run on the
    # same set — and saves a lot of memory on Test.
    keep_va, attack_idx_va, n_attacks_va = _attack_filter_and_idx(attack_va)
    keep_te, attack_idx_te, n_attacks_te = _attack_filter_and_idx(attack_te)
    for name, keep, arr in (("Val", keep_va, attack_va), ("Test", keep_te, attack_te)):
        n_drop = int((~keep).sum())
        if n_drop:
            dropped = sorted(set(np.asarray(arr)[~keep].tolist()))
            print(f"  [{name}] dropped {n_drop:,} samples "
                  f"(not in selected_attacks): {dropped}")
    X_va, y_va, attack_va = X_va[keep_va], y_va[keep_va], attack_va[keep_va]
    X_te, y_te, attack_te = X_te[keep_te], y_te[keep_te], attack_te[keep_te]
    print(f"  after filter: Val {X_va.shape[0]:,} | Test {X_te.shape[0]:,}  "
          f"({n_attacks_va} attack groups)")

    msgs_va = np.concatenate([X_va.astype(np.float64),
                              y_va.astype(np.float64).reshape(-1, 1)], axis=1)
    msgs_te = np.concatenate([X_te.astype(np.float64),
                              y_te.astype(np.float64).reshape(-1, 1)], axis=1)

    # Forward pass on Val
    b_va, d_va, u_va, truth_va = sl_main.compute_opinions(
        msgs_va, alpB, betB, alpD, betD, thr, fusion_op=fusion_op, trust=trust
    )
    p_va = b_va + 0.5 * u_va

    # ------------------------------------------------------------------
    # Operating point = DECISION_THR_FIXED, identical to the sweep. NO refit.
    #
    # Justification (not laziness): from b + d + u = 1 it follows that
    #     p = b + 0.5*u = 0.5 + 0.5*(b - d)
    # so
    #     p >= 0.5   <=>   b >= d
    # t = 0.5 is therefore exactly the SL decision rule "more evidence wins",
    # and the only operating point without a free parameter. Any other t
    # implies the rule b - d > 2*(t - 0.5), i.e. an arbitrary evidence bias.
    # Since B-Spline and MLP use the same b + 0.5u convention, 0.5 is also the
    # shared, method-independent comparison point.
    #
    # Second — and this is why the refit was REMOVED here: the sweep
    # evaluates ALL five objectives at DECISION_THR_FIXED. misclass_auroc,
    # delta_u and aurc are all defined entirely through
    # correct = (y_pred == y_true). A threshold refit shifts the
    # correct/wrong partition, and with it the metrics, away from the
    # selection point: you would be selecting on one quantity and then
    # measuring it somewhere else.
    #
    # Verified on Trial 8025 (Validation, 7.0M samples):
    #     @0.500 (sweep):  macro_f1 0.3723 | mauroc 0.8398 | delta_u +0.6034
    #     @0.480 (refit):  macro_f1 0.4455 | mauroc 0.4485 | delta_u -0.0086
    # Cause: ~20.3 % of samples have p in [0.480, 0.500) and all flip at once.
    # The F1 gain and the uncertainty collapse are the same event.
    # ------------------------------------------------------------------
    y_va_flat = truth_va.astype(np.int64)   # attack_idx_va built above during filtering

    t_star = float(DECISION_THR_FIXED)
    val_macro_f1 = _f1_attacker_macro(y_va_flat, (p_va >= t_star).astype(int),
                                      attack_idx_va, n_attacks_va)
    print(f"  Operating point = sweep point {t_star:.3f}  "
          f"(Val macro-F1 there = {val_macro_f1:.4f})")

    # For transparency only: what a refit WOULD have produced — intentionally
    # NOT used.
    ts = np.linspace(0.01, 0.99, 199)
    f1s = [_f1_attacker_macro(y_va_flat, (p_va >= t).astype(int),
                              attack_idx_va, n_attacks_va) for t in ts]
    t_refit = float(ts[int(np.argmax(f1s))])
    print(f"  [Info] macro-F1 optimum would be t={t_refit:.3f} "
          f"(Val macro-F1={max(f1s):.4f}) -- intentionally NOT used, see comment above.")

    # Forward pass on Test
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
        attack_all=attack_te,
        method_name=method_name, out_dir=out_dir,
    )

    # Save the "model" — for SL-HP this is the per-feature parameters +
    # decision_thr
    model = {
        "variant": "sl_hp_per_feature",
        "feature_order": FEATURE_ORDER,
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
    # NSGA-II — consistent with the B-spline / MLP sweeps.
    # Larger population than for B-Spline/MLP (10), because the HP search
    # space is much higher-dimensional here: 40 dimensions (8 features x 5 HPs).
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
                    f"current config has {len(OBJECTIVE_DIRECTION)}.")
            print(f"[Optuna] Study loaded — {len(study.trials)} trials so far.")
            return study
        except KeyError:
            print("[Optuna] No existing study found, creating a new one.")
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB,
        directions=OBJECTIVE_DIRECTION, sampler=sampler,
        load_if_exists=True,
    )
    if len(study.directions) != len(OBJECTIVE_DIRECTION):
        raise RuntimeError(
            "Schema mismatch on creation! "
            "Fix: remove hp_optuna_v2.db, or change STUDY_NAME.")
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
    parser.add_argument("--n-trials", type=int, default=40000)
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count // 2)")
    parser.add_argument("--cache-dir", default=dc.CACHE_DIR_DEFAULT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for sweep outputs: CSV, Pareto plots, "
                             "balanced_trial.json, final_test/ "
                             f"(default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run knee-point selection + final test, skip the sweep")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Sweep HP V2] Objectives:  {OBJECTIVE_NAMES}")
    print(f"[Sweep HP V2] Directions:  {OBJECTIVE_DIRECTION}")
    print(f"[Sweep HP V2] Output dir:  {args.output_dir}/")
    print(f"[Sweep HP V2] Cache dir:   {args.cache_dir}")

    # Check whether the cache exists
    if not dc.cache_exists(cache_dir=args.cache_dir):
        print(f"\n[ERROR] Cache missing in {args.cache_dir}/")
        print("        Run first: python data_cache.py --build")
        sys.exit(1)

    # Worker pool: only loads Validation for the sweep (Train is not needed
    # iteratively for SL-HP — the HPs define the function directly).
    # If you also want to sweep on Train (e.g. to check whether Val is
    # representative), add "Train" to this list.
    splits_for_worker = ["Validation"]

    n_workers = args.n_workers if args.n_workers is not None \
                else max(1, (os.cpu_count() or 8) // 2)
    print(f"[Sweep HP V2] N workers:   {n_workers}")

    study = create_or_load_study(resume=args.resume or args.analyze_only)

    # ---------- Phase 1: Sweep ----------
    if not args.analyze_only:
        n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_remaining = args.n_trials - n_completed
        if n_remaining <= 0:
            print(f"\n[Sweep] Target ({args.n_trials}) already reached ({n_completed} trials).")
        else:
            print(f"\n[Sweep] {n_completed} done, running {n_remaining} more trials")
            pool = MetricsPoolHP(
                max_workers=n_workers, cache_dir=args.cache_dir,
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
                            "delta_u_mean", "macro_delta_u",
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
            print(f"\n[Sweep] Total sweep duration: {(time.time()-t_sweep)/60:.1f} min")

    # ---------- Phase 2: Analysis ----------
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

    # ---------- Phase 3: Knee-point selection ----------
    winner, info = select_balanced_trial(study, verbose=True)
    with open(os.path.join(args.output_dir, "balanced_trial.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"  balanced JSON: {os.path.join(args.output_dir, 'balanced_trial.json')}")

    # ---------- Phase 4: Final test ----------
    winner_trial = next(t for t in study.trials if t.number == winner)
    final_dir = os.path.join(args.output_dir, "final_test")
    run_final_test(study, winner_trial, cache_dir=args.cache_dir, out_dir=final_dir)
    print(f"\n[Sweep HP V2] DONE. Final outputs in {final_dir}/")


if __name__ == "__main__":
    main()
