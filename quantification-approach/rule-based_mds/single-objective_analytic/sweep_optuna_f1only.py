"""
Single-objective Optuna sweep — optimizes EXCLUSIVELY for F1 (negative example).

Purpose (paper): baseline/negative example. Optimizes only macro_f1 (or
micro-F1) and ignores every uncertainty-quality objective of the main sweep
(macro_aurc, macro_misclass_auroc, macro_delta_u, macro_belief_correctness).

Expected result: F1 comparable to the multi-objective sweep, but poor
uncertainty quality — the "confident-wrong" phenomenon occurs unchecked
because uncertainty quality was never part of the objective. That is exactly
the didactic point: show what happens when trust/uncertainty is NOT
co-optimized.

Design: reuses EVERYTHING from sweep_optuna.py — the worker pool
(_trial_task, MetricsPoolHP), HP suggestion (_build_trial_params),
paper outputs (Tables 1-5, boxplots), run_final_test.
The only things that change here:
  - The study is single-objective (direction="maximize", one objective).
  - Selection: study.best_trial instead of knee-consensus (no Pareto front
    needed).
  - Selectable sampler:
      nsga2  (default) — configured identically to the multi-objective
                         sweep, so the objective set is the ONLY difference
                         from the main procedure. Cleanest ablation for the
                         paper.
      tpe             — a more sample-efficient standard single-objective
                         sampler.
  - kl_weight (lambda in Sensoy's evidential loss L = MSE + lambda*KL) is
    suggested per trial in [0, 1] instead of sweep_optuna.py's fixed 1.0.
    It has no effect on predictions/F1 (only feeds the diagnostic
    "evidential_loss" user_attr), so this doesn't change what gets
    optimized — it lets that diagnostic reflect a swept lambda instead of
    an arbitrary fixed value.

Important: ALL metrics (including the non-optimized ones) are stored per
trial as user_attrs, so it can be shown directly afterwards that the F1
winner has e.g. a poor macro_delta_u / macro_misclass_auroc.

Usage:
    python data_cache.py --build
    python sweep_optuna_f1only.py --n-trials 2000
    python sweep_optuna_f1only.py --resume --n-trials 5000
    python sweep_optuna_f1only.py --metric f1            # micro-F1 instead of macro
    python sweep_optuna_f1only.py --sampler tpe
    python sweep_optuna_f1only.py --analyze-only
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import optuna

import data_cache as dc
import sweep_optuna as hp   # reuse the entire machinery


# ============================================================================
# Configuration — own namespace/DB so nothing collides with the
# multi-objective study.
# ============================================================================

# Own namespace for rule-based MBD (8 features). Kept separate from:
#   - the old 6-feature F1-only run (incompatible search space), and
#   - the multi-objective HP study (hp_optuna_v2.db).
# Output directory, relative to the working directory.
SWEEP_ROOT = "sweep_output/f1only"
STUDY_DB = "sqlite:///f1only_rule_mbd_optuna.db"
STUDY_NAME = "f1only_rule_mbd_singleobj_meandu"

# Which F1 variant is optimized. Both are present in hp._trial_task's result
# dict: "macro_f1" (averaged per attack, fair vs. the main sweep) or "f1"
# (micro, pooled).
DEFAULT_METRIC = "macro_f1"

# Range for the lambda/kl_weight ablation dimension (see module docstring).
# [0, 1] matches Sensoy et al.'s annealing convention (lambda_t = min(1, t/10),
# capped at 1.0) — larger values aren't explored in the literature since they
# increasingly force evidence on the wrong class toward zero regardless of
# fit quality.
KL_WEIGHT_RANGE = (0.0, 1.0)

# For later analysis: all metrics that get stored as user_attr.
ALL_METRIC_KEYS = [
    "f1", "macro_f1", "aurc", "macro_aurc",
    "misclass_auroc", "macro_misclass_auroc",
    "delta_u_mean", "macro_delta_u",
    "macro_belief_correctness", "evidential_loss",
]


# ============================================================================
# Study
# ============================================================================

def make_sampler(kind):
    if kind == "nsga2":
        # Configured identically to the multi-objective sweep in
        # sweep_optuna.create_or_load_study -> only the objective set
        # differs, not the optimizer. Cleanest ablation for the paper.
        return optuna.samplers.NSGAIISampler(
            population_size=50,
            crossover=optuna.samplers.nsgaii.UniformCrossover(),
            crossover_prob=0.9,
            swapping_prob=0.5,
            seed=42,
        )
    if kind == "tpe":
        return optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)
    raise ValueError(f"Unknown sampler: {kind}")


def create_or_load_study(sampler_kind="nsga2", resume=False):
    sampler = make_sampler(sampler_kind)
    if resume:
        try:
            study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB,
                                      sampler=sampler)
            if len(study.directions) != 1:
                raise RuntimeError(
                    f"Schema mismatch! Study has {len(study.directions)} "
                    f"objectives, expected 1 (single-objective).")
            print(f"[Optuna] Study loaded — {len(study.trials)} trials so far.")
            return study
        except KeyError:
            print("[Optuna] No existing study, creating a new one.")
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB,
        direction="maximize", sampler=sampler, load_if_exists=True,
    )
    if len(study.directions) != 1:
        raise RuntimeError(
            f"Schema mismatch on creation! Study has {len(study.directions)} "
            f"objectives. Fix: 'rm f1only_rule_mbd_optuna.db' or change STUDY_NAME.")
    return study


# ============================================================================
# Objective — returns a SINGLE scalar (F1), but stores all metrics.
# ============================================================================

def build_objective(pool, metric_key):
    def objective(trial):
        alpB, betB, alpD, betD, thr, trust, _kl_weight_fixed, decision_thr, fusion_op = \
            hp._build_trial_params(trial)
        # Sweep lambda/kl_weight instead of using sweep_optuna.py's fixed
        # value — see module docstring. Has no effect on predictions/F1,
        # only on the diagnostic "evidential_loss" user_attr below.
        kl_weight = trial.suggest_float("kl_weight", *KL_WEIGHT_RANGE)
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

        # Store ALL metrics (including the non-optimized ones) — shows later
        # that the F1 winner has poor uncertainty quality.
        for key in ALL_METRIC_KEYS:
            if key in result:
                trial.set_user_attr(key, result[key])
        trial.set_user_attr("fusion_op", fusion_op)
        trial.set_user_attr("duration_min", dt / 60)

        val = result.get(metric_key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise optuna.TrialPruned()
        return float(val)
    return objective


# ============================================================================
# Summary CSV (single-objective: all metrics from user_attrs)
# ============================================================================

def write_summary_csv(study, metric_key, path):
    best_num = study.best_trial.number if study.best_trials else None
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": t.number,
            "is_best":      (t.number == best_num),
            "objective":    metric_key,
            "objective_value": t.value if t.value is not None else float("nan"),
            "duration_min": t.user_attrs.get("duration_min", 0),
        }
        for k in ALL_METRIC_KEYS:
            row[k] = t.user_attrs.get(k, float("nan"))
        for k, v in t.params.items():
            row[k] = v
        rows.append(row)
    if not rows:
        return
    all_keys = list(rows[0].keys()) + sorted(
        {k for r in rows for k in r.keys()} - set(rows[0].keys())
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  CSV: {path}")


def report_winner_tradeoff(winner_trial, metric_key):
    """Console summary: the point of the negative example at a glance."""
    ua = winner_trial.user_attrs
    print(f"\n{'='*70}")
    print(f"F1-ONLY WINNER: trial {winner_trial.number}   "
          f"(optimized for {metric_key} = {winner_trial.value:.4f})")
    print(f"{'='*70}")
    print("  Optimized objective:")
    print(f"    macro_f1                 = {ua.get('macro_f1', float('nan')):.4f}")
    print(f"    f1 (micro)               = {ua.get('f1', float('nan')):.4f}")
    print("  NOT optimized (collateral — this is where confident-wrong shows up):")
    print(f"    macro_misclass_auroc     = {ua.get('macro_misclass_auroc', float('nan')):.4f}"
          f"   (0.5 = chance)")
    print(f"    macro_delta_u            = {ua.get('macro_delta_u', float('nan')):.4f}"
          f"   (>0 = u higher on errors)")
    print(f"    macro_aurc               = {ua.get('macro_aurc', float('nan')):.4f}"
          f"   (lower = better)")
    print(f"    macro_belief_correctness = {ua.get('macro_belief_correctness', float('nan')):.4f}")
    print(f"{'='*70}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=40000)
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count // 2)")
    parser.add_argument("--cache-dir", default=dc.CACHE_DIR_DEFAULT)
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        choices=["macro_f1", "f1"],
                        help="F1 variant to optimize (default macro_f1)")
    parser.add_argument("--sampler", default="nsga2", choices=["nsga2", "tpe"],
                        help="nsga2 = identical to the main sweep (cleanest "
                             "ablation); tpe = more sample-efficient (default nsga2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only best-trial + final test, no sweep")
    args = parser.parse_args()

    Path(SWEEP_ROOT).mkdir(parents=True, exist_ok=True)

    print(f"[Sweep F1-only] Objective: {args.metric} (maximize) — SINGLE OBJECTIVE")
    print(f"[Sweep F1-only] Sampler:   {args.sampler}")
    print(f"[Sweep F1-only] Output:    {SWEEP_ROOT}/")
    print(f"[Sweep F1-only] Cache:     {args.cache_dir}")

    if not dc.cache_exists(cache_dir=args.cache_dir):
        print(f"\n[ERROR] Cache missing in {args.cache_dir}/")
        print("        Run first: python data_cache.py --build")
        raise SystemExit(1)

    splits_for_worker = ["Validation"]
    n_workers = args.n_workers if args.n_workers is not None \
                else max(1, (os.cpu_count() or 8) // 2)
    print(f"[Sweep F1-only] N-Workers: {n_workers}")

    study = create_or_load_study(sampler_kind=args.sampler,
                                 resume=args.resume or args.analyze_only)

    # ---------- Phase 1: sweep ----------
    if not args.analyze_only:
        n_completed = sum(1 for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE)
        n_remaining = args.n_trials - n_completed
        if n_remaining <= 0:
            print(f"\n[Sweep] Target ({args.n_trials}) already reached "
                  f"({n_completed} trials).")
        else:
            print(f"\n[Sweep] {n_completed} done, running {n_remaining} more trials")
            pool = hp.MetricsPoolHP(
                max_workers=n_workers, cache_dir=args.cache_dir,
                splits_to_load=splits_for_worker,
            )
            objective = build_objective(pool, args.metric)
            t_sweep = time.time()
            try:
                study.optimize(
                    objective, n_trials=n_remaining, n_jobs=n_workers,
                    catch=(RuntimeError, MemoryError, TimeoutError),
                )
            finally:
                pool.shutdown()
            print(f"\n[Sweep] Total sweep duration: {(time.time()-t_sweep)/60:.1f} min")

    # ---------- Phase 2: analysis ----------
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[ERROR] No successful trials — aborting.")
        return
    print(f"\n{'='*70}")
    print(f"SWEEP ANALYSIS  ({len(completed)} completed trials)")
    print(f"{'='*70}")
    write_summary_csv(study, args.metric,
                      os.path.join(SWEEP_ROOT, "sweep_summary.csv"))

    # ---------- Phase 3: best trial (no knee — single objective) ----------
    winner_trial = study.best_trial
    report_winner_tradeoff(winner_trial, args.metric)
    winner_info = {
        "selection":     "single_objective_best_trial",
        "objective":     args.metric,
        "winner_trial":  winner_trial.number,
        "objective_value": winner_trial.value,
        "all_metrics":   {k: winner_trial.user_attrs.get(k) for k in ALL_METRIC_KEYS},
        "params":        winner_trial.params,
    }
    with open(os.path.join(SWEEP_ROOT, "best_trial.json"), "w") as f:
        json.dump(winner_info, f, indent=2, default=str)
    print(f"  best JSON: {os.path.join(SWEEP_ROOT, 'best_trial.json')}")

    # ---------- Phase 4: final test (reused, only a different label) ----------
    final_dir = os.path.join(SWEEP_ROOT, "final_test")
    hp.run_final_test(study, winner_trial, cache_dir=args.cache_dir,
                      out_dir=final_dir, method_prefix="F1only")
    print(f"\n[Sweep F1-only] DONE. Final outputs in {final_dir}/")


if __name__ == "__main__":
    main()
