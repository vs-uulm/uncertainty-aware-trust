"""
Negative example / ablation: SL-HP sweep with macro_f1 as the ONLY objective.

Purpose
-------
The main sweep (../multi-objective_analytic/sweep_optuna.py) optimizes five
Pareto objectives: macro_f1 plus four uncertainty-quality metrics. This
script optimizes ONLY macro_f1 and serves as a counter-check: it should show
that a detector with comparable F1 delivers clearly worse uncertainty quality
when uncertainty was never part of the optimization objective
("confident-wrong").

Cleanliness of the ablation
---------------------------
This script IMPORTS the local sweep_optuna.py (a verbatim copy of
../multi-objective_analytic/sweep_optuna.py, together with sl_model.py,
metrics.py, data_cache_gnss.py and paper_outputs.py) and uses its building
blocks unchanged:

    Search space       hp._build_trial_params    (incl. betB/betD <= 1e3)
    Operating point    hp.DECISION_THR_FIXED     (0.5, no refit)
    Attack groups      hp._attack_filter_and_idx (selected_attacks, fixed order)
    Worker/metrics     hp._trial_task, hp.MetricsPoolHP
    Sampler            NSGA-II, population_size=50, seed=42  (identical)
    Final test         hp.run_final_test         (Tables 1-5, boxplots)
    delta_u statistic  hp.DELTA_U_STAT


Usage:
    python sweep_optuna_f1only.py --n-trials 2000 --data-dir ../data
    python sweep_optuna_f1only.py --resume --n-trials 5000
    python sweep_optuna_f1only.py --analyze-only
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna

import sweep_optuna as hp


# ============================================================================
# Configuration — everything else comes from sweep_optuna
# ============================================================================

DEFAULT_OUTPUT_DIR = "results/analytic_f1only"
STUDY_DB = "sqlite:///hp_optuna_gnss_f1only_v1.db"
# The number of groups and the delta_u statistic enter the logged metrics even
# though only macro_f1 is optimized -> they belong in the name. "lambda0"
# marks the kl_weight dimension in [0, 1].
STUDY_NAME = f"gnss_hp_f1only_v2_8scn_{hp.DELTA_U_STAT}du_lambda0"

# The ONLY optimization objective.
OBJECTIVE_NAME = "macro_f1"
OBJECTIVE_DIRECTION = "maximize"

# Range for the lambda/kl_weight dimension (see module docstring).
# [0, 1] matches Sensoy et al.'s annealing convention (lambda_t = min(1, t/10),
# capped at 1.0).
KL_WEIGHT_RANGE = (0.0, 1.0)

# These metrics are logged but NOT optimized. That is the whole point.
LOGGED_METRICS = [
    "f1", "macro_f1", "aurc", "macro_aurc",
    "misclass_auroc", "macro_misclass_auroc",
    "delta_u_median", "macro_delta_u",
    "macro_belief_correctness", "evidential_loss",
]

METHOD_PREFIX = "F1only"


# ============================================================================
# Study
# ============================================================================

def create_or_load_study(resume=False):
    """
    Sampler BIT-IDENTICAL to the main sweep (hp.create_or_load_study):
    NSGA-II, population_size=50, UniformCrossover, crossover_prob=0.9,
    swapping_prob=0.5, seed=42. Only the number of objectives differs.
    """
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
            if len(study.directions) != 1:
                raise RuntimeError(
                    f"Schema mismatch! Study has {len(study.directions)} objectives, "
                    f"expected 1. This is probably a main-sweep DB. "
                    f"Check STUDY_NAME/STUDY_DB.")
            print(f"[Optuna] Study loaded — {len(study.trials)} trials so far.")
            return study
        except KeyError:
            print("[Optuna] No existing study, creating a new one.")
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB,
        direction=OBJECTIVE_DIRECTION, sampler=sampler,
        load_if_exists=True,
    )
    if len(study.directions) != 1:
        raise RuntimeError(
            f"Schema mismatch on creation: study has {len(study.directions)} "
            f"objectives. Fix: delete the DB from {STUDY_DB} or change STUDY_NAME.")
    return study


def write_summary_csv(study, path):
    """All trials with params + ALL logged metrics (not just the objective)."""
    rows = []
    best = study.best_trial.number if study.best_trial else None
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": t.number,
            "is_best": t.number == best,
            "duration_min": t.user_attrs.get("duration_min", 0),
            OBJECTIVE_NAME: t.value if t.value is not None else float("nan"),
        }
        row.update(t.params)
        for k in LOGGED_METRICS:
            if k in t.user_attrs:
                row[k] = t.user_attrs[k]
        rows.append(row)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  CSV: {path}")


def report_winner(study):
    """
    Core statement of the ablation: the F1 winner, evaluated on the four
    metrics it never saw — next to best/median over all trials as context.
    """
    best = study.best_trial
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'='*74}")
    print(f"F1-ONLY WINNER: Trial {best.number}   macro_f1 = {best.value:.4f}")
    print(f"{'='*74}")
    print(f"{'Metric':<28}{'Winner':>10}{'Best trial':>16}{'Median':>10}")
    print("-" * 74)
    spec = [
        ("macro_f1",                 "max"),
        ("macro_aurc",               "min"),
        ("macro_misclass_auroc",     "max"),
        ("macro_delta_u",            "max"),
        ("macro_belief_correctness", "max"),
    ]
    info = {"winner_trial": best.number, "objective": OBJECTIVE_NAME,
            "objective_value": float(best.value),
            "params": dict(best.params), "metrics": {}}
    for key, direction in spec:
        vals = [t.user_attrs[key] for t in completed if key in t.user_attrs]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        if not vals:
            continue
        w = best.user_attrs.get(key, float("nan"))
        b = max(vals) if direction == "max" else min(vals)
        opt = "" if key == OBJECTIVE_NAME else "  (not optimized)"
        print(f"{key:<28}{w:>10.4f}{b:>16.4f}{np.median(vals):>10.4f}{opt}")
        info["metrics"][key] = {
            "winner": float(w), "best_over_trials": float(b),
            "median_over_trials": float(np.median(vals)),
            "optimized": key == OBJECTIVE_NAME,
        }
    print("-" * 74)
    print("Column 'Best trial' = what would have been reachable in the SAME search space.")
    print("The gap to the winner is the price of optimizing for F1 only.")
    return info


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=40000,
                        help="Total number of completed trials to reach (cumulative)")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count // 2)")
    parser.add_argument("--data-dir", "--cache-dir", dest="data_dir",
                        default=hp.dc.DATA_DIR_DEFAULT,
                        help="Directory with oneclass_sl_{train,val,test}.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where the summary CSV, best-trial JSON and "
                             "final_test/ are written")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the existing study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip the sweep; only winner report + final test")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[F1only] Objective:     {OBJECTIVE_NAME} ({OBJECTIVE_DIRECTION})")
    print("[F1only] Logged only:   macro_aurc, macro_misclass_auroc, "
          "macro_delta_u, macro_belief_correctness")
    print(f"[F1only] Attacks:       {hp.selected_attacks}")
    print(f"[F1only] decision_thr:  {hp.DECISION_THR_FIXED} (fixed, from sweep_optuna)")
    print(f"[F1only] kl_weight:     searched in {list(KL_WEIGHT_RANGE)} (diagnostic only)")
    print(f"[F1only] delta_u stat:  {hp.DELTA_U_STAT} (from sweep_optuna)")
    print(f"[F1only] Output dir:    {args.output_dir}/")

    if not hp.dc.data_exists(data_dir=args.data_dir):
        print(f"\n[ERROR] GNSS split CSVs missing in {args.data_dir}/")
        sys.exit(1)

    n_workers = args.n_workers if args.n_workers is not None \
                else max(1, (os.cpu_count() or 8) // 2)
    print(f"[F1only] N workers:     {n_workers}")

    study = create_or_load_study(resume=args.resume or args.analyze_only)

    # ---------- Phase 1: sweep ----------
    if not args.analyze_only:
        n_done = sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE)
        n_remaining = args.n_trials - n_done
        if n_remaining <= 0:
            print(f"\n[Sweep] Target ({args.n_trials}) already reached ({n_done}).")
        else:
            print(f"\n[Sweep] {n_done} done, running {n_remaining} more trials")
            pool = hp.MetricsPoolHP(
                max_workers=n_workers, data_dir=args.data_dir,
                splits_to_load=["Validation"],
            )

            def objective(trial):
                # IDENTICAL model search space as in the main sweep.
                alpB, betB, alpD, betD, thr, trust, _kl_weight_fixed, decision_thr, fusion_op = \
                    hp._build_trial_params(trial)
                # Sweep lambda/kl_weight instead of the fixed value — see the
                # module docstring. Only affects the diagnostic evidential_loss.
                kl_weight = trial.suggest_float("kl_weight", *KL_WEIGHT_RANGE)
                t0 = time.time()
                try:
                    result = pool.eval(
                        alpB, betB, alpD, betD, thr, decision_thr,
                        fusion_op=fusion_op, trust=trust, kl_weight=kl_weight,
                        split="Validation",
                    )
                except (RuntimeError, MemoryError, TimeoutError) as e:
                    print(f"  [WARN] Trial {trial.number} failed: {e}")
                    raise optuna.TrialPruned()

                # Log ALL metrics — including the non-optimized ones.
                for key in LOGGED_METRICS:
                    if key in result:
                        trial.set_user_attr(key, result[key])
                trial.set_user_attr("fusion_op", fusion_op)
                trial.set_user_attr("duration_min", (time.time() - t0) / 60)

                v = result.get(OBJECTIVE_NAME)
                if v is None or np.isnan(float(v)):
                    return -1.0
                return float(v)

            t_sweep = time.time()
            try:
                study.optimize(objective, n_trials=n_remaining, n_jobs=n_workers,
                               catch=(RuntimeError, MemoryError, TimeoutError))
            finally:
                pool.shutdown()
            print(f"\n[Sweep] Duration: {(time.time()-t_sweep)/60:.1f} min")

    # ---------- Phase 2: analysis ----------
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[ERROR] No successful trials — aborting.")
        return
    print(f"\n[Analysis] {len(completed)} completed trials")
    write_summary_csv(study, os.path.join(args.output_dir, "sweep_summary_f1only.csv"))

    info = report_winner(study)
    with open(os.path.join(args.output_dir, "best_trial_f1only.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)

    # ---------- Phase 3: final test ----------
    # Same function as in the main sweep, only a different label and directory.
    final_dir = os.path.join(args.output_dir, "final_test")
    hp.run_final_test(study, study.best_trial, data_dir=args.data_dir,
                      out_dir=final_dir, method_prefix=METHOD_PREFIX)
    print(f"\n[F1only] DONE. Final outputs in {final_dir}/")


if __name__ == "__main__":
    main()
