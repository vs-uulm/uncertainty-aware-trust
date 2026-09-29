"""
Multi-objective Optuna sweep for an evidential MLP misbehavior detector.

Pipeline
  1. Load features once from the binary cache (data_cache.py) and standardize
     them once (no reload / re-standardization per trial).
  2. NSGA-II sweep over loss weights and MLP hyperparameters. Every trial is
     trained inline in the same process and evaluated on the VALIDATION split
     only, against five macro-averaged (per-attack) objectives.
  3. Pick a single "balanced" trial from all completed trials via four
     knee-point methods plus a majority vote.
  4. Final TEST run with the balanced trial only: paper tables 1-5, boxplots,
     per-sample CSV.

Model: evidential MLP head over standardized raw detector features.
Standard Subjective Logic convention: b = e_b/S, d = e_a/S, u = 2/S,
p = b + a*u (a = base rate).

Usage:
    python data_cache.py --build --data-root <raw_json_dir> --cache-dir <cache_dir>
    python sweep_optuna.py --cache-dir <cache_dir> --n-trials 20
    python sweep_optuna.py --cache-dir <cache_dir> --resume --n-trials 30
    python sweep_optuna.py --cache-dir <cache_dir> --analyze-only
"""
import argparse
import csv
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn

from sl_metrics import evaluate_all, delta_u_cohens_d
import paper_outputs
import data_cache as dc


# ============================================================================
# Configuration
# ============================================================================

SWEEP_ROOT = "results"
STUDY_DB = "sqlite:///mlp_optuna.db"
# Bump the study name whenever objectives, input encoding, model architecture
# or the search space change. With --resume, NSGA-II would otherwise mix
# incompatible trials into one Pareto front. The current study uses macro
# averages over attacks, delta_u as the MEAN over samples (the median is blind
# to the confident-wrong tail), valid-only standardization with mean
# imputation and a missingness mask, and loss weights (lambdas) searched from 0.
STUDY_NAME = "mlp_pareto_v6_5obj_meandu_naimpute_lambda0"

# Operating point for ALL evaluations (sweep and final test).
# p = b + 0.5*u = 0.5 + 0.5*(b - d)  =>  p >= 0.5  <=>  b >= d.
# This is the SL decision rule "more evidence wins" and the only threshold
# without a free parameter.
DECISION_THR_FIXED = 0.5

OBJECTIVE_NAMES = [
    "macro_f1", "macro_aurc", "macro_misclass_auroc",
    "macro_delta_u", "macro_belief_correctness",
]
OBJECTIVE_DIRECTION = ["maximize", "minimize", "maximize", "maximize", "maximize"]

# timeDelayAttack and trafficCongestionSybil are deliberately excluded: they
# cannot be detected by local MDSs.
selected_attacks = [
    "constantPositionOffset", "randomPositionOffset", "positionMirroring",
    "suddenStop", "accelerationMultiplication", "feignedBraking",
    "constantSpeedOffset", "randomSpeedOffset", "suddenConstantSpeed",
    "zeroSpeedReport", "reversedHeading", "dataReplay", "dosAttack",
]
FEATURE_NAMES = dc.FEATURE_NAMES

U_MIN = 1e-3
BASE_RATE = 0.5

# ----------------------------------------------------------------------------
# Search space (documented in README.md)
# ----------------------------------------------------------------------------
# (low, high, step, log). The loss weights (lambdas) start at 0 so the sweep
# can switch the corresponding loss term off entirely; they are therefore
# sampled on a linear scale (a log scale cannot contain 0).
SEARCH_SPACE_FLOAT = {
    "DELTA_U_WEIGHT_MAX":       (0.0,  20.0, None, False),
    "DELTA_U_MARGIN":           (0.20, 0.60, 0.05, False),
    "BELIEF_WEIGHT":            (0.0,  5.0,  None, False),
    "DROPOUT":                  (0.0,  0.3,  0.05, False),
    "LR_MLP":                   (1e-4, 1e-2, None, True),
    "COMPOSITE_DELTA_U_WEIGHT": (0.0,  5.0,  None, False),
}
SEARCH_SPACE_CATEGORICAL = {
    "HIDDEN_DIM_1": [32, 64, 128],
    "HIDDEN_DIM_2": [16, 32, 64],
}
WEIGHT_DECAY = 1e-5

# NSGA-II sampler
NSGA_POPULATION_SIZE = 24
NSGA_CROSSOVER_PROB = 0.9
NSGA_SWAPPING_PROB = 0.5
SEED = 42

# Training defaults per trial
NUM_EPOCHS_PER_TRIAL = 3000
KL_ANNEAL_STEPS = 200
DELTA_U_START_EPOCH = 0
DELTA_U_ANNEAL_STEPS = 250
DELTA_U_MAX_PAIRS_PER_CLASS = 1500
VAL_CHECK_EVERY = 25
PATIENCE = 15
EVAL_CHUNK_SIZE = 200_000

# ----------------------------------------------------------------------------
# Rule-based input handling
# ----------------------------------------------------------------------------
# Feature values are implausibilities in [0, 1] (data_cache.py:
# x = 1 - plausibility). NA_SENTINEL (-1) means "detector not applicable".
#
# Unlike the SL-HP / B-spline approaches, an MLP cannot treat N/A as a vacuous
# opinion: a standardized -1 is just another input value, and a biased linear
# layer does not turn 0 into a neutral signal. Including the -1 mass would
# also distort mean/std. Standard handling of missing values for an MLP:
#   1) mean/std computed on valid values only (x >= 0).
#   2) N/A is mean-imputed, i.e. becomes 0 after standardization.
#   3) Optional binary missingness mask per feature (1 = N/A) so the MLP can
#      distinguish "missing" from "average value".
# USE_NA_MASK=False -> 1 column per feature, input_dim = N_FEATURES.
# USE_NA_MASK=True  -> 2 columns per feature, input_dim = 2 * N_FEATURES.
NA_SENTINEL = -1.0
USE_NA_MASK = True

# ----------------------------------------------------------------------------
# Per-sample CSV I/O
# ----------------------------------------------------------------------------
# If the output directory lives on a network mount, millions of small buffered
# writes are the bottleneck. The CSV is therefore written locally first and
# then moved in one large sequential operation.
#   CSV_LOCAL_TMP_DIR = "."   -> temp file in the working directory (recommended).
#   CSV_LOCAL_TMP_DIR = None  -> write directly to the output directory.
CSV_LOCAL_TMP_DIR = "."
CSV_WRITE_CHUNK = 500_000


def _standardize_feature_with_na(col_tr, col_va, col_te):
    """
    Standardize one feature column with N/A handling.

    mean/std are computed on valid train values only (x >= 0); N/A values are
    mean-imputed (-> 0 after standardization).
    Returns (Phi_tr, Phi_va, Phi_te, mean, std) with Phi of shape (N, 1), or
    (N, 2) when USE_NA_MASK is set.
    """
    col_tr = np.asarray(col_tr, dtype=np.float32).reshape(-1)
    col_va = np.asarray(col_va, dtype=np.float32).reshape(-1)
    col_te = np.asarray(col_te, dtype=np.float32).reshape(-1)

    valid_tr = col_tr[col_tr >= 0.0]
    if valid_tr.size > 0:
        mean = float(valid_tr.mean())
        std = float(valid_tr.std()) + 1e-8
    else:
        mean, std = 0.0, 1.0          # feature is entirely N/A in train

    def _val_col(col):
        na = col < 0.0
        imputed = np.where(na, mean, col)
        return ((imputed - mean) / std).astype(np.float32)

    def _phi(col):
        v = _val_col(col).reshape(-1, 1)
        if not USE_NA_MASK:
            return v
        m = (col < 0.0).astype(np.float32).reshape(-1, 1)   # 1 = N/A
        return np.concatenate([v, m], axis=1)

    return _phi(col_tr), _phi(col_va), _phi(col_te), mean, std


# ============================================================================
# Model (evidential MLP head, standard SL)
# ============================================================================

class EvidentialMLPHead(nn.Module):
    """
    Flat MLP head over raw features.

    Input : (N, input_dim)
    Output: (N, 2) evidence (e_benign, e_attacker), both >= 0 via softplus.
    """
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, Phi_concat):
        return torch.nn.functional.softplus(self.net(Phi_concat))


def opinion_joint(Phi_concat, mlp_head):
    """Standard-SL evidential joint opinion (b, d, u)."""
    evidence = mlp_head(Phi_concat)
    e_b = evidence[:, 0]
    e_a = evidence[:, 1]
    S = e_b + e_a + 2.0
    b = e_b / S
    d = e_a / S
    u = torch.clamp(2.0 / S, min=U_MIN)
    return b, d, u


def forward_prob_multi(Phi_by_f, mlp_head, feature_names, base_rate=BASE_RATE):
    Phi_concat = torch.cat([Phi_by_f[f] for f in feature_names], dim=1)
    b, d, u = opinion_joint(Phi_concat, mlp_head)
    # Reconstruct Dirichlet parameters: alpha_k = b_k * S + 1
    S = 2.0 / u
    alpha = torch.stack([b * S + 1.0, d * S + 1.0], dim=1)
    p = torch.clamp(b + base_rate * u, 1e-6, 1 - 1e-6)
    return p, (b, d, u), alpha


def forward_prob_multi_chunked(Phi_by_f_cpu, mlp_head, feature_names, device,
                               base_rate=BASE_RATE, chunk_size=EVAL_CHUNK_SIZE):
    """Memory-safe inference: CPU features are moved to `device` chunk by chunk."""
    n = next(iter(Phi_by_f_cpu.values())).shape[0]
    p_out = torch.empty(n, dtype=torch.float32)
    b_out = torch.empty(n, dtype=torch.float32)
    d_out = torch.empty(n, dtype=torch.float32)
    u_out = torch.empty(n, dtype=torch.float32)
    alpha_out = torch.empty(n, 2, dtype=torch.float32)
    mlp_head.eval()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Phi_chunk = {f: Phi_by_f_cpu[f][start:end].to(device, non_blocking=True)
                     for f in feature_names}
        with torch.no_grad():
            p, (b, d, u), alpha = forward_prob_multi(
                Phi_chunk, mlp_head, feature_names, base_rate)
        p_out[start:end] = p.cpu()
        b_out[start:end] = b.cpu()
        d_out[start:end] = d.cpu()
        u_out[start:end] = u.cpu()
        alpha_out[start:end] = alpha.cpu()
        del Phi_chunk, p, b, d, u, alpha
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return p_out, (b_out, d_out, u_out), alpha_out


# ============================================================================
# Loss functions (standard SL)
# ============================================================================

def evidential_mse_loss(alpha, y_onehot, neg_weight=1.0):
    S = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    err = (y_onehot - p_hat) ** 2
    var = alpha * (S - alpha) / (S * S * (S + 1.0))
    w = torch.tensor([1.0, neg_weight], device=alpha.device).view(1, 2)
    return (w * (err + var)).sum(dim=1).mean()


def evidential_kl_loss(alpha, y_onehot):
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    K = alpha_tilde.shape[1]
    sum_a = alpha_tilde.sum(dim=1, keepdim=True)
    log_term = (
        torch.lgamma(sum_a).squeeze(1)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=1)
    )
    digamma_term = (
        (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_a))
    ).sum(dim=1)
    return (log_term + digamma_term).mean()


def evidential_loss(alpha, y, neg_weight=1.0, kl_weight=0.0):
    y_b = y.unsqueeze(1)
    y_onehot = torch.cat([y_b, 1.0 - y_b], dim=1)
    l_mse = evidential_mse_loss(alpha, y_onehot, neg_weight=neg_weight)
    if kl_weight > 0.0:
        return l_mse + kl_weight * evidential_kl_loss(alpha, y_onehot)
    return l_mse


def delta_u_pairwise_hinge_loss(u, p, y, margin=0.25, max_pairs_per_class=1500):
    """Hinge loss pushing u(wrong) above u(correct) by at least `margin`."""
    with torch.no_grad():
        # Must use the same operating point as the evaluation; otherwise the
        # loss optimizes a different correct/wrong partition than the one
        # macro_delta_u is measured on.
        y_pred = (p >= DECISION_THR_FIXED).float()
        correct_mask = (y_pred == y)
        wrong_mask = ~correct_mask
    n_c = int(correct_mask.sum().item())
    n_w = int(wrong_mask.sum().item())
    if n_c == 0 or n_w == 0:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    u_c = u[correct_mask]
    u_w = u[wrong_mask]
    if n_c > max_pairs_per_class:
        u_c = u_c[torch.randperm(n_c, device=u.device)[:max_pairs_per_class]]
    if n_w > max_pairs_per_class:
        u_w = u_w[torch.randperm(n_w, device=u.device)[:max_pairs_per_class]]
    gap = u_c.unsqueeze(1) - u_w.unsqueeze(0) + margin
    return torch.clamp(gap, min=0.0).mean()


def belief_correctness_loss(b, d, y):
    """
    Class-balanced belief-correctness loss.

    Averaged PER CLASS, then (loss_benign + loss_attacker) / 2, so both
    channels (b for benign, d for attacker) receive equal gradient regardless
    of how imbalanced an attack set is:

        Loss = 0.5 * (mean((1-b)^2 | y=1) + mean((1-d)^2 | y=0))
    """
    benign_mask = y > 0.5
    attacker_mask = ~benign_mask
    zero = torch.tensor(0.0, device=b.device, dtype=b.dtype)
    loss_b = ((1.0 - b[benign_mask]) ** 2).mean() if benign_mask.any() else zero
    loss_d = ((1.0 - d[attacker_mask]) ** 2).mean() if attacker_mask.any() else zero
    return 0.5 * (loss_b + loss_d)


# ============================================================================
# Data setup
# ============================================================================

class DataContext:
    """
    Holds all data needed by the sweep:
      - per-attack training sets (features on `device`)
      - validation + test features on CPU (for chunked inference)
      - feature_means / feature_stds (saved with the final model)
      - test metadata
    """
    def __init__(self, device, X_tr, y_tr, attack_tr,
                 X_va, y_va, attack_va, X_te, y_te, attack_te,
                 density_te, scenario_te):
        self.device = device
        self.y_va = y_va
        self.y_te = y_te
        self.attack_va = attack_va
        self.attack_te = attack_te
        self.density_te = density_te
        self.scenario_te = scenario_te

        # Attack groups for the validation macros, in the FIXED order of
        # selected_attacks (not sorted(set(attack_va))), so the group set does
        # not depend on which folders happen to be in the cache.
        self.attacks_va_list = [a for a in selected_attacks
                                if (np.asarray(attack_va).reshape(-1) == a).any()]
        missing = [a for a in selected_attacks if a not in self.attacks_va_list]
        if missing:
            print(f"  [WARN] in selected_attacks but not in the validation split: {missing}")

        # Per-feature standardization (statistics from train only)
        self.feature_means = {}
        self.feature_stds = {}
        self.Phi_va_by_f = {}
        self.Phi_te_by_f = {}
        Phi_tr_by_f = {}
        for j, fname in enumerate(FEATURE_NAMES):
            phi_tr, phi_va, phi_te, mean, std = _standardize_feature_with_na(
                X_tr[:, j], X_va[:, j], X_te[:, j]
            )
            self.feature_means[fname] = mean
            self.feature_stds[fname] = std
            Phi_tr_by_f[fname] = torch.from_numpy(phi_tr).to(device)
            self.Phi_va_by_f[fname] = torch.from_numpy(phi_va)  # CPU
            self.Phi_te_by_f[fname] = torch.from_numpy(phi_te)  # CPU

        self.basis_sizes = [Phi_tr_by_f[f].shape[1] for f in FEATURE_NAMES]
        self.total_dim = sum(self.basis_sizes)

        self.y_va_t = torch.from_numpy(y_va).float().squeeze(1).to(device)

        # One training set per attack, each with its own class-balance weight
        self.train_attack_sets = []
        for atk in selected_attacks:
            idx = np.where(attack_tr == atk)[0]
            if len(idx) == 0:
                continue
            Phi_attack = {f: Phi_tr_by_f[f][idx] for f in FEATURE_NAMES}
            y_attack = torch.from_numpy(y_tr[idx]).float().squeeze(1).to(device)
            n_attacker = int((y_attack == 0).sum().item())
            n_benign = int((y_attack == 1).sum().item())
            neg_w = (n_benign / n_attacker) if n_attacker > 0 else 1.0
            self.train_attack_sets.append({
                "attack": atk, "Phi": Phi_attack, "y": y_attack,
                "neg_weight": max(neg_w, 1.0),
            })
        del Phi_tr_by_f


def build_data_context(device, data_root=dc.DATA_ROOT_DEFAULT,
                       cache_dir=dc.CACHE_DIR_DEFAULT, verbose=True):
    if verbose:
        print("\n[Data] Loading from cache ...")
    dc.ensure_cache(data_root=data_root, cache_dir=cache_dir, verbose=verbose)
    X_tr, y_tr, attack_tr, _, _ = dc.load_split("Train", cache_dir)
    X_va, y_va, attack_va, _, _ = dc.load_split("Validation", cache_dir)
    X_te, y_te, attack_te, density_te, scenario_te = dc.load_split("Test", cache_dir)
    if verbose:
        print(f"  Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}  Test: {X_te.shape[0]}")
        print("[Data] Standardizing features (once) ...")
    ctx = DataContext(device, X_tr, y_tr, attack_tr,
                      X_va, y_va, attack_va, X_te, y_te, attack_te,
                      density_te, scenario_te)
    if verbose:
        print(f"  MLP input dim:     {ctx.total_dim}")
        print(f"  Train attack sets: {len(ctx.train_attack_sets)}")
    return ctx


# ============================================================================
# Training (inline, one call per trial)
# ============================================================================

def _snapshot(module):
    return {k: v.detach().clone().cpu() for k, v in module.state_dict().items()}


def train_single_trial(ctx, hp, num_epochs, val_check_every=VAL_CHECK_EVERY,
                       patience=PATIENCE, composite_delta_u_weight=0.3,
                       verbose=False):
    """
    Train one MLP with hyperparameters `hp` and return (best_state, val_metrics).

    hp keys: DELTA_U_WEIGHT_MAX, DELTA_U_MARGIN, BELIEF_WEIGHT,
             HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT, LR_MLP, WEIGHT_DECAY

    Model selection uses a composite validation loss
        L_val = L_evidential - composite_delta_u_weight * delta_u_val.

    val_metrics contains the five objectives plus descriptive delta_u variants.
    """
    device = ctx.device
    torch.manual_seed(SEED)

    hidden_dims = (int(hp["HIDDEN_DIM_1"]), int(hp["HIDDEN_DIM_2"]))
    dropout = float(hp["DROPOUT"])
    mlp_head = EvidentialMLPHead(ctx.total_dim, hidden_dims, dropout).to(device)

    opt = torch.optim.Adam(
        mlp_head.parameters(),
        lr=float(hp["LR_MLP"]),
        weight_decay=float(hp.get("WEIGHT_DECAY", WEIGHT_DECAY)),
    )

    delta_u_weight_max = float(hp["DELTA_U_WEIGHT_MAX"])
    delta_u_margin = float(hp["DELTA_U_MARGIN"])
    belief_weight = float(hp.get("BELIEF_WEIGHT", 0.0))
    # Only track the best checkpoint once all loss terms are fully ramped up.
    tracking_start = DELTA_U_START_EPOCH + DELTA_U_ANNEAL_STEPS + 50

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    inv_n_attacks = 1.0 / len(ctx.train_attack_sets)

    for epoch in range(num_epochs):
        mlp_head.train()
        kl_weight = min(1.0, epoch / KL_ANNEAL_STEPS)
        if epoch >= DELTA_U_START_EPOCH:
            delta_u_weight = min(
                delta_u_weight_max,
                delta_u_weight_max * (epoch - DELTA_U_START_EPOCH) / DELTA_U_ANNEAL_STEPS
            )
        else:
            delta_u_weight = 0.0

        # Full batch per attack with gradient accumulation: one forward/backward
        # over each complete attack set, gradients averaged over attacks, then a
        # single optimizer step per epoch. Only one attack graph is alive at a
        # time (freed after backward), so peak memory is bounded by the largest
        # single attack. Accumulating into one step also prevents attacks from
        # overwriting each other within an epoch.
        opt.zero_grad()
        for ds in ctx.train_attack_sets:
            p_a, (b_a, d_a, u_a), alpha_a = forward_prob_multi(
                ds["Phi"], mlp_head, FEATURE_NAMES)
            loss = evidential_loss(alpha_a, ds["y"],
                                   neg_weight=ds["neg_weight"], kl_weight=kl_weight)
            if delta_u_weight > 0.0:
                loss = loss + delta_u_weight * delta_u_pairwise_hinge_loss(
                    u_a, p_a, ds["y"], margin=delta_u_margin,
                    max_pairs_per_class=DELTA_U_MAX_PAIRS_PER_CLASS)
            if belief_weight > 0.0:
                loss = loss + belief_weight * belief_correctness_loss(b_a, d_a, ds["y"])
            (loss * inv_n_attacks).backward()
            del p_a, b_a, d_a, u_a, alpha_a, loss
        opt.step()

        if epoch % val_check_every != 0:
            continue

        mlp_head.eval()
        with torch.no_grad():
            p_va_cpu, fused_va_cpu, alpha_va_cpu = forward_prob_multi_chunked(
                ctx.Phi_va_by_f, mlp_head, FEATURE_NAMES, device=device)
            u_va_t = fused_va_cpu[2].to(device)
            p_va_t = p_va_cpu.to(device)
            alpha_va_t = alpha_va_cpu.to(device)
            val_loss = evidential_loss(alpha_va_t, ctx.y_va_t,
                                       neg_weight=1.0, kl_weight=kl_weight).item()
            correct = ((p_va_t >= DECISION_THR_FIXED).float() == ctx.y_va_t)
            if correct.any() and (~correct).any():
                delta_u_val = (u_va_t[~correct].mean() - u_va_t[correct].mean()).item()
                val_loss -= composite_delta_u_weight * delta_u_val
            del u_va_t, p_va_t, alpha_va_t
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if epoch < tracking_start:
            continue
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = _snapshot(mlp_head)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}.")
                break

    if best_state is None:
        best_state = _snapshot(mlp_head)

    # Final validation evaluation with the best checkpoint.
    eval_head = EvidentialMLPHead(ctx.total_dim, hidden_dims, dropout).to(device)
    eval_head.load_state_dict(best_state)
    eval_head.eval()
    # attacks_list must be set so that all objectives are true MACRO averages
    # over attacks. Attacker prevalence ranges from 2.2 % (suddenConstantSpeed)
    # to 52 % (dosAttack); a micro average would be dominated by the large,
    # balanced groups.
    val_metrics, _ = compute_macro_metrics_per_attack(
        ctx.Phi_va_by_f, eval_head, ctx.y_va,
        attack_arr=ctx.attack_va, attacks_list=ctx.attacks_va_list,
        device=device,
    )

    # Architecture metadata needed to rebuild the model later
    best_state["_meta_hidden_dims"] = hidden_dims
    best_state["_meta_dropout"] = dropout
    best_state["_meta_input_dim"] = ctx.total_dim

    return best_state, val_metrics


def _belief_correctness(b, d, y_true, y_pred):
    """Mean of b over correctly classified benign and d over correctly classified attackers."""
    correct = (y_pred == y_true)
    mask_b = (y_true == 1) & correct
    mask_a = (y_true == 0) & correct
    if not (mask_b.any() and mask_a.any()):
        return float("nan")
    return (float(b[mask_b].mean()) + float(d[mask_a].mean())) / 2.0


def compute_macro_metrics_per_attack(Phi_by_f, mlp_head, y_np,
                                     attack_arr, attacks_list, device):
    """Inference on all samples, then per-attack metrics and their macro average."""
    with torch.no_grad():
        p_t, fused_t, _ = forward_prob_multi_chunked(
            Phi_by_f, mlp_head, FEATURE_NAMES, device=device)
    p = p_t.numpy()
    b, d, u = (x.numpy() for x in fused_t)
    y_true = y_np.reshape(-1)
    y_pred = (p >= DECISION_THR_FIXED).astype(int)

    per_attack = {}
    for atk in attacks_list:
        idx = np.where(attack_arr == atk)[0]
        if len(idx) == 0:
            continue
        yt, yp, ui = y_true[idx], y_pred[idx], u[idx]
        m = evaluate_all(y_true=yt, y_prob=p[idx], y_pred=yp, uncertainty=ui)
        correct = (yp == yt)
        u_c, u_w = ui[correct], ui[~correct]
        has_both = len(u_c) and len(u_w)
        per_attack[atk] = {
            "f1": _f1_attacker(yt, yp),
            "aurc": m.get("aurc", float("nan")),
            "misclass_auroc": m.get("misclass_auroc", float("nan")),
            # Objective: MEAN over samples (level 1), mean over attacks (level 2)
            "delta_u": (u_w.mean() - u_c.mean()) if has_both else float("nan"),
            # Descriptive only (not optimized): shows how much the choice of
            # estimator moves the result.
            "delta_u_median": (np.median(u_w) - np.median(u_c)) if has_both else float("nan"),
            "delta_u_cohens_d": delta_u_cohens_d(yt, yp, ui),
            "belief_correctness": _belief_correctness(b[idx], d[idx], yt, yp),
        }

    def _mm(key):
        vs = [pa[key] for pa in per_attack.values() if not np.isnan(pa[key])]
        return float(np.mean(vs)) if vs else float("nan")

    return {
        "macro_f1":                 _mm("f1"),
        "macro_aurc":               _mm("aurc"),
        "macro_misclass_auroc":     _mm("misclass_auroc"),
        "macro_delta_u":            _mm("delta_u"),
        "macro_delta_u_median":     _mm("delta_u_median"),
        "macro_delta_u_cohens_d":   _mm("delta_u_cohens_d"),
        "macro_belief_correctness": _mm("belief_correctness"),
    }, per_attack


def _f1_attacker_macro_named(y_true, y_pred, attack_arr, attacks_list):
    """
    Macro F1 (attacker = positive class) over the groups in attacks_list.

    Same semantics as the macro branch of compute_macro_metrics_per_attack, but
    works on precomputed predictions (used for threshold diagnostics).
    """
    a = np.asarray(attack_arr).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    f1s = []
    for atk in attacks_list:
        idx = np.where(a == atk)[0]
        if len(idx):
            f1s.append(_f1_attacker(y_true[idx], y_pred[idx]))
    return float(np.mean(f1s)) if f1s else float("nan")


def _f1_attacker(y_true, y_pred):
    """F1 score with the attacker class (label 0) as positive class."""
    tp = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp) / denom if denom > 0 else 0.0


# ============================================================================
# Trial state storage
# ============================================================================

def save_trial_states(states, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(states, path)


def load_trial_states(path):
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location="cpu")


# ============================================================================
# Knee-point selection
# ============================================================================

METRICS_FOR_KNEE = list(zip(OBJECTIVE_NAMES, OBJECTIVE_DIRECTION))
TIEBREAKER_PRIORITY = ["chebyshev", "closest_to_utopia",
                       "weighted_sum", "farthest_from_nadir"]


def normalize_trials_for_knee(trials_metrics):
    """Min-max normalize every objective to [0, 1] with 1 = best."""
    Z = np.zeros((len(trials_metrics), len(METRICS_FOR_KNEE)), dtype=np.float64)
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
    return int(np.argmin(np.linalg.norm(Z - 1.0, axis=1)))


def knee_farthest_from_nadir(Z):
    return int(np.argmax(np.linalg.norm(Z, axis=1)))


def knee_chebyshev(Z):
    return int(np.argmin(np.max(np.abs(Z - 1.0), axis=1)))


def knee_weighted_sum(Z):
    return int(np.argmax(Z.mean(axis=1)))


KNEE_METHODS = {
    "closest_to_utopia":   knee_closest_to_utopia,
    "farthest_from_nadir": knee_farthest_from_nadir,
    "chebyshev":           knee_chebyshev,
    "weighted_sum":        knee_weighted_sum,
}


def select_balanced_trial(completed, verbose=True):
    """Majority vote over the four knee methods; ties broken by TIEBREAKER_PRIORITY."""
    completed = [t for t in completed
                 if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if not completed:
        raise RuntimeError("No completed trials to select from.")
    trials_data = []
    for t in completed:
        md = {name: t.values[j] for j, name in enumerate(OBJECTIVE_NAMES)}
        md["trial_number"] = t.number
        trials_data.append(md)

    Z = normalize_trials_for_knee(trials_data)
    knee_results = {}
    if verbose:
        print(f"\n{'='*70}")
        print(f"KNEE-POINT SELECTION ({len(trials_data)} trials)")
        print(f"{'='*70}")
    for name, fn in KNEE_METHODS.items():
        idx = fn(Z)
        tnum = trials_data[idx]["trial_number"]
        knee_results[name] = {
            "trial_number": tnum,
            "metrics": {k: trials_data[idx][k] for k, _ in METRICS_FOR_KNEE},
        }
        if verbose:
            print(f"\n[{name}]  -> trial {tnum}")
            for k, _ in METRICS_FOR_KNEE:
                print(f"    {k:30s} = {trials_data[idx][k]:.4f}")

    votes = Counter(r["trial_number"] for r in knee_results.values())
    methods_per_trial = defaultdict(list)
    for mname, r in knee_results.items():
        methods_per_trial[r["trial_number"]].append(mname)
    sorted_votes = votes.most_common()
    top_count = sorted_votes[0][1]
    top_trials = [tn for tn, c in sorted_votes if c == top_count]
    info = {
        "votes_per_trial":    dict(votes),
        "methods_per_trial":  dict(methods_per_trial),
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
            if knee_results[tb]["trial_number"] in top_trials:
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
        print(f"CONSENSUS WINNER: trial {winner}  "
              f"({top_count}/{len(knee_results)} votes = "
              f"{100*info['consensus_strength']:.0f}%)")
        if info["tie_breaker_used"]:
            print(f"  Tie-breaker: {info['tie_breaker_method']}")
        print(f"{'='*70}")
    return winner, info


# ============================================================================
# Paper outputs for the final test run
# ============================================================================

def _render_extended_boxplot(boxplot_samples, out_path, title_suffix=""):
    """
    Boxplot with 12 categories, grouped by SL component:
        group U:  u_benign | u_malicious | u_correct | u_wrong
        group B:  b_benign | b_malicious | b_correct | b_wrong
        group D:  d_benign | d_malicious | d_correct | d_wrong
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
        ("u_benign", "u — benign"), ("u_malicious", "u — malicious"),
        ("u_correct", "u — correct"), ("u_wrong", "u — misclassified"),
        ("b_benign", "b — benign"), ("b_malicious", "b — malicious"),
        ("b_correct", "b — correct"), ("b_wrong", "b — misclassified"),
        ("d_benign", "d — benign"), ("d_malicious", "d — malicious"),
        ("d_correct", "d — correct"), ("d_wrong", "d — misclassified"),
    ]
    BOX_SPACING = 1.0
    GROUP_GAP = 1.2
    v_data, v_pos, v_col, v_lbl = [], [], [], []
    cursor = 1.0
    for grp_start in (0, 4, 8):
        for key, lbl in order[grp_start:grp_start + 4]:
            arr = np.asarray(boxplot_samples.get(key, []), dtype=np.float64)
            if arr.size and not np.isnan(arr).all():
                v_data.append(arr); v_pos.append(cursor)
                v_col.append(colors[key]); v_lbl.append(lbl)
            cursor += BOX_SPACING
        cursor += GROUP_GAP

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
    # Group separators between U | B | D (just after the last box of a group)
    group_width = 4 * BOX_SPACING + GROUP_GAP
    for k in (0, 1):
        ax.axvline(1.0 + k * group_width + 3 * BOX_SPACING + 0.6,
                   color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(out_path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close()


def _nan_stats(a, with_std=False):
    if len(a) == 0:
        return (float("nan"),) * (3 if with_std else 2)
    stats = (float(a.mean()), float(np.median(a)))
    return stats + (float(a.std()),) if with_std else stats


def _risk_at_coverage(u, correct, cov):
    """Error rate among the `cov` fraction of samples with the lowest uncertainty."""
    n = len(u)
    if n == 0:
        return float("nan")
    k = max(1, int(cov * n))
    order = np.argsort(u)
    return float((~correct[order]).astype(float)[:k].mean())


def _write_per_sample_csv(fcsv, atk, y_t, yp_t, correct_t, p_t, b_t, d_t, u_t):
    # .tolist() avoids per-cell numpy scalar boxing; chunked "".join() bounds
    # memory and issues one write per chunk (fewer syscalls on network mounts).
    cols = [y_t.astype(np.int64).tolist(), yp_t.astype(np.int64).tolist(),
            correct_t.astype(np.int64).tolist(),
            p_t.tolist(), b_t.tolist(), d_t.tolist(), u_t.tolist()]
    n = len(y_t)
    for s0 in range(0, n, CSV_WRITE_CHUNK):
        s1 = min(s0 + CSV_WRITE_CHUNK, n)
        fcsv.write("".join(
            f"{atk},{yt_},{yp_},{c_},{p_:.6f},{b_:.6f},{d_:.6f},{u_:.6f}\n"
            for yt_, yp_, c_, p_, b_, d_, u_ in zip(*(c[s0:s1] for c in cols))
        ))


def build_paper_outputs_final(p_all, u_all, b_all, d_all,
                              y_pred_all, y_true_all,
                              attack_all, method_name, out_dir,
                              boxplot_n=50_000):
    """Write paper tables 1-5, boxplots, results JSON and the per-sample CSV."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    per_attack_results = {}
    pooled_idx_list = []

    # Write locally, then move to out_dir (see CSV_LOCAL_TMP_DIR).
    csv_path = os.path.join(out_dir, f"opinions_per_sample_{method_name}.csv")
    if CSV_LOCAL_TMP_DIR is not None:
        os.makedirs(CSV_LOCAL_TMP_DIR, exist_ok=True)
        tmp_fd, write_target = tempfile.mkstemp(
            suffix=".csv", prefix=f"opinions_{method_name}_", dir=CSV_LOCAL_TMP_DIR)
        os.close(tmp_fd)
    else:
        write_target = csv_path

    with open(write_target, "w", encoding="utf-8", buffering=1 << 20) as fcsv:
        fcsv.write("attack,y_true,y_pred,correct,p,b,d,u\n")
        for atk in selected_attacks:
            idx = np.where(attack_all == atk)[0]
            if len(idx) == 0:
                continue
            y_t = y_true_all[idx]
            yp_t = y_pred_all[idx]
            correct_t = (yp_t == y_t)
            p_t = p_all[idx]; b_t = b_all[idx]; d_t = d_all[idx]; u_t = u_all[idx]

            _write_per_sample_csv(fcsv, atk, y_t, yp_t, correct_t, p_t, b_t, d_t, u_t)

            m = evaluate_all(y_true=y_t, y_prob=p_t, y_pred=yp_t, uncertainty=u_t)
            mask_benign = (y_t == 1)
            mask_malicious = (y_t == 0)
            mask_bc = mask_benign & correct_t
            mask_ac = mask_malicious & correct_t
            bcb = float(b_t[mask_bc].mean()) if mask_bc.any() else float("nan")
            dca = float(d_t[mask_ac].mean()) if mask_ac.any() else float("nan")

            entry = {
                "f1": _f1_attacker(y_t, yp_t),
                "aurc": m.get("aurc", float("nan")),
                "misclass_auroc": m.get("misclass_auroc", float("nan")),
                "ece": m.get("ece", float("nan")),
                "brier": m.get("brier", float("nan")),
            }
            uc_m, uc_md, uc_sd = _nan_stats(u_t[correct_t], with_std=True)
            uw_m, uw_md, uw_sd = _nan_stats(u_t[~correct_t], with_std=True)
            entry.update({
                "u_correct_mean": uc_m, "u_correct_median": uc_md, "u_correct_std": uc_sd,
                "u_wrong_mean":   uw_m, "u_wrong_median":   uw_md, "u_wrong_std":   uw_sd,
            })
            subsets = {"correct": correct_t, "wrong": ~correct_t,
                       "benign": mask_benign, "malicious": mask_malicious}
            for comp, arr in (("u", u_t), ("b", b_t), ("d", d_t)):
                for sname, smask in subsets.items():
                    if comp == "u" and sname in ("correct", "wrong"):
                        continue   # already added above (with std)
                    mean, median = _nan_stats(arr[smask])
                    entry[f"{comp}_{sname}_mean"] = mean
                    entry[f"{comp}_{sname}_median"] = median
            entry.update({
                "b_correct_benign_mean":   bcb,
                "d_correct_attacker_mean": dca,
                "belief_correctness":      (float("nan") if np.isnan(bcb) or np.isnan(dca)
                                            else (bcb + dca) / 2.0),
                "n_correct":   int(correct_t.sum()),
                "n_wrong":     int((~correct_t).sum()),
                "n_benign":    int(mask_benign.sum()),
                "n_malicious": int(mask_malicious.sum()),
            })
            per_attack_results[atk] = entry
            pooled_idx_list.append(idx)
    if CSV_LOCAL_TMP_DIR is not None:
        shutil.move(write_target, csv_path)
    print(f"  per-sample CSV: {csv_path}")

    if not pooled_idx_list:
        print("  [WARN] no test data")
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
    m_pool = evaluate_all(y_true=y_pool, y_prob=p_pool, y_pred=yp_pool, uncertainty=u_pool)
    u_c_pool = u_pool[correct_pool]
    u_w_pool = u_pool[~correct_pool]
    has_both = len(u_c_pool) and len(u_w_pool)

    overall_micro = {
        "f1": _f1_attacker(y_pool, yp_pool),
        "aurc": m_pool.get("aurc", float("nan")),
        "misclass_auroc": m_pool.get("misclass_auroc", float("nan")),
        "ece": m_pool.get("ece", float("nan")),
        "brier": m_pool.get("brier", float("nan")),
        "risk_at_70":  _risk_at_coverage(u_pool, correct_pool, 0.70),
        "risk_at_80":  _risk_at_coverage(u_pool, correct_pool, 0.80),
        "risk_at_90":  _risk_at_coverage(u_pool, correct_pool, 0.90),
        "risk_at_100": float(1.0 - correct_pool.mean()),
        "delta_u_mean":   float(u_w_pool.mean() - u_c_pool.mean())
                          if has_both else float("nan"),
        "delta_u_median": float(np.median(u_w_pool) - np.median(u_c_pool))
                          if has_both else float("nan"),
        "delta_u_cohens_d": delta_u_cohens_d(y_pool, yp_pool, u_pool),
    }

    def _macro(key):
        vs = [pa[key] for pa in per_attack_results.values() if not np.isnan(pa[key])]
        return float(np.mean(vs)) if vs else float("nan")

    def _macro_diff(stat):
        # Per-attack u_wrong - u_correct, then mean over attacks. The mean
        # variant is the objective; the median variant is descriptive (the
        # mean-vs-median gap reflects the confident-wrong tail).
        diffs = [pa[f"u_wrong_{stat}"] - pa[f"u_correct_{stat}"]
                 for pa in per_attack_results.values()
                 if not (np.isnan(pa[f"u_wrong_{stat}"]) or np.isnan(pa[f"u_correct_{stat}"]))]
        return float(np.mean(diffs)) if diffs else float("nan")

    macro = {
        "macro_f1":                 _macro("f1"),
        "macro_aurc":               _macro("aurc"),
        "macro_misclass_auroc":     _macro("misclass_auroc"),
        "macro_ece":                _macro("ece"),
        "macro_brier":              _macro("brier"),
        "macro_delta_u":            _macro_diff("mean"),
        "macro_delta_u_median":     _macro_diff("median"),
        "macro_belief_correctness": _macro("belief_correctness"),
    }

    def _sample(arr):
        arr = np.asarray(arr)
        return arr if len(arr) <= boxplot_n else rng.choice(arr, size=boxplot_n, replace=False)

    pool_subsets = {"correct": correct_pool, "wrong": ~correct_pool,
                    "benign": benign_pool, "malicious": malicious_pool}
    boxplot_samples = {
        f"{comp}_{sname}": _sample(arr[smask])
        for comp, arr in (("u", u_pool), ("b", b_pool), ("d", d_pool))
        for sname, smask in pool_subsets.items()
    }
    method_entry = {
        "method_name":     method_name,
        "macro":           macro,
        "overall_micro":   overall_micro,
        "per_attack":      per_attack_results,
        "boxplot_samples": boxplot_samples,
    }
    json_out = {
        **{k: v for k, v in method_entry.items() if k != "boxplot_samples"},
        "boxplot_samples": {k: v.tolist() for k, v in boxplot_samples.items()},
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
    _render_extended_boxplot(boxplot_samples, ext_box_path,
                             title_suffix=f"({method_name})")
    print(f"  extended boxplot: {ext_box_path}")

    print(f"\n  Macro F1: {macro['macro_f1']:.4f}   AURC: {macro['macro_aurc']:.4f}   "
          f"MAUROC: {macro['macro_misclass_auroc']:.4f}   Δu: {macro['macro_delta_u']:.4f}   "
          f"BeliefCorr: {macro['macro_belief_correctness']:.4f}")
    return method_entry


def run_final_test(ctx, best_state, method_name, out_dir):
    """
    Rebuild the MLP from best_state, run inference on the test split, then
    write paper outputs and save the model.
    """
    print(f"\n{'='*70}")
    print(f"FINAL TEST RUN (MLP): {method_name}")
    print(f"{'='*70}")
    device = ctx.device

    hidden_dims = best_state["_meta_hidden_dims"]
    dropout = best_state["_meta_dropout"]
    input_dim = best_state["_meta_input_dim"]
    mlp_head = EvidentialMLPHead(input_dim, hidden_dims, dropout).to(device)
    state_dict_pure = {k: v for k, v in best_state.items() if not k.startswith("_meta_")}
    mlp_head.load_state_dict(state_dict_pure)
    mlp_head.eval()

    # Validation inference (for threshold diagnostics)
    with torch.no_grad():
        p_va_cpu, _, _ = forward_prob_multi_chunked(
            ctx.Phi_va_by_f, mlp_head, FEATURE_NAMES, device=device)
    p_va_np = p_va_cpu.numpy()
    y_va_flat = ctx.y_va.reshape(-1)

    # Operating point = DECISION_THR_FIXED, identical to the sweep; NO refit.
    # misclass_auroc, delta_u and aurc are all defined via
    # correct = (y_pred == y_true). Refitting the threshold would shift that
    # partition away from the point at which the trial was selected.
    t_star = float(DECISION_THR_FIXED)
    val_macro_f1 = _f1_attacker_macro_named(y_va_flat, (p_va_np >= t_star).astype(int),
                                            ctx.attack_va, ctx.attacks_va_list)
    print(f"  Operating point = sweep point {t_star:.3f}  "
          f"(val macro-F1 = {val_macro_f1:.4f})")

    # For transparency only: what a threshold refit would have given (NOT used).
    ts = np.linspace(0.01, 0.99, 199)
    f1s = [_f1_attacker_macro_named(y_va_flat, (p_va_np >= t).astype(int),
                                    ctx.attack_va, ctx.attacks_va_list) for t in ts]
    t_refit = float(ts[int(np.argmax(f1s))])
    print(f"  [Info] macro-F1 optimum would be t={t_refit:.3f} "
          f"(val macro-F1={max(f1s):.4f}) -- intentionally NOT used.")

    with torch.no_grad():
        p_te_t, fused_te, _ = forward_prob_multi_chunked(
            ctx.Phi_te_by_f, mlp_head, FEATURE_NAMES, device=device)
    p_all = p_te_t.numpy()
    b_all, d_all, u_all = (x.numpy() for x in fused_te)
    y_true = ctx.y_te.reshape(-1)
    y_pred = (p_all >= t_star).astype(int)

    method_entry = build_paper_outputs_final(
        p_all=p_all, u_all=u_all, b_all=b_all, d_all=d_all,
        y_pred_all=y_pred, y_true_all=y_true,
        attack_all=ctx.attack_te,
        method_name=method_name, out_dir=out_dir,
    )

    model = {
        "variant":         "mlp_raw_with_delta_u",
        "feature_means":   ctx.feature_means,
        "feature_stds":    ctx.feature_stds,
        "na_sentinel":     NA_SENTINEL,
        "use_na_mask":     USE_NA_MASK,
        "mlp_state_dict":  state_dict_pure,
        "mlp_input_dim":   input_dim,
        "mlp_hidden_dims": hidden_dims,
        "mlp_dropout":     dropout,
        "threshold":       t_star,
        "features":        FEATURE_NAMES,
    }
    model_path = os.path.join(out_dir, f"mlp_raw_model_{method_name}.pt")
    torch.save(model, model_path)
    print(f"\n  Model: {model_path}")
    return method_entry


# ============================================================================
# Optuna study
# ============================================================================

def create_or_load_study(storage, study_name, resume=False):
    sampler = optuna.samplers.NSGAIISampler(
        population_size=NSGA_POPULATION_SIZE,
        crossover=optuna.samplers.nsgaii.UniformCrossover(),
        crossover_prob=NSGA_CROSSOVER_PROB,
        swapping_prob=NSGA_SWAPPING_PROB,
        seed=SEED,
    )
    if resume:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage,
                                      sampler=sampler)
            if len(study.directions) != len(OBJECTIVE_DIRECTION):
                raise RuntimeError(
                    f"Schema mismatch: study has {len(study.directions)} objectives, "
                    f"expected {len(OBJECTIVE_DIRECTION)}.")
            print(f"[Optuna] Study loaded — "
                  f"{len(study.get_trials(deepcopy=False))} trials so far.")
            return study
        except KeyError:
            print("[Optuna] No existing study found, creating a new one.")
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        directions=OBJECTIVE_DIRECTION, sampler=sampler,
        load_if_exists=True,
    )
    if len(study.directions) != len(OBJECTIVE_DIRECTION):
        raise RuntimeError(
            "Schema mismatch on creation. Delete the study database or "
            "choose a different --study-name.")
    return study


def suggest_hyperparameters(trial):
    """Sample one configuration from SEARCH_SPACE_FLOAT / SEARCH_SPACE_CATEGORICAL."""
    hp = {}
    for name, (low, high, step, log) in SEARCH_SPACE_FLOAT.items():
        hp[name] = trial.suggest_float(name, low, high, step=step, log=log)
    for name, choices in SEARCH_SPACE_CATEGORICAL.items():
        hp[name] = trial.suggest_categorical(name, choices)
    hp["WEIGHT_DECAY"] = WEIGHT_DECAY
    return hp


def write_summary_csv(all_trials, pareto_set, path):
    rows = []
    for t in all_trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": t.number,
            "method_name":  t.user_attrs.get("method_name", f"trial_{t.number:03d}"),
            "is_pareto":    t.number in pareto_set,
            "duration_min": t.user_attrs.get("duration_min", 0),
            **t.params,
        }
        for i, name in enumerate(OBJECTIVE_NAMES):
            row[name] = t.values[i] if t.values else float("nan")
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {path}")


def plot_pareto_2d(completed, pareto_set, obj_x, obj_y, out_path):
    if not completed:
        return
    idx_x = OBJECTIVE_NAMES.index(obj_x)
    idx_y = OBJECTIVE_NAMES.index(obj_y)
    dir_x = OBJECTIVE_DIRECTION[idx_x]
    dir_y = OBJECTIVE_DIRECTION[idx_y]
    fig, ax = plt.subplots(figsize=(9, 6))
    np_x = [t.values[idx_x] for t in completed if t.number not in pareto_set]
    np_y = [t.values[idx_y] for t in completed if t.number not in pareto_set]
    p_xy = sorted((t.values[idx_x], t.values[idx_y])
                  for t in completed if t.number in pareto_set)
    if np_x:
        ax.scatter(np_x, np_y, c="#888888", s=60, alpha=0.5,
                   label=f"dominated (n={len(np_x)})")
    if p_xy:
        sx, sy = zip(*p_xy)
        ax.scatter(sx, sy, c="#D85A30", s=120, alpha=0.95, edgecolors="black",
                   linewidths=1.2, label=f"Pareto (n={len(p_xy)})", zorder=3)
        ax.plot(sx, sy, c="#D85A30", alpha=0.4, linestyle="--", zorder=2)
    for t in completed:
        ax.annotate(str(t.number), (t.values[idx_x], t.values[idx_y]),
                    fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"{obj_x}  ({'↑' if dir_x == 'maximize' else '↓'})")
    ax.set_ylabel(f"{obj_y}  ({'↑' if dir_y == 'maximize' else '↓'})")
    ax.set_title(f"MLP Pareto: {obj_x} vs {obj_y}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=1000,
                        help="Total number of completed trials to reach (default: 1000)")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS_PER_TRIAL,
                        help=f"Max. epochs per trial (default: {NUM_EPOCHS_PER_TRIAL})")
    parser.add_argument("--data-root", default=dc.DATA_ROOT_DEFAULT,
                        help="Raw JSON root, used only if the cache must be built "
                             f"(default: $MBD_DATA_ROOT or {dc.DATA_ROOT_DEFAULT})")
    parser.add_argument("--cache-dir", default=dc.CACHE_DIR_DEFAULT,
                        help="Feature cache directory "
                             f"(default: $MBD_CACHE_DIR or {dc.CACHE_DIR_DEFAULT})")
    parser.add_argument("--out-dir", default=SWEEP_ROOT,
                        help=f"Output directory (default: {SWEEP_ROOT})")
    parser.add_argument("--storage", default=STUDY_DB,
                        help=f"Optuna storage URL (default: {STUDY_DB})")
    parser.add_argument("--study-name", default=STUDY_NAME,
                        help=f"Optuna study name (default: {STUDY_NAME})")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an existing study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip the sweep; run analysis, knee selection and final test")
    parser.add_argument("--final-test-only", action="store_true",
                        help="Skip sweep and Pareto plots; run knee selection and final test")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Run on CPU even if CUDA is available")
    args = parser.parse_args()

    out_dir = args.out_dir
    trial_states_file = os.path.join(out_dir, "trial_states.pt")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[Sweep MLP] Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Sweep MLP] Objectives: {OBJECTIVE_NAMES}")
    print(f"[Sweep MLP] Directions: {OBJECTIVE_DIRECTION}")
    print(f"[Sweep MLP] Output:     {out_dir}/")

    ctx = build_data_context(device, data_root=args.data_root,
                             cache_dir=args.cache_dir, verbose=True)

    skip_sweep = args.analyze_only or args.final_test_only
    study = create_or_load_study(args.storage, args.study_name,
                                 resume=args.resume or skip_sweep)

    # ---------- Phase 1: sweep ----------
    if not skip_sweep:
        trial_states = load_trial_states(trial_states_file)
        n_completed = sum(1 for t in study.get_trials(deepcopy=False)
                          if t.state == optuna.trial.TrialState.COMPLETE)
        n_remaining = args.n_trials - n_completed
        if n_remaining <= 0:
            print(f"\n[Sweep] Target ({args.n_trials}) already reached "
                  f"({n_completed} trials).")
        else:
            print(f"\n[Sweep] {n_completed} done, running {n_remaining} more trials")

            def objective(trial):
                hp = suggest_hyperparameters(trial)
                composite_w = hp.pop("COMPOSITE_DELTA_U_WEIGHT")
                t_start = time.time()
                print(f"\n{'─'*60}")
                print(f"Trial {trial.number}:")
                for k, v in hp.items():
                    print(f"  {k:25s} = {v}")
                print(f"  {'COMPOSITE_DELTA_U_WEIGHT':25s} = {composite_w:.3f}")
                best_state, val_metrics = train_single_trial(
                    ctx, hp, num_epochs=args.num_epochs,
                    composite_delta_u_weight=composite_w,
                )
                dt = time.time() - t_start
                trial_states[trial.number] = best_state
                save_trial_states(trial_states, trial_states_file)
                trial.set_user_attr("method_name", f"trial_{trial.number:03d}")
                trial.set_user_attr("duration_min", dt / 60)
                for k, v in val_metrics.items():
                    trial.set_user_attr(k, v)
                print(f"  done in {dt/60:.1f} min:  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                for name in OBJECTIVE_NAMES:
                    if np.isnan(val_metrics.get(name, float("nan"))):
                        print(f"  [WARN] {name} is NaN -> pruning trial")
                        raise optuna.TrialPruned()
                return tuple(val_metrics[n] for n in OBJECTIVE_NAMES)

            t_sweep = time.time()
            study.optimize(objective, n_trials=n_remaining,
                           gc_after_trial=True, show_progress_bar=False)
            print(f"\n[Sweep] Total sweep time: {(time.time()-t_sweep)/60:.1f} min")

    # ---------- Phase 2: analysis ----------
    # Materialize trials once: study.trials deep-copies on every access and
    # study.best_trials recomputes the Pareto front each time, which is very
    # slow for large studies.
    all_trials = study.get_trials(deepcopy=False)
    completed = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[ERROR] No completed trials — aborting.")
        return

    if not args.final_test_only:
        pareto_set = {t.number for t in study.best_trials}
        print(f"\n{'='*70}")
        print(f"SWEEP ANALYSIS  ({len(completed)} completed trials)")
        print(f"{'='*70}")
        print(f"  Pareto front: {len(pareto_set)} trials")
        write_summary_csv(all_trials, pareto_set, os.path.join(out_dir, "sweep_summary.csv"))
        for i in range(len(OBJECTIVE_NAMES)):
            for j in range(i + 1, len(OBJECTIVE_NAMES)):
                plot_pareto_2d(completed, pareto_set, OBJECTIVE_NAMES[i], OBJECTIVE_NAMES[j],
                               os.path.join(out_dir,
                                            f"pareto_{OBJECTIVE_NAMES[i]}_vs_{OBJECTIVE_NAMES[j]}.png"))

    # ---------- Phase 3: knee selection ----------
    winner, info = select_balanced_trial(completed, verbose=True)
    balanced_path = os.path.join(out_dir, "balanced_trial.json")
    with open(balanced_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"  balanced JSON: {balanced_path}")

    # ---------- Phase 4: final test with the balanced trial ----------
    trial_states = load_trial_states(trial_states_file)
    if winner not in trial_states:
        print(f"\n[ERROR] Best state for trial {winner} missing in {trial_states_file}.")
        return
    method_name = f"MLP_balanced_trial{winner:03d}"
    final_dir = os.path.join(out_dir, "final_test")
    run_final_test(ctx, trial_states[winner], method_name=method_name, out_dir=final_dir)
    print(f"\n[Sweep MLP] Done. Final outputs in {final_dir}/")


if __name__ == "__main__":
    main()
