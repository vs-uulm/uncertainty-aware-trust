"""
Optuna-based multi-objective sweep for the MLP — V2 with inline training and
a data cache (analogous to sweep_optuna_v2.py for the B-Spline model).

Model: an evidential MLP head over standardized raw features.
Standard-SL convention (b = e/S, d = e/S, u = 2/S, p = alpha_b/S).

Usage:
    python data_cache.py --build
    python sweep_optuna_mlp.py --n-trials 20

    python sweep_optuna_mlp.py --resume --n-trials 30
    python sweep_optuna_mlp.py --analyze-only
"""
import argparse
import csv
import json
import os
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

# Default output directory, relative to the working directory. Override with
# --output-dir.
DEFAULT_OUTPUT_DIR = "sweep_output"
STUDY_DB = "sqlite:///mlp_optuna_v2.db"
# Bump the study name whenever objectives, model or search space change or
# selected_attacks changes. With --resume, NSGA-II would otherwise mix
# incompatible trials into one Pareto front. The objectives are macro averages
# over attacks, and delta_u uses the MEAN over samples (the median has a 50 %
# breakdown point and is blind to the confident-wrong tail).
STUDY_NAME = "mlp_pareto_v4_5obj_meandu"

# Operating point for ALL evaluations — sweep AND final test.
# p = b + 0.5*u = 0.5 + 0.5*(b - d)  =>  p >= 0.5  <=>  b >= d.
# The SL decision rule "more evidence wins", the only point without a free
# parameter, identical to sweep_optuna.py (HP) and sweep_optuna_v2.py (B-Spline).
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

# Trial defaults
NUM_EPOCHS_PER_TRIAL = 3000
KL_ANNEAL_STEPS = 200
DELTA_U_START_EPOCH = 0
DELTA_U_ANNEAL_STEPS = 250
DELTA_U_MAX_PAIRS_PER_CLASS = 1500

TRIAL_STATES_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "trial_states.pt")


# ============================================================================
# Model (MLP head, standard SL)
# ============================================================================

class EvidentialMLPHead(nn.Module):
    """
    Shallow MLP head over raw features.

    Input : (N, input_dim)
    Output: (N, 2) evidence values (e_benign, e_attacker), both >= 0 via
            softplus.
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
    """Standard-SL evidential joint opinion."""
    evidence = mlp_head(Phi_concat)
    e_b = evidence[:, 0]
    e_a = evidence[:, 1]
    alpha_b = e_b + 1.0
    alpha_a = e_a + 1.0
    S = alpha_b + alpha_a
    b = e_b / S                 # standard SL: b = e/S
    d = e_a / S
    u = 2.0 / S
    u = torch.clamp(u, min=U_MIN)
    return b, d, u


def forward_prob_multi(Phi_by_f, mlp_head, feature_names, base_rate=BASE_RATE):
    Phi_concat = torch.cat([Phi_by_f[f] for f in feature_names], dim=1)
    b, d, u = opinion_joint(Phi_concat, mlp_head)
    # alpha reconstruction (standard SL): alpha_k = b_k * S + 1
    S = 2.0 / u
    alpha_b = b * S + 1.0
    alpha_a = d * S + 1.0
    alpha = torch.stack([alpha_b, alpha_a], dim=1)
    p = b + base_rate * u
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    return p, (b, d, u), alpha


def forward_prob_multi_chunked(Phi_by_f_cpu, mlp_head, feature_names, device,
                                base_rate=BASE_RATE, chunk_size=200_000):
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
        l_kl = evidential_kl_loss(alpha, y_onehot)
        return l_mse + kl_weight * l_kl
    return l_mse


def delta_u_pairwise_hinge_loss(u, p, y, margin=0.25, max_pairs_per_class=1500):
    with torch.no_grad():
        # The operating point MUST match the evaluation point, otherwise the
        # hinge loss optimizes a different correct/wrong partition than the
        # one macro_delta_u is measured against.
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
        idx = torch.randperm(n_c, device=u.device)[:max_pairs_per_class]
        u_c = u_c[idx]
    if n_w > max_pairs_per_class:
        idx = torch.randperm(n_w, device=u.device)[:max_pairs_per_class]
        u_w = u_w[idx]
    gap = u_c.unsqueeze(1) - u_w.unsqueeze(0) + margin
    return torch.clamp(gap, min=0.0).mean()


def belief_correctness_loss(b, d, y):
    """
    Class-balanced variant: averages PER CLASS, then (loss_benign + loss_attacker) / 2.

    This gives both channels (b for benign, d for attacker) equal gradient
    weight regardless of how skewed the attack set is.

    Loss = 0.5 * (mean((1-b)^2 | y=1) + mean((1-d)^2 | y=0))
    """
    benign_mask   = y > 0.5
    attacker_mask = ~benign_mask
    if benign_mask.any():
        loss_b = ((1.0 - b[benign_mask]) ** 2).mean()
    else:
        loss_b = torch.tensor(0.0, device=b.device, dtype=b.dtype)
    if attacker_mask.any():
        loss_d = ((1.0 - d[attacker_mask]) ** 2).mean()
    else:
        loss_d = torch.tensor(0.0, device=b.device, dtype=b.dtype)
    return 0.5 * (loss_b + loss_d)


# ============================================================================
# Data setup
# ============================================================================

class DataContext:
    """
    Holds:
      - Phi_by_f on device for Train (per attack)
      - Phi_by_f on CPU for Val + Test (chunked forward)
      - feature_means / feature_stds (for model saving)
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

        # Attack groups for the VAL macros. Deliberately selected_attacks in
        # FIXED order instead of sorted(set(attack_va)) — otherwise the group
        # set would depend on which folders happen to be present in the
        # cache. Identical to sweep_optuna.py (HP).
        self.attacks_va_list = [a for a in selected_attacks
                                if (np.asarray(attack_va).reshape(-1) == a).any()]
        _missing = [a for a in selected_attacks if a not in self.attacks_va_list]
        if _missing:
            print(f"  [WARN] in selected_attacks but not in the Val split: {_missing}")
        self.density_te = density_te
        self.scenario_te = scenario_te

        # Per-feature standardization (mean/std on Train)
        self.feature_means = {}
        self.feature_stds = {}
        self.Phi_va_by_f = {}
        self.Phi_te_by_f = {}
        Phi_tr_by_f = {}
        for j, fname in enumerate(FEATURE_NAMES):
            col_tr = X_tr[:, [j]].astype(np.float32)
            col_va = X_va[:, [j]].astype(np.float32)
            col_te = X_te[:, [j]].astype(np.float32)
            mean = float(col_tr.mean())
            std = float(col_tr.std()) + 1e-8
            self.feature_means[fname] = mean
            self.feature_stds[fname] = std
            col_tr_s = (col_tr - mean) / std
            col_va_s = (col_va - mean) / std
            col_te_s = (col_te - mean) / std
            Phi_tr_by_f[fname] = torch.from_numpy(col_tr_s).to(device)
            self.Phi_va_by_f[fname] = torch.from_numpy(col_va_s)  # CPU
            self.Phi_te_by_f[fname] = torch.from_numpy(col_te_s)  # CPU

        # MLP input dim = sum_j Phi[f].shape[1] = 6 (one column per feature)
        self.basis_sizes = [Phi_tr_by_f[f].shape[1] for f in FEATURE_NAMES]
        self.total_dim = sum(self.basis_sizes)

        # y_va on device
        self.y_va_t = torch.from_numpy(y_va).float().squeeze(1).to(device)

        # train_attack_sets
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
            neg_w = max(neg_w, 1.0)
            self.train_attack_sets.append({
                "attack": atk, "Phi": Phi_attack, "y": y_attack,
                "neg_weight": neg_w,
            })
        del Phi_tr_by_f


def build_data_context(device, verbose=True):
    if verbose:
        print("\n[Data] Loading from cache ...")
    dc.ensure_cache(verbose=verbose)
    X_tr, y_tr, attack_tr, _, _ = dc.load_split("Train")
    X_va, y_va, attack_va, _, _ = dc.load_split("Validation")
    X_te, y_te, attack_te, density_te, scenario_te = dc.load_split("Test")
    if verbose:
        print(f"  Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}  Test: {X_te.shape[0]}")
        print("[Data] Standardizing features (once) ...")
    ctx = DataContext(device, X_tr, y_tr, attack_tr,
                       X_va, y_va, attack_va, X_te, y_te, attack_te,
                       density_te, scenario_te)
    if verbose:
        print(f"  MLP input dim: {ctx.total_dim}")
        print(f"  Train attack sets: {len(ctx.train_attack_sets)}")
    return ctx


# ============================================================================
# Training (inline, per trial)
# ============================================================================

def train_single_trial(ctx, hp, num_epochs, val_check_every=25, patience=15,
                       composite_delta_u_weight=0.3, verbose=False):
    """
    Trains an MLP with hp and returns (best_state, val_metrics).

    hp: {DELTA_U_WEIGHT_MAX, DELTA_U_MARGIN,
         HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT, LR_MLP, WEIGHT_DECAY}

    val_metrics: dict with macro_f1, macro_aurc, macro_misclass_auroc, macro_delta_u
    """
    device = ctx.device
    torch.manual_seed(42)

    hidden_dims = (int(hp["HIDDEN_DIM_1"]), int(hp["HIDDEN_DIM_2"]))
    mlp_head = EvidentialMLPHead(
        input_dim=ctx.total_dim,
        hidden_dims=hidden_dims,
        dropout=float(hp["DROPOUT"]),
    ).to(device)

    opt = torch.optim.Adam(
        mlp_head.parameters(),
        lr=float(hp["LR_MLP"]),
        weight_decay=float(hp.get("WEIGHT_DECAY", 1e-5)),
    )

    delta_u_weight_max = float(hp["DELTA_U_WEIGHT_MAX"])
    delta_u_margin = float(hp["DELTA_U_MARGIN"])
    belief_weight = float(hp.get("BELIEF_WEIGHT", 0.0))
    tracking_start = DELTA_U_START_EPOCH + DELTA_U_ANNEAL_STEPS + 50

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        mlp_head.train()
        opt.zero_grad()
        kl_weight = min(1.0, epoch / KL_ANNEAL_STEPS)
        if epoch >= DELTA_U_START_EPOCH:
            delta_u_weight = min(
                delta_u_weight_max,
                delta_u_weight_max * (epoch - DELTA_U_START_EPOCH) / DELTA_U_ANNEAL_STEPS
            )
        else:
            delta_u_weight = 0.0

        attack_losses = []
        for ds in ctx.train_attack_sets:
            p_tr, (b_tr, d_tr, u_tr), alpha_tr = forward_prob_multi(
                ds["Phi"], mlp_head, FEATURE_NAMES
            )
            l_evid = evidential_loss(alpha_tr, ds["y"],
                                     neg_weight=ds["neg_weight"],
                                     kl_weight=kl_weight)
            if delta_u_weight > 0.0:
                l_du = delta_u_pairwise_hinge_loss(
                    u_tr, p_tr, ds["y"],
                    margin=delta_u_margin,
                    max_pairs_per_class=DELTA_U_MAX_PAIRS_PER_CLASS,
                )
            else:
                l_du = torch.tensor(0.0, device=device)
            if belief_weight > 0.0:
                l_belief = belief_correctness_loss(b_tr, d_tr, ds["y"])
            else:
                l_belief = torch.tensor(0.0, device=device)
            loss_a = l_evid + delta_u_weight * l_du + belief_weight * l_belief
            attack_losses.append(loss_a)

        loss = torch.stack(attack_losses).mean()
        loss.backward()
        opt.step()

        if epoch % val_check_every == 0:
            mlp_head.eval()
            with torch.no_grad():
                p_va_cpu, fused_va_cpu, alpha_va_cpu = forward_prob_multi_chunked(
                    ctx.Phi_va_by_f, mlp_head, FEATURE_NAMES,
                    device=device, chunk_size=200_000,
                )
                u_va_t = fused_va_cpu[2].to(device)
                p_va_t = p_va_cpu.to(device)
                alpha_va_t = alpha_va_cpu.to(device)
                l_evid_va = evidential_loss(alpha_va_t, ctx.y_va_t,
                                            neg_weight=1.0, kl_weight=kl_weight)
                val_base = l_evid_va.item()
                y_pred_va = (p_va_t >= DECISION_THR_FIXED).float()
                correct = (y_pred_va == ctx.y_va_t)
                if correct.any() and (~correct).any():
                    delta_u_val = (u_va_t[~correct].mean() - u_va_t[correct].mean()).item()
                else:
                    delta_u_val = float("nan")
                if not np.isnan(delta_u_val):
                    val_loss = val_base - composite_delta_u_weight * delta_u_val
                else:
                    val_loss = val_base
                del u_va_t, p_va_t, alpha_va_t
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if epoch < tracking_start:
                pass
            elif val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().clone().cpu()
                    for k, v in mlp_head.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch}.")
                    break

    if best_state is None:
        best_state = {
            k: v.detach().clone().cpu()
            for k, v in mlp_head.state_dict().items()
        }

    # Final Val evaluation with best_state.
    # Rebuild the MLP + load the state dict
    eval_head = EvidentialMLPHead(
        input_dim=ctx.total_dim,
        hidden_dims=hidden_dims,
        dropout=float(hp["DROPOUT"]),
    ).to(device)
    eval_head.load_state_dict(best_state)
    eval_head.eval()
    # attacks_list MUST be set. With attacks_list=None,
    # compute_macro_metrics_per_attack pools all samples into ONE group and
    # returns pooled MICRO values under the "macro_*" keys. The five Pareto
    # objectives were then named macro_* but were actually micro — a
    # DIFFERENT metric than the HP approach's objectives and than what
    # Table 1 later reports as f1_macro (build_paper_outputs_final macro-
    # averages over selected_attacks).
    #
    # This matters a lot given this dataset's prevalence spread: the
    # attacker share ranges from 2.2% (suddenConstantSpeed) to 52%
    # (dosAttack), a factor of ~13. Micro is dominated by the large,
    # balanced groups; macro weights every attack equally. "macro == micro"
    # only holds for equally sized, equally hard groups — not the case here.
    # (B-Spline already has the same fix, see sweep_optuna_v2.py.)
    val_metrics, _ = compute_macro_metrics_per_attack(
        ctx.Phi_va_by_f, eval_head, ctx.y_va,
        attack_for_te=ctx.attack_va, attacks_list=ctx.attacks_va_list,
        device=device,
    )

    # Also store hidden_dims in best_state (needed to reconstruct the model later)
    best_state["_meta_hidden_dims"] = hidden_dims
    best_state["_meta_dropout"] = float(hp["DROPOUT"])
    best_state["_meta_input_dim"] = ctx.total_dim

    return best_state, val_metrics


def compute_macro_metrics_per_attack(Phi_by_f, mlp_head, y_np,
                                      attack_for_te, attacks_list, device):
    """Forward pass on the full Phi, then metrics (per attack, or global)."""
    with torch.no_grad():
        p_t, fused_t, _ = forward_prob_multi_chunked(
            Phi_by_f, mlp_head, FEATURE_NAMES, device=device, chunk_size=200_000,
        )
    p = p_t.numpy()
    b = fused_t[0].numpy()
    d = fused_t[1].numpy()
    u = fused_t[2].numpy()
    y_true = y_np.reshape(-1)
    y_pred = (p >= DECISION_THR_FIXED).astype(int)

    def _belief_correctness(b_arr, d_arr, y_t, y_p):
        correct = (y_p == y_t)
        mask_b = (y_t == 1) & correct
        mask_a = (y_t == 0) & correct
        bcb = float(b_arr[mask_b].mean()) if mask_b.any() else float("nan")
        dca = float(d_arr[mask_a].mean()) if mask_a.any() else float("nan")
        if np.isnan(bcb) or np.isnan(dca):
            return float("nan")
        return (bcb + dca) / 2.0

    if attacks_list is None:
        m = evaluate_all(y_true=y_true, y_prob=p, y_pred=y_pred, uncertainty=u)
        f1 = _f1_attacker(y_true, y_pred)
        correct = (y_pred == y_true)
        u_c = u[correct]
        u_w = u[~correct]
        #delta_u = (np.median(u_w) - np.median(u_c)) if len(u_c) and len(u_w) else float("nan")
        delta_u = (u_w.mean() - u_c.mean()) if len(u_c) and len(u_w) else float("nan")
        delta_u_med = (np.median(u_w) - np.median(u_c)) if len(u_c) and len(u_w) else float("nan")
        bc = _belief_correctness(b, d, y_true, y_pred)
        return {
            "macro_f1":                 float(f1),
            "macro_aurc":               float(m.get("aurc", float("nan"))),
            "macro_misclass_auroc":     float(m.get("misclass_auroc", float("nan"))),
            "macro_delta_u":            float(delta_u),
            "macro_delta_u_median":     float(delta_u_med),
            "macro_delta_u_cohens_d":   float(delta_u_cohens_d(y_true, y_pred, u)),
            "macro_belief_correctness": float(bc),
        }, None

    per_attack = {}
    for atk in attacks_list:
        mask = (attack_for_te == atk)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        m = evaluate_all(y_true=y_true[idx], y_prob=p[idx],
                         y_pred=y_pred[idx], uncertainty=u[idx])
        f1 = _f1_attacker(y_true[idx], y_pred[idx])
        correct = (y_pred[idx] == y_true[idx])
        u_c = u[idx][correct]
        u_w = u[idx][~correct]
        #du = (np.median(u_w) - np.median(u_c)) if len(u_c) and len(u_w) else float("nan")
        du = (u_w.mean() - u_c.mean()) if len(u_c) and len(u_w) else float("nan")
        # Non-optimized comparison values: they do NOT go into the objective,
        # but are logged per trial to show how much the estimator choice
        # moves the result.
        du_med = (np.median(u_w) - np.median(u_c)) if len(u_c) and len(u_w) else float("nan")
        du_d = delta_u_cohens_d(y_true[idx], y_pred[idx], u[idx])
        bc = _belief_correctness(b[idx], d[idx], y_true[idx], y_pred[idx])
        per_attack[atk] = {
            "f1": f1,
            "aurc": m.get("aurc", float("nan")),
            "misclass_auroc": m.get("misclass_auroc", float("nan")),
            "delta_u": du,
            "delta_u_median": du_med,
            "delta_u_cohens_d": du_d,
            "belief_correctness": bc,
        }

    def _mm(key):
        vs = [pa[key] for pa in per_attack.values()
              if not (isinstance(pa.get(key), float) and np.isnan(pa[key]))]
        return float(np.mean(vs)) if vs else float("nan")

    return {
        "macro_f1":                 _mm("f1"),
        "macro_aurc":               _mm("aurc"),
        "macro_misclass_auroc":     _mm("misclass_auroc"),
        # Optuna objective — MEAN at level 1, mean over attacks at level 2.
        "macro_delta_u":            _mm("delta_u"),
        "macro_delta_u_median":     _mm("delta_u_median"),
        "macro_delta_u_cohens_d":   _mm("delta_u_cohens_d"),
        "macro_belief_correctness": _mm("belief_correctness"),
    }, per_attack


def _f1_attacker_macro_named(y_true, y_pred, attack_arr, attacks_list):
    """
    Macro-F1 (attacker = positive class) over the groups in attacks_list.

    Semantics identical to the macro branch of compute_macro_metrics_per_attack:
    the mean over the per-attack F1 of the groups that are PRESENT. Needed
    for the threshold diagnostics in run_final_test, where no Phi forward
    pass runs — p is already available.
    """
    a = np.asarray(attack_arr).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    f1s = []
    for atk in attacks_list:
        idx = np.where(a == atk)[0]
        if len(idx) == 0:
            continue
        f1s.append(_f1_attacker(y_true[idx], y_pred[idx]))
    return float(np.mean(f1s)) if f1s else float("nan")


def _f1_attacker(y_true, y_pred):
    tp = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp) / denom if denom > 0 else 0.0


# ============================================================================
# Trial-state storage
# ============================================================================

def save_trial_states(states):
    Path(os.path.dirname(TRIAL_STATES_FILE)).mkdir(parents=True, exist_ok=True)
    torch.save(states, TRIAL_STATES_FILE)


def load_trial_states():
    if not os.path.exists(TRIAL_STATES_FILE):
        return {}
    return torch.load(TRIAL_STATES_FILE, map_location="cpu")


# ============================================================================
# Knee-point logic (identical in all multi-objective sweeps)
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


def select_balanced_trial(study, verbose=True):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if not completed:
        raise RuntimeError("No completed trials — nothing to select.")
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
            print(f"\n[{name}]  → Trial {tnum}")
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
              f"({top_count}/{len(knee_results)} votes = "
              f"{100*info['consensus_strength']:.0f}%)")
        if info["tie_breaker_used"]:
            print(f"  Tiebreaker: {info['tie_breaker_method']}")
        print(f"{'='*70}")
    return winner, info


# ============================================================================
# Paper outputs for the final test run
# ============================================================================

def _render_extended_boxplot(boxplot_samples, out_path, title_suffix=""):
    """
    Renders a boxplot with 12 categories, grouped by SL component:
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
        ("u_benign", "u — benign"),
        ("u_malicious", "u — malicious"),
        ("u_correct", "u — correct"),
        ("u_wrong",   "u — misclassified"),
        ("b_benign", "b — benign"),
        ("b_malicious", "b — malicious"),
        ("b_correct", "b — correct"),
        ("b_wrong",   "b — misclassified"),
        ("d_benign", "d — benign"),
        ("d_malicious", "d — malicious"),
        ("d_correct", "d — correct"),
        ("d_wrong",   "d — misclassified"),
    ]
    BOX_SPACING = 1.0
    GROUP_GAP = 1.2
    positions = []; data = []; box_colors = []; labels = []
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
    pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def build_paper_outputs_final(p_all, u_all, b_all, d_all,
                              y_pred_all, y_true_all,
                              attack_all, method_name, out_dir,
                              boxplot_n=50_000):
    """
    Same as in sweep_optuna_v2.py: Tables 1-5 + boxplot + per-sample CSV.
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
            m = evaluate_all(y_true=y_t, y_prob=p_t, y_pred=yp_t, uncertainty=u_t)

            mask_benign    = (y_t == 1)
            mask_malicious = (y_t == 0)

            u_c = u_t[correct_t]; u_w = u_t[~correct_t]
            b_c = b_t[correct_t]; b_w = b_t[~correct_t]
            d_c = d_t[correct_t]; d_w = d_t[~correct_t]
            u_bn = u_t[mask_benign];    u_ml = u_t[mask_malicious]
            b_bn = b_t[mask_benign];    b_ml = b_t[mask_malicious]
            d_bn = d_t[mask_benign];    d_ml = d_t[mask_malicious]

            # Belief correctness for this attack
            mask_bc = (y_t == 1) & correct_t
            mask_ac = (y_t == 0) & correct_t
            bcb = float(b_t[mask_bc].mean()) if mask_bc.any() else float("nan")
            dca = float(d_t[mask_ac].mean()) if mask_ac.any() else float("nan")
            if np.isnan(bcb) or np.isnan(dca):
                belief_corr = float("nan")
            else:
                belief_corr = (bcb + dca) / 2.0

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
        "delta_u_cohens_d": delta_u_cohens_d(y_pool, yp_pool, u_pool),
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
    # Level 1 = MEAN (the objective). The median variant is carried along as
    # a descriptive value — the mean-vs-median spread measures the skew of
    # the wrong-prediction distribution, i.e. the confident-wrong tail.
    # paper_outputs needs it for the delta_u_macro_median column and the
    # __macro__ row of Table 5.
    delta_u_per_attack_median = [
        pa["u_wrong_median"] - pa["u_correct_median"]
        for pa in per_attack_results.values()
        if not np.isnan(pa.get("u_wrong_median", float("nan")))
        and not np.isnan(pa.get("u_correct_median", float("nan")))
    ]
    macro = {
        "macro_f1":                 _macro("f1"),
        "macro_aurc":               _macro("aurc"),
        "macro_misclass_auroc":     _macro("misclass_auroc"),
        "macro_ece":                _macro("ece"),
        "macro_brier":              _macro("brier"),
        "macro_delta_u":            float(np.mean(delta_u_per_attack)) if delta_u_per_attack else float("nan"),
        "macro_delta_u_median":     float(np.mean(delta_u_per_attack_median)) if delta_u_per_attack_median else float("nan"),
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


def run_final_test(ctx, best_state, method_name, out_dir):
    """
    Reconstruct the MLP from best_state's metadata, forward pass on Test,
    then write the paper outputs.
    """
    print(f"\n{'='*70}")
    print(f"FINAL TEST RUN (MLP): {method_name}")
    print(f"{'='*70}")
    device = ctx.device

    # Reconstruct the MLP from the metadata carried in best_state
    hidden_dims = best_state["_meta_hidden_dims"]
    dropout = best_state["_meta_dropout"]
    input_dim = best_state["_meta_input_dim"]
    mlp_head = EvidentialMLPHead(
        input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    # Load only the actual layer weights (without the _meta_ keys)
    state_dict_pure = {k: v for k, v in best_state.items() if not k.startswith("_meta_")}
    mlp_head.load_state_dict(state_dict_pure)
    mlp_head.eval()

    # Val forward pass for the threshold diagnostics
    with torch.no_grad():
        p_va_cpu, _, _ = forward_prob_multi_chunked(
            ctx.Phi_va_by_f, mlp_head, FEATURE_NAMES,
            device=device, chunk_size=200_000,
        )
    p_va_np = p_va_cpu.numpy()
    y_va_flat = ctx.y_va.reshape(-1)

    # ------------------------------------------------------------------
    # Operating point = DECISION_THR_FIXED, identical to the sweep. NO refit.
    #
    # The sweep evaluates all five objectives at 0.5 (train_and_eval:
    # y_pred_va = (p_va_t >= DECISION_THR_FIXED)). misclass_auroc, delta_u and
    # aurc are all defined entirely through correct = (y_pred == y_true). A
    # refit shifts the correct/wrong partition, and with it the metrics,
    # away from the selection point — you would be selecting on one quantity
    # and measuring it somewhere else.
    #
    # Verified on the HP approach at 7.0M Val samples:
    #     @0.500 (sweep):  macro_f1 0.3723 | mauroc 0.8398 | delta_u +0.6034
    #     @0.480 (refit):  macro_f1 0.4455 | mauroc 0.4485 | delta_u -0.0086
    #
    # The old fit also ran on MICRO-F1 (_f1_attacker), while Table 1 reports
    # f1_macro — so the threshold used to be optimized against a different
    # quantity than the one reported in the paper.
    # ------------------------------------------------------------------
    t_star = float(DECISION_THR_FIXED)
    val_macro_f1 = _f1_attacker_macro_named(y_va_flat, (p_va_np >= t_star).astype(int),
                                            ctx.attack_va, ctx.attacks_va_list)
    print(f"  Operating point = sweep point {t_star:.3f}  "
          f"(Val macro-F1 there = {val_macro_f1:.4f})")

    # For transparency only: what a refit WOULD have produced — intentionally
    # NOT used.
    ts = np.linspace(0.01, 0.99, 199)
    f1s = [_f1_attacker_macro_named(y_va_flat, (p_va_np >= t).astype(int),
                                    ctx.attack_va, ctx.attacks_va_list) for t in ts]
    t_refit = float(ts[int(np.argmax(f1s))])
    print(f"  [Info] macro-F1 optimum would be t={t_refit:.3f} "
          f"(Val macro-F1={max(f1s):.4f}) -- intentionally NOT used, see comment above.")

    # Test forward pass
    with torch.no_grad():
        p_te_t, fused_te, _ = forward_prob_multi_chunked(
            ctx.Phi_te_by_f, mlp_head, FEATURE_NAMES,
            device=device, chunk_size=200_000,
        )
    p_all = p_te_t.numpy()
    b_all = fused_te[0].numpy()
    d_all = fused_te[1].numpy()
    u_all = fused_te[2].numpy()
    y_true = ctx.y_te.reshape(-1)
    y_pred = (p_all >= t_star).astype(int)

    method_entry = build_paper_outputs_final(
        p_all=p_all, u_all=u_all, b_all=b_all, d_all=d_all,
        y_pred_all=y_pred, y_true_all=y_true,
        attack_all=ctx.attack_te,
        method_name=method_name, out_dir=out_dir,
    )

    # Save the model
    model = {
        "variant": "mlp_raw_with_delta_u",
        "feature_means": ctx.feature_means,
        "feature_stds":  ctx.feature_stds,
        "mlp_state_dict": state_dict_pure,
        "mlp_input_dim": input_dim,
        "mlp_hidden_dims": hidden_dims,
        "mlp_dropout": dropout,
        "threshold": t_star,
        "features": FEATURE_NAMES,
    }
    model_path = os.path.join(out_dir, f"mlp_raw_model_{method_name}.pt")
    torch.save(model, model_path)
    print(f"\n  Model:           {model_path}")
    return method_entry


# ============================================================================
# Optuna study
# ============================================================================

def create_or_load_study(resume=False):
    sampler = optuna.samplers.NSGAIISampler(
        population_size=24,
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
            "Fix: remove mlp_optuna_v2.db, or change STUDY_NAME.")
    return study


def write_summary_csv(study, path):
    pareto_set = {t.number for t in study.best_trials}
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {
            "trial_number": t.number,
            "method_name":  t.user_attrs.get("method_name", f"trial_{t.number:03d}"),
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
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
        ax.scatter(np_x, np_y, c="#888888", s=60, alpha=0.5,
                   label=f"dominated (n={len(np_x)})")
    if p_x:
        sorted_p = sorted(zip(p_x, p_y), key=lambda xy: xy[0])
        sx = [xy[0] for xy in sorted_p]; sy = [xy[1] for xy in sorted_p]
        ax.scatter(sx, sy, c="#D85A30", s=120, alpha=0.95, edgecolors="black",
                   linewidths=1.2, label=f"Pareto (n={len(p_x)})", zorder=3)
        ax.plot(sx, sy, c="#D85A30", alpha=0.4, linestyle="--", zorder=2)
    for t in completed:
        ax.annotate(str(t.number), (t.values[idx_x], t.values[idx_y]),
                    fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"{obj_x}  ({'↑' if dir_x=='maximize' else '↓'})")
    ax.set_ylabel(f"{obj_y}  ({'↑' if dir_y=='maximize' else '↓'})")
    ax.set_title(f"MLP Pareto: {obj_x} vs {obj_y}")
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
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS_PER_TRIAL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for sweep outputs: CSV, Pareto plots, "
                             "balanced_trial.json, trial_states.pt, final_test/ "
                             f"(default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run knee-point selection + final test")
    parser.add_argument("--final-test-only", action="store_true",
                        help="Only run the final test on an existing best state")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    global TRIAL_STATES_FILE
    TRIAL_STATES_FILE = os.path.join(args.output_dir, "trial_states.pt")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sweep MLP V2] Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Sweep MLP V2] Objectives:  {OBJECTIVE_NAMES}")
    print(f"[Sweep MLP V2] Directions:  {OBJECTIVE_DIRECTION}")
    print(f"[Sweep MLP V2] Output dir:  {args.output_dir}/")

    ctx = build_data_context(device, verbose=True)

    study = create_or_load_study(resume=args.resume or args.analyze_only or args.final_test_only)
    trial_states = load_trial_states()

    # ---------- Phase 1: Sweep ----------
    if not (args.analyze_only or args.final_test_only):
        n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        n_remaining = args.n_trials - n_completed
        if n_remaining <= 0:
            print(f"\n[Sweep] Target ({args.n_trials}) already reached ({n_completed} trials).")
        else:
            print(f"\n[Sweep] {n_completed} done, running {n_remaining} more trials")

            def objective(trial):
                hp = {
                    # delta_u loss — widened search range for more headroom
                    "DELTA_U_WEIGHT_MAX": trial.suggest_float(
                        "DELTA_U_WEIGHT_MAX", 2.0, 20.0, log=True),
                    "DELTA_U_MARGIN":     trial.suggest_float(
                        "DELTA_U_MARGIN", 0.20, 0.60, step=0.05),
                    # Belief-correctness loss — range pulled back so that
                    # delta_u gets more room (belief pushes u down globally)
                    "BELIEF_WEIGHT":      trial.suggest_float(
                        "BELIEF_WEIGHT", 0.2, 5.0, log=True),
                    # MLP architecture
                    "HIDDEN_DIM_1": trial.suggest_categorical(
                        "HIDDEN_DIM_1", [32, 64, 128]),
                    "HIDDEN_DIM_2": trial.suggest_categorical(
                        "HIDDEN_DIM_2", [16, 32, 64]),
                    "DROPOUT":      trial.suggest_float(
                        "DROPOUT", 0.0, 0.3, step=0.05),
                    "LR_MLP":       trial.suggest_float(
                        "LR_MLP", 1e-4, 1e-2, log=True),
                    "WEIGHT_DECAY": 1e-5,
                }
                composite_w = trial.suggest_float(
                    "COMPOSITE_DELTA_U_WEIGHT", 0.3, 5.0, log=True)
                t_start = time.time()
                print(f"\n{'─'*60}")
                print(f"Trial {trial.number}:")
                for k, v in hp.items():
                    print(f"  {k:25s} = {v}")
                print(f"  COMPOSITE_DELTA_U_WEIGHT  = {composite_w:.3f}")
                best_state, val_metrics = train_single_trial(
                    ctx, hp, num_epochs=args.num_epochs,
                    composite_delta_u_weight=composite_w,
                )
                dt = time.time() - t_start
                trial_states[trial.number] = best_state
                save_trial_states(trial_states)
                trial.set_user_attr("method_name", f"trial_{trial.number:03d}")
                trial.set_user_attr("duration_min", dt / 60)
                for k, v in val_metrics.items():
                    trial.set_user_attr(k, v)
                print(f"  done in {dt/60:.1f} min:  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                for name in OBJECTIVE_NAMES:
                    if np.isnan(val_metrics.get(name, float("nan"))):
                        print(f"  [WARN] {name} is NaN → TrialPruned")
                        raise optuna.TrialPruned()
                return tuple(val_metrics[n] for n in OBJECTIVE_NAMES)

            t_sweep = time.time()
            study.optimize(objective, n_trials=n_remaining,
                           gc_after_trial=True, show_progress_bar=False)
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
        for j in range(i+1, len(OBJECTIVE_NAMES)):
            plot_pareto_2d(study, OBJECTIVE_NAMES[i], OBJECTIVE_NAMES[j],
                           os.path.join(args.output_dir,
                                        f"pareto_{OBJECTIVE_NAMES[i]}_vs_{OBJECTIVE_NAMES[j]}.png"))

    # ---------- Phase 3: Knee-point selection ----------
    winner, info = select_balanced_trial(study, verbose=True)
    with open(os.path.join(args.output_dir, "balanced_trial.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"  balanced JSON: {os.path.join(args.output_dir, 'balanced_trial.json')}")

    # ---------- Phase 4: Final test with the balanced trial ----------
    trial_states = load_trial_states()
    if winner not in trial_states:
        print(f"\n[ERROR] Best state for Trial {winner} missing in {TRIAL_STATES_FILE}.")
        return
    best_state = trial_states[winner]
    method_name = f"MLP_balanced_trial{winner:03d}"
    final_dir = os.path.join(args.output_dir, "final_test")
    run_final_test(ctx, best_state, method_name=method_name, out_dir=final_dir)
    print(f"\n[Sweep MLP V2] DONE. Final outputs in {final_dir}/")


if __name__ == "__main__":
    main()
