"""
Multi-objective Optuna sweep for the B-spline Subjective-Logic trust model.

Pipeline
--------
1. Load the Train/Validation/Test splits once from the binary cache
   (see ``data_cache.py``) and build the per-feature B-spline basis once.
2. Run an NSGA-II sweep over the loss weights (lambda values). Every trial
   trains the model inline (no subprocess) and is scored on VALIDATION only,
   on five objectives (macro-averaged over attack types):
       macro_f1 (max), macro_aurc (min), macro_misclass_auroc (max),
       macro_delta_u (max), macro_belief_correctness (max)
3. Pick one "balanced" trial from all completed trials with four knee-point
   methods plus majority vote (ties broken by a fixed priority).
4. Evaluate ONLY the balanced trial on TEST: paper tables 1-5, box plots,
   results JSON, optional per-sample CSV.

Usage
-----
    # 1. Build the feature cache (once)
    python data_cache.py --build --data-root /path/to/dataset

    # 2. Run the sweep
    python sweep_optuna.py --n-trials 20

    # 3. Resume / analysis only / final test only
    python sweep_optuna.py --resume --n-trials 30
    python sweep_optuna.py --analyze-only
    python sweep_optuna.py --final-test-only

See README.md for all parameters and search ranges.
"""
import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.preprocessing import SplineTransformer

import data_cache as dc
import paper_outputs
from sl_metrics import evaluate_all


# ============================================================================
# Configuration
# ============================================================================

OUT_DIR_DEFAULT = os.environ.get("MBD_SWEEP_DIR", "runs/bspline_sweep")
STUDY_NAME_DEFAULT = "bspline_pareto_5obj"

# Pareto objectives, all computed on VALIDATION during the sweep.
OBJECTIVE_NAMES = [
    "macro_f1", "macro_aurc", "macro_misclass_auroc",
    "macro_delta_u", "macro_belief_correctness",
]
OBJECTIVE_DIRECTION = ["maximize", "minimize", "maximize", "maximize", "maximize"]

# Optuna search space. All lambda weights start at 0 so the sweep can also
# switch the corresponding loss/selection term off entirely (linear sampling,
# because a log scale cannot include 0).
#   name: (low, high, step)   step=None -> continuous
SEARCH_SPACE = {
    "DELTA_U_WEIGHT_MAX":       (0.0, 20.0, None),   # lambda_du   (delta-u hinge loss)
    "DELTA_U_MARGIN":           (0.20, 0.60, 0.05),  # m           (hinge margin)
    "BELIEF_WEIGHT":            (0.0, 5.0, None),    # lambda_bel  (belief-correctness loss)
    "COMPOSITE_DELTA_U_WEIGHT": (0.0, 5.0, None),    # lambda_sel  (checkpoint selection)
}

# timeDelayAttack and trafficCongestionSybil are deliberately excluded: they
# cannot be detected by local MDSs.
SELECTED_ATTACKS = [
    "constantPositionOffset", "randomPositionOffset", "positionMirroring",
    "suddenStop", "accelerationMultiplication", "feignedBraking",
    "constantSpeedOffset", "randomSpeedOffset", "suddenConstantSpeed",
    "zeroSpeedReport", "reversedHeading", "dataReplay", "dosAttack",
]
FEATURE_NAMES = dc.FEATURE_NAMES

# Subjective Logic
U_MIN = 1e-3            # lower clamp for the uncertainty mass u
BASE_RATE = 0.5         # prior a in p = b + a * u
DECISION_THRESHOLD = 0.5  # p >= 0.5  <=>  b >= d  (canonical SL decision)

# B-spline basis
# Feature values are implausibilities in [0, 1] (x = 1 - plausibility, see
# data_cache.py). NA_SENTINEL (-1) means "detector not applicable".
#   * Knots are fitted on valid values (x >= 0) only, so the -1 mass does not
#     distort the knot placement.
#   * N/A samples get an all-zero basis for the affected feature, i.e. that
#     feature contributes NO evidence (Phi @ w = 0 in both logits). This is the
#     vacuous opinion (u = 1): more N/A features -> less evidence -> higher u.
NA_SENTINEL = dc.NA_SENTINEL
SPLINE_N_KNOTS = 16
SPLINE_DEGREE = 3
# Checks are heavily concentrated at 0 (most checks say "plausible"). Quantile
# knots put resolution where the data is and preserve the spread of u;
# uniform knots would waste resolution and flatten u (smaller delta_u).
SPLINE_KNOTS_STRATEGY = "quantile"

# Fixed training hyperparameters (not part of the search space)
SEED = 42
LEARNING_RATE = 0.03
NUM_EPOCHS_PER_TRIAL = 3000
VAL_CHECK_EVERY = 25          # epochs between validation checks
PATIENCE = 60                 # early stopping, counted in validation checks
KL_ANNEAL_STEPS = 200
DELTA_U_START_EPOCH = 0
DELTA_U_ANNEAL_STEPS = 250
DELTA_U_MAX_PAIRS_PER_CLASS = 1500
LAM_SMOOTH = 1e-3             # second-order smoothness penalty on spline weights
EVAL_CHUNK_SIZE = 200_000

# NSGA-II
NSGA_POPULATION_SIZE = 10
NSGA_CROSSOVER_PROB = 0.9
NSGA_SWAPPING_PROB = 0.5


# ============================================================================
# B-spline basis
# ============================================================================

def fit_feature_spline(col_tr):
    """
    Fit a SplineTransformer on the VALID (x >= 0) values of one feature column.

    Prefers SPLINE_KNOTS_STRATEGY ("quantile"). Falls back to uniform knots if
    a check has too little variation (quantile knots would degenerate), and
    finally to a [0, 1] grid if the feature is entirely N/A or constant in
    Train, so the basis dimension stays identical across features.
    """
    col_tr = np.asarray(col_tr).reshape(-1)
    valid = col_tr[col_tr >= 0.0].reshape(-1, 1)
    n_unique = int(np.unique(valid).size)

    if n_unique >= SPLINE_N_KNOTS:
        try:
            sp = SplineTransformer(n_knots=SPLINE_N_KNOTS, degree=SPLINE_DEGREE,
                                   knots=SPLINE_KNOTS_STRATEGY, include_bias=True)
            sp.fit(valid)
            return sp
        except ValueError:
            pass  # degenerate quantile knots -> uniform fallback below

    sp = SplineTransformer(n_knots=SPLINE_N_KNOTS, degree=SPLINE_DEGREE,
                           knots="uniform", include_bias=True)
    if n_unique >= 2:
        sp.fit(valid)
    else:
        sp.fit(np.linspace(0.0, 1.0, max(2 * SPLINE_N_KNOTS, 64)).reshape(-1, 1))
    return sp


def transform_with_na(sp, col):
    """Spline-transform a feature column; N/A rows (x < 0) get an all-zero basis."""
    col = np.asarray(col).reshape(-1)
    P = sp.transform(col.reshape(-1, 1)).astype(np.float32)
    P[col < 0.0] = 0.0
    return P


# ============================================================================
# Model (standard Subjective Logic, binary frame {benign, attacker})
# ============================================================================

def opinion_joint(Phi, w_b, w_u):
    """Map the spline basis to a binomial opinion (b, d, u)."""
    e_benign = torch.nn.functional.softplus(Phi @ w_b)
    e_attacker = torch.nn.functional.softplus(Phi @ w_u)
    S = (e_benign + 1.0) + (e_attacker + 1.0)   # Dirichlet strength alpha_b + alpha_a
    b = e_benign / S
    d = e_attacker / S
    u = torch.clamp(2.0 / S, min=U_MIN)
    return b, d, u


def forward_prob_concat(Phi, w_b, w_u, base_rate=BASE_RATE):
    """Returns p(benign), (b, d, u) and the Dirichlet parameters alpha (N, 2)."""
    b, d, u = opinion_joint(Phi, w_b, w_u)
    S = 2.0 / u
    alpha = torch.stack([b * S + 1.0, d * S + 1.0], dim=1)
    p = torch.clamp(b + base_rate * u, 1e-6, 1 - 1e-6)
    return p, (b, d, u), alpha


def forward_prob_multi(Phi_by_f, w_b, w_u, base_rate=BASE_RATE):
    Phi = torch.cat([Phi_by_f[f] for f in FEATURE_NAMES], dim=1)
    return forward_prob_concat(Phi, w_b, w_u, base_rate)


def forward_prob_chunked(Phi_by_f_cpu, w_b, w_u, device,
                         base_rate=BASE_RATE, chunk_size=EVAL_CHUNK_SIZE):
    """Chunked no-grad forward pass over a large CPU-resident split."""
    n = next(iter(Phi_by_f_cpu.values())).shape[0]
    p_out = torch.empty(n, dtype=torch.float32)
    b_out = torch.empty(n, dtype=torch.float32)
    d_out = torch.empty(n, dtype=torch.float32)
    u_out = torch.empty(n, dtype=torch.float32)
    alpha_out = torch.empty(n, 2, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            Phi_chunk = {f: Phi_by_f_cpu[f][start:end].to(device, non_blocking=True)
                         for f in FEATURE_NAMES}
            p, (b, d, u), alpha = forward_prob_multi(Phi_chunk, w_b, w_u, base_rate)
            p_out[start:end] = p.cpu()
            b_out[start:end] = b.cpu()
            d_out[start:end] = d.cpu()
            u_out[start:end] = u.cpu()
            alpha_out[start:end] = alpha.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p_out, (b_out, d_out, u_out), alpha_out


# ============================================================================
# Losses
# ============================================================================

def smoothness_penalty(w, order=2):
    d = w
    for _ in range(order):
        d = d[1:] - d[:-1]
    return (d ** 2).sum()


def evidential_mse_loss(alpha, y_onehot, neg_weight=1.0):
    S = alpha.sum(dim=1, keepdim=True)
    p_hat = alpha / S
    err = (y_onehot - p_hat) ** 2
    var = alpha * (S - alpha) / (S * S * (S + 1.0))
    weights = torch.tensor([1.0, neg_weight], device=alpha.device).view(1, 2)
    return (weights * (err + var)).sum(dim=1).mean()


def evidential_kl_loss(alpha, y_onehot):
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    K = alpha_tilde.shape[1]
    sum_alpha = alpha_tilde.sum(dim=1, keepdim=True)
    log_term = (
        torch.lgamma(sum_alpha).squeeze(1)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=1)
    )
    digamma_term = (
        (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha))
    ).sum(dim=1)
    return (log_term + digamma_term).mean()


def evidential_loss(alpha, y, neg_weight=1.0, kl_weight=0.0):
    y_b = y.unsqueeze(1)
    y_onehot = torch.cat([y_b, 1.0 - y_b], dim=1)
    loss = evidential_mse_loss(alpha, y_onehot, neg_weight=neg_weight)
    if kl_weight > 0.0:
        loss = loss + kl_weight * evidential_kl_loss(alpha, y_onehot)
    return loss


def delta_u_pairwise_hinge_loss(u, p, y, margin=0.25,
                                max_pairs_per_class=DELTA_U_MAX_PAIRS_PER_CLASS):
    """Pairwise hinge: every misclassified sample should have u >= u_correct + margin."""
    with torch.no_grad():
        correct_mask = ((p >= DECISION_THRESHOLD).float() == y)
        wrong_mask = ~correct_mask
    n_correct = int(correct_mask.sum().item())
    n_wrong = int(wrong_mask.sum().item())
    if n_correct == 0 or n_wrong == 0:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    u_correct = u[correct_mask]
    u_wrong = u[wrong_mask]
    if n_correct > max_pairs_per_class:
        u_correct = u_correct[torch.randperm(n_correct, device=u.device)[:max_pairs_per_class]]
    if n_wrong > max_pairs_per_class:
        u_wrong = u_wrong[torch.randperm(n_wrong, device=u.device)[:max_pairs_per_class]]
    gap = u_correct.unsqueeze(1) - u_wrong.unsqueeze(0) + margin
    return torch.clamp(gap, min=0.0).mean()


def belief_correctness_loss(b, d, y):
    """
    Class-balanced belief loss: 0.5 * (mean((1-b)^2 | y=1) + mean((1-d)^2 | y=0)).

    Averaging per class first gives the b channel (benign) and the d channel
    (attacker) the same gradient share regardless of class imbalance within
    an attack set.
    """
    benign_mask = y > 0.5
    attacker_mask = ~benign_mask
    zero = torch.tensor(0.0, device=b.device, dtype=b.dtype)
    loss_b = ((1.0 - b[benign_mask]) ** 2).mean() if benign_mask.any() else zero
    loss_d = ((1.0 - d[attacker_mask]) ** 2).mean() if attacker_mask.any() else zero
    return 0.5 * (loss_b + loss_d)


# ============================================================================
# Data setup (once per run)
# ============================================================================

class DataContext:
    """
    Everything that is reused across trials:
      - per-attack train basis Phi, resident on the device
      - validation/test basis on the CPU (evaluated in chunks)
      - fitted splines, feature offsets, test metadata
    """

    def __init__(self, device, X_tr, y_tr, attack_tr,
                 X_va, y_va, attack_va, X_te, y_te, attack_te):
        self.device = device
        self.y_va = y_va
        self.attack_va = attack_va
        self.y_te = y_te
        self.attack_te = attack_te

        # Fit splines on Train, transform all three splits
        self.splines = {}
        self.Phi_va_by_f = {}
        self.Phi_te_by_f = {}
        Phi_tr_cols = {}
        basis_sizes = []
        for j, fname in enumerate(FEATURE_NAMES):
            sp = fit_feature_spline(X_tr[:, j])
            self.splines[fname] = sp
            Phi_tr_cols[fname] = transform_with_na(sp, X_tr[:, j])
            self.Phi_va_by_f[fname] = torch.from_numpy(transform_with_na(sp, X_va[:, j]))
            self.Phi_te_by_f[fname] = torch.from_numpy(transform_with_na(sp, X_te[:, j]))
            basis_sizes.append(Phi_tr_cols[fname].shape[1])

        self.total_dim = sum(basis_sizes)
        self.feature_offsets, cursor = [], 0
        for sz in basis_sizes:
            self.feature_offsets.append((cursor, cursor + sz))
            cursor += sz

        self.y_va_t = torch.from_numpy(y_va).float().squeeze(1).to(device)

        # Per attack: select rows, concatenate features, keep resident on device
        self.train_attack_sets = []
        for attack in SELECTED_ATTACKS:
            idx = np.where(attack_tr == attack)[0]
            if len(idx) == 0:
                continue
            Phi_cat = np.concatenate([Phi_tr_cols[f][idx] for f in FEATURE_NAMES], axis=1)
            y_attack = torch.from_numpy(y_tr[idx]).float().squeeze(1).to(device)
            n_attacker = int((y_attack == 0).sum().item())
            n_benign = int((y_attack == 1).sum().item())
            self.train_attack_sets.append({
                "attack": attack,
                "Phi": torch.from_numpy(Phi_cat).to(device),
                "y": y_attack,
                "neg_weight": (n_benign / n_attacker) if n_attacker > 0 else 1.0,
            })


def build_data_context(device, cache_dir, data_root, num_workers):
    print("\n[Data] Loading splits from cache ...")
    dc.ensure_cache(data_root=data_root, cache_dir=cache_dir,
                    num_workers=num_workers, verbose=True)
    X_tr, y_tr, attack_tr, _, _ = dc.load_split("Train", cache_dir)
    X_va, y_va, attack_va, _, _ = dc.load_split("Validation", cache_dir)
    X_te, y_te, attack_te, _, _ = dc.load_split("Test", cache_dir)
    print(f"  Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}  Test: {X_te.shape[0]}")
    print("[Data] Building spline basis ...")
    ctx = DataContext(device, X_tr, y_tr, attack_tr,
                      X_va, y_va, attack_va, X_te, y_te, attack_te)
    print(f"  Total basis dimension: {ctx.total_dim}")
    print(f"  Train attack sets:     {len(ctx.train_attack_sets)}")
    return ctx


# ============================================================================
# Training (inline, one call per trial)
# ============================================================================

def _smoothness_total(w_b, w_u, feature_offsets):
    return sum(smoothness_penalty(w_b[s:e]) + smoothness_penalty(w_u[s:e])
               for (s, e) in feature_offsets)


def train_single_trial(ctx, hp, num_epochs=NUM_EPOCHS_PER_TRIAL,
                       val_check_every=VAL_CHECK_EVERY, patience=PATIENCE,
                       verbose=False):
    """
    Train one model with the trial hyperparameters ``hp`` (keys: see
    SEARCH_SPACE) and return (best_state, val_metrics).

    Checkpoint selection uses a composite validation score
        val_loss = evidential_loss + LAM_SMOOTH * smoothness
                   - COMPOSITE_DELTA_U_WEIGHT * delta_u_val
    tracked only after the delta-u warm-up has finished.
    """
    device = ctx.device
    torch.manual_seed(SEED)

    w_b = nn.Parameter(torch.zeros(ctx.total_dim, device=device))
    w_u = nn.Parameter(torch.zeros(ctx.total_dim, device=device))
    opt = torch.optim.Adam([w_b, w_u], lr=LEARNING_RATE)

    delta_u_weight_max = float(hp["DELTA_U_WEIGHT_MAX"])
    delta_u_margin = float(hp["DELTA_U_MARGIN"])
    belief_weight = float(hp["BELIEF_WEIGHT"])
    composite_weight = float(hp["COMPOSITE_DELTA_U_WEIGHT"])
    tracking_start = DELTA_U_START_EPOCH + DELTA_U_ANNEAL_STEPS + 50
    inv_n_sets = 1.0 / len(ctx.train_attack_sets)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        kl_weight = min(1.0, epoch / KL_ANNEAL_STEPS)
        if epoch >= DELTA_U_START_EPOCH:
            ramp = (epoch - DELTA_U_START_EPOCH) / DELTA_U_ANNEAL_STEPS
            delta_u_weight = delta_u_weight_max * min(1.0, ramp)
        else:
            delta_u_weight = 0.0

        # Accumulate smoothness + all per-attack losses into one scalar and
        # call backward once (gradient of the sum == sum of the gradients).
        opt.zero_grad()
        total_loss = LAM_SMOOTH * _smoothness_total(w_b, w_u, ctx.feature_offsets)
        for ds in ctx.train_attack_sets:
            p_tr, (b_tr, d_tr, u_tr), alpha_tr = forward_prob_concat(ds["Phi"], w_b, w_u)
            loss_a = evidential_loss(alpha_tr, ds["y"], neg_weight=ds["neg_weight"],
                                     kl_weight=kl_weight)
            if delta_u_weight > 0.0:
                loss_a = loss_a + delta_u_weight * delta_u_pairwise_hinge_loss(
                    u_tr, p_tr, ds["y"], margin=delta_u_margin)
            if belief_weight > 0.0:
                loss_a = loss_a + belief_weight * belief_correctness_loss(b_tr, d_tr, ds["y"])
            total_loss = total_loss + loss_a * inv_n_sets
        total_loss.backward()
        opt.step()

        if epoch % val_check_every != 0 or epoch < tracking_start:
            continue

        with torch.no_grad():
            p_va, (_, _, u_va), alpha_va = forward_prob_chunked(
                ctx.Phi_va_by_f, w_b, w_u, device=device)
            p_va, u_va, alpha_va = p_va.to(device), u_va.to(device), alpha_va.to(device)
            val_loss = (evidential_loss(alpha_va, ctx.y_va_t, kl_weight=kl_weight)
                        + LAM_SMOOTH * _smoothness_total(w_b, w_u, ctx.feature_offsets)).item()
            correct = ((p_va >= DECISION_THRESHOLD).float() == ctx.y_va_t)
            if correct.any() and (~correct).any():
                delta_u_val = (u_va[~correct].mean() - u_va[correct].mean()).item()
                val_loss -= composite_weight * delta_u_val
            del p_va, u_va, alpha_va

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {"w_b_joint": w_b.detach().cpu().clone(),
                          "w_u_joint": w_u.detach().cpu().clone()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}.")
                break

    if best_state is None:  # no checkpoint was tracked -> use the last weights
        best_state = {"w_b_joint": w_b.detach().cpu().clone(),
                      "w_u_joint": w_u.detach().cpu().clone()}

    val_metrics, _ = compute_macro_metrics(
        ctx.Phi_va_by_f, best_state["w_b_joint"].to(device),
        best_state["w_u_joint"].to(device), ctx.y_va, ctx.attack_va, device)
    return best_state, val_metrics


# ============================================================================
# Metrics
# ============================================================================

def f1_attacker(y_true, y_pred):
    """F1 with the attacker class (label 0) as positive class."""
    tp = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 1) & (y_pred == 0)))
    fn = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = 2 * tp + fp + fn
    return (2.0 * tp) / denom if denom > 0 else 0.0


def belief_correctness(b, d, y_true, y_pred):
    """
    (mean(b | benign, correct) + mean(d | attacker, correct)) / 2.
    Returns (score, b_part, d_part); NaN if one of the two groups is empty.
    """
    correct = (y_pred == y_true)
    mask_b = (y_true == 1) & correct
    mask_d = (y_true == 0) & correct
    bcb = float(b[mask_b].mean()) if mask_b.any() else float("nan")
    dca = float(d[mask_d].mean()) if mask_d.any() else float("nan")
    score = float("nan") if np.isnan(bcb) or np.isnan(dca) else (bcb + dca) / 2.0
    return score, bcb, dca


def _nanmean(values):
    vs = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vs)) if vs else float("nan")


def compute_macro_metrics(Phi_by_f, w_b, w_u, y_np, attack_labels, device):
    """
    Forward pass over a split, then F1 / AURC / misclassification AUROC /
    delta_u / belief correctness per attack, macro-averaged over attacks.
    Decision threshold: DECISION_THRESHOLD (fixed, not fitted).
    """
    p_t, (b_t, d_t, u_t), _ = forward_prob_chunked(Phi_by_f, w_b, w_u, device=device)
    p, b, d, u = p_t.numpy(), b_t.numpy(), d_t.numpy(), u_t.numpy()
    y_true = y_np.reshape(-1)
    y_pred = (p >= DECISION_THRESHOLD).astype(int)

    per_attack = {}
    for atk in SELECTED_ATTACKS:
        idx = np.where(attack_labels == atk)[0]
        if len(idx) == 0:
            continue
        yt, yp, ua = y_true[idx], y_pred[idx], u[idx]
        m = evaluate_all(y_true=yt, y_prob=p[idx], y_pred=yp, uncertainty=ua)
        correct = (yp == yt)
        du = (float(ua[~correct].mean() - ua[correct].mean())
              if correct.any() and (~correct).any() else float("nan"))
        per_attack[atk] = {
            "f1": f1_attacker(yt, yp),
            "aurc": m.get("aurc", float("nan")),
            "misclass_auroc": m.get("misclass_auroc", float("nan")),
            "delta_u": du,
            "belief_correctness": belief_correctness(b[idx], d[idx], yt, yp)[0],
        }

    out = {
        f"macro_{k}": _nanmean([pa[k] for pa in per_attack.values()])
        for k in ("f1", "aurc", "misclass_auroc", "delta_u", "belief_correctness")
    }
    return out, per_attack


# ============================================================================
# Per-trial state storage (best weights of every trial, for the final test)
# ============================================================================

def save_trial_states(states, path):
    torch.save(states, path)


def load_trial_states(path):
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location="cpu")


# ============================================================================
# Balanced-trial selection (knee points + consensus)
# ============================================================================

METRICS_FOR_KNEE = list(zip(OBJECTIVE_NAMES, OBJECTIVE_DIRECTION))
TIEBREAKER_PRIORITY = ["chebyshev", "closest_to_utopia",
                       "weighted_sum", "farthest_from_nadir"]


def normalize_trials_for_knee(trials_metrics):
    """Min-max normalise every objective to [0, 1] with 1 = best."""
    Z = np.zeros((len(trials_metrics), len(METRICS_FOR_KNEE)), dtype=np.float64)
    for j, (key, direction) in enumerate(METRICS_FOR_KNEE):
        vals = np.array([t[key] for t in trials_metrics], dtype=float)
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


def select_balanced_trial(study):
    """
    Apply the four knee methods to all completed trials and take the majority
    vote. Returns (winner_trial_number, info_dict).
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if not completed:
        raise RuntimeError("No completed trials in the study - nothing to select.")

    trials_data = []
    for t in completed:
        md = dict(zip(OBJECTIVE_NAMES, t.values))
        md["trial_number"] = t.number
        trials_data.append(md)

    Z = normalize_trials_for_knee(trials_data)
    print(f"\n{'=' * 70}\nKNEE-POINT SELECTION ({len(trials_data)} trials)\n{'=' * 70}")
    knee_results = {}
    for name, fn in KNEE_METHODS.items():
        row = trials_data[fn(Z)]
        knee_results[name] = {
            "trial_number": row["trial_number"],
            "metrics": {k: row[k] for k, _ in METRICS_FOR_KNEE},
        }
        print(f"\n[{name}]  -> trial {row['trial_number']}")
        for k, _ in METRICS_FOR_KNEE:
            print(f"    {k:30s} = {row[k]:.4f}")

    votes = Counter(r["trial_number"] for r in knee_results.values())
    methods_per_trial = defaultdict(list)
    for mname, r in knee_results.items():
        methods_per_trial[r["trial_number"]].append(mname)
    top_count = votes.most_common(1)[0][1]
    top_trials = [tn for tn, c in votes.most_common() if c == top_count]

    info = {
        "votes_per_trial":    dict(votes),
        "methods_per_trial":  dict(methods_per_trial),
        "n_methods_total":    len(knee_results),
        "tie_breaker_used":   len(top_trials) > 1,
        "tie_breaker_method": None,
    }
    winner = top_trials[0]
    if len(top_trials) > 1:
        info["tied_trials"] = top_trials
        for tb in TIEBREAKER_PRIORITY:
            if knee_results[tb]["trial_number"] in top_trials:
                winner = knee_results[tb]["trial_number"]
                info["tie_breaker_method"] = tb
                break
    info["winner_trial"] = winner
    info["n_votes"] = top_count
    info["consensus_strength"] = top_count / len(knee_results)
    info["knee_results"] = knee_results

    print(f"\n{'=' * 70}")
    print(f"CONSENSUS WINNER: trial {winner}  ({top_count}/{len(knee_results)} votes "
          f"= {100 * info['consensus_strength']:.0f}%)")
    if info["tie_breaker_used"]:
        print(f"  Tie-breaker: {info['tie_breaker_method']}")
    print("=" * 70)
    return winner, info


# ============================================================================
# Final test run (balanced trial only)
# ============================================================================

def _extended_delta_metrics(u, b, d, y_pred, y_true, atk_labels, benign_label=1):
    """
    Extended separation metrics (macro over attacks and micro over all samples):
    delta_u (mean difference and Cohen's d), fraction of confidently-wrong
    samples, and belief/disbelief gaps between the two classes.
    """
    u = np.asarray(u, float)
    b = np.asarray(b, float)
    d = np.asarray(d, float)
    y_true = np.asarray(y_true)
    atk_labels = np.asarray(atk_labels)
    correct = (np.asarray(y_pred) == y_true)
    ok = np.isfinite(u) & np.isfinite(b) & np.isfinite(d)
    u, b, d, y_true, correct, atk_labels = (u[ok], b[ok], d[ok],
                                            y_true[ok], correct[ok], atk_labels[ok])
    BEN, ATK = benign_label, 1 - benign_label

    def _delta_mean(uu, cc):
        uw, uc = uu[~cc], uu[cc]
        return np.nan if uw.size == 0 or uc.size == 0 else uw.mean() - uc.mean()

    def _cohens_d(uu, cc):
        uw, uc = uu[~cc], uu[cc]
        n_w, n_c = uw.size, uc.size
        if n_w < 2 or n_c < 2:
            return np.nan
        sp = ((n_w - 1) * uw.var(ddof=1) + (n_c - 1) * uc.var(ddof=1)) / (n_w + n_c - 2)
        return np.nan if sp <= 0 else (uw.mean() - uc.mean()) / np.sqrt(sp)

    def _confident_wrong_frac(uu, cc):
        uw, uc = uu[~cc], uu[cc]
        return np.nan if uw.size == 0 or uc.size == 0 else float((uw < np.median(uc)).mean())

    def _delta_belief(mass, cc, tt, pos):
        hi = mass[cc & (tt == pos)]
        lo = mass[cc & (tt != pos)]
        return np.nan if hi.size == 0 or lo.size == 0 else hi.mean() - lo.mean()

    def _macro(fn, *arrays):
        vals = []
        for atk in SELECTED_ATTACKS:
            m = (atk_labels == atk)
            if m.any():
                vals.append(fn(*[a[m] for a in arrays]))
        return _nanmean([float(v) for v in vals])

    return {
        "delta_u_macro_mean":         _macro(_delta_mean, u, correct),
        "delta_u_micro_mean":         _delta_mean(u, correct),
        "delta_u_macro_cohens_d":     _macro(_cohens_d, u, correct),
        "delta_u_micro_cohens_d":     _cohens_d(u, correct),
        "confident_wrong_frac_macro": _macro(_confident_wrong_frac, u, correct),
        "confident_wrong_frac_micro": _confident_wrong_frac(u, correct),
        "delta_b_macro": _macro(lambda m, cc, tt: _delta_belief(m, cc, tt, BEN), b, correct, y_true),
        "delta_b_micro": _delta_belief(b, correct, y_true, BEN),
        "delta_d_macro": _macro(lambda m, cc, tt: _delta_belief(m, cc, tt, ATK), d, correct, y_true),
        "delta_d_micro": _delta_belief(d, correct, y_true, ATK),
    }


def _render_extended_boxplot(boxplot_samples, out_path, title_suffix=""):
    """
    Box plot with 12 categories grouped by SL component:
        u: benign | malicious | correct | misclassified
        b: benign | malicious | correct | misclassified
        d: benign | malicious | correct | misclassified
    """
    colors = {
        "u_benign": "#5DA0CB", "u_malicious": "#1C4E80",
        "b_benign": "#A07ABF", "b_malicious": "#5E3A8F",
        "d_benign": "#D4A24C", "d_malicious": "#7A5400",
    }
    positions, data, box_colors, labels, separators = [], [], [], [], []
    cursor = 1.0
    for comp in ("u", "b", "d"):
        if comp != "u":
            separators.append(cursor - 1.1)  # midway between the u/b/d groups
        for split, lbl in (("benign", "benign"), ("malicious", "malicious"),
                           ("correct", "correct"), ("wrong", "misclassified")):
            key = f"{comp}_{split}"
            arr = np.asarray(boxplot_samples.get(key, []), dtype=np.float64)
            if arr.size == 0 or np.isnan(arr).all():
                cursor += 1.0
                continue
            data.append(arr)
            positions.append(cursor)
            box_colors.append(colors.get(key, "#3CB371" if split == "correct" else "#A5292A"))
            labels.append(f"{comp} — {lbl}")
            cursor += 1.0
        cursor += 1.2  # gap between the u/b/d groups
    if not data:
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black"), capprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("SL opinion value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Extended SL opinion distribution {title_suffix}".strip())
    for sep_x in separators:
        ax.axvline(sep_x, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.savefig(out_path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _mean_median_std(a):
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(a.mean()), float(np.median(a)), float(a.std())


def _risk_at_coverage(u, correct, coverage):
    """Error rate on the `coverage` fraction of samples with the lowest u."""
    if len(u) == 0:
        return float("nan")
    k = max(1, int(coverage * len(u)))
    order = np.argsort(u)
    return float((~correct[order][:k]).mean())


def build_paper_outputs_final(p_all, u_all, b_all, d_all, y_pred_all, y_true_all,
                              attack_all, method_name, out_dir,
                              boxplot_n=50_000, write_per_sample_csv=False):
    """
    Write the final test outputs for one method:
      - paper tables 1-5 + box plot (via paper_outputs.py)
      - table_1_extended_<method>.csv, extended 12-box box plot
      - results_paper_<method>.json
      - opinions_per_sample_<method>.csv (optional)
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    per_attack_results = {}
    pooled_idx_list = []

    fcsv = None
    if write_per_sample_csv:
        csv_path = os.path.join(out_dir, f"opinions_per_sample_{method_name}.csv")
        fcsv = open(csv_path, "w", encoding="utf-8")
        fcsv.write("attack,y_true,y_pred,correct,p,b,d,u\n")

    for atk in SELECTED_ATTACKS:
        idx = np.where(attack_all == atk)[0]
        if len(idx) == 0:
            continue
        y_t, yp_t = y_true_all[idx], y_pred_all[idx]
        correct_t = (yp_t == y_t)
        p_t, b_t, d_t, u_t = p_all[idx], b_all[idx], d_all[idx], u_all[idx]

        if fcsv is not None:  # stream rows to keep memory flat
            for i in range(len(idx)):
                fcsv.write(f"{atk},{int(y_t[i])},{int(yp_t[i])},{int(correct_t[i])},"
                           f"{p_t[i]:.6f},{b_t[i]:.6f},{d_t[i]:.6f},{u_t[i]:.6f}\n")

        m = evaluate_all(y_true=y_t, y_prob=p_t, y_pred=yp_t, uncertainty=u_t)
        bc, bcb, dca = belief_correctness(b_t, d_t, y_t, yp_t)
        groups = {
            "correct": correct_t, "wrong": ~correct_t,
            "benign": (y_t == 1), "malicious": (y_t == 0),
        }
        entry = {
            "f1":             f1_attacker(y_t, yp_t),
            "aurc":           m.get("aurc", float("nan")),
            "misclass_auroc": m.get("misclass_auroc", float("nan")),
            "ece":            m.get("ece", float("nan")),
            "brier":          m.get("brier", float("nan")),
        }
        for comp, arr in (("u", u_t), ("b", b_t), ("d", d_t)):
            for gname, gmask in groups.items():
                mean, median, std = _mean_median_std(arr[gmask])
                entry[f"{comp}_{gname}_mean"] = mean
                entry[f"{comp}_{gname}_median"] = median
                if comp == "u" and gname in ("correct", "wrong"):
                    entry[f"u_{gname}_std"] = std
        entry.update({
            "b_correct_benign_mean":   bcb,
            "d_correct_attacker_mean": dca,
            "belief_correctness":      bc,
            "n_correct":   int(correct_t.sum()),
            "n_wrong":     int((~correct_t).sum()),
            "n_benign":    int(groups["benign"].sum()),
            "n_malicious": int(groups["malicious"].sum()),
        })
        per_attack_results[atk] = entry
        pooled_idx_list.append(idx)

    if fcsv is not None:
        fcsv.close()
        print(f"  per-sample CSV:   {csv_path}")

    if not pooled_idx_list:
        print("  [WARN] no test data found")
        return None

    # --- Pooled (micro) metrics over all selected attacks ---
    pooled_idx = np.concatenate(pooled_idx_list)
    y_pool, yp_pool = y_true_all[pooled_idx], y_pred_all[pooled_idx]
    p_pool, u_pool = p_all[pooled_idx], u_all[pooled_idx]
    b_pool, d_pool = b_all[pooled_idx], d_all[pooled_idx]
    correct_pool = (yp_pool == y_pool)
    m_pool = evaluate_all(y_true=y_pool, y_prob=p_pool, y_pred=yp_pool, uncertainty=u_pool)
    u_c_pool, u_w_pool = u_pool[correct_pool], u_pool[~correct_pool]
    has_both = len(u_c_pool) > 0 and len(u_w_pool) > 0

    overall_micro = {
        "f1":             f1_attacker(y_pool, yp_pool),
        "aurc":           m_pool.get("aurc", float("nan")),
        "misclass_auroc": m_pool.get("misclass_auroc", float("nan")),
        "ece":            m_pool.get("ece", float("nan")),
        "brier":          m_pool.get("brier", float("nan")),
        "risk_at_70":     _risk_at_coverage(u_pool, correct_pool, 0.70),
        "risk_at_80":     _risk_at_coverage(u_pool, correct_pool, 0.80),
        "risk_at_90":     _risk_at_coverage(u_pool, correct_pool, 0.90),
        "risk_at_100":    float(1.0 - correct_pool.mean()),
        "delta_u_mean":   float(u_w_pool.mean() - u_c_pool.mean()) if has_both else float("nan"),
        "delta_u_median": (float(np.median(u_w_pool) - np.median(u_c_pool))
                           if has_both else float("nan")),
    }

    def _macro(key):
        return _nanmean([pa[key] for pa in per_attack_results.values()])

    macro = {
        "macro_f1":                 _macro("f1"),
        "macro_aurc":               _macro("aurc"),
        "macro_misclass_auroc":     _macro("misclass_auroc"),
        "macro_ece":                _macro("ece"),
        "macro_brier":              _macro("brier"),
        "macro_delta_u":            _nanmean([pa["u_wrong_mean"] - pa["u_correct_mean"]
                                              for pa in per_attack_results.values()]),
        "macro_belief_correctness": _macro("belief_correctness"),
    }

    def _sample(arr):
        return arr if len(arr) <= boxplot_n else rng.choice(arr, size=boxplot_n, replace=False)

    pool_groups = {
        "correct": correct_pool, "wrong": ~correct_pool,
        "benign": (y_pool == 1), "malicious": (y_pool == 0),
    }
    boxplot_samples = {
        f"{comp}_{gname}": np.asarray(_sample(arr[gmask]))
        for comp, arr in (("u", u_pool), ("b", b_pool), ("d", d_pool))
        for gname, gmask in pool_groups.items()
    }
    method_entry = {
        "method_name":     method_name,
        "macro":           macro,
        "overall_micro":   overall_micro,
        "per_attack":      per_attack_results,
        "boxplot_samples": boxplot_samples,
    }

    # --- Extended table 1 ---
    ext = _extended_delta_metrics(u=u_pool, b=b_pool, d=d_pool, y_pred=yp_pool,
                                  y_true=y_pool, atk_labels=attack_all[pooled_idx])
    method_entry["extended"] = ext
    ext_csv = os.path.join(out_dir, f"table_1_extended_{method_name}.csv")
    core_cols = list(OBJECTIVE_NAMES)
    ext_cols = list(ext.keys())
    with open(ext_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + core_cols + ext_cols)
        w.writeheader()
        w.writerow({"method": method_name,
                    **{k: macro.get(k, float("nan")) for k in core_cols},
                    **ext})
    print(f"  table_1 extended: {ext_csv}")
    print("    " + "  ".join(f"{k}={ext[k]:.4f}" for k in
                            ("delta_u_macro_mean", "confident_wrong_frac_macro",
                             "delta_b_macro", "delta_d_macro")))

    # --- Results JSON ---
    json_out = {**{k: v for k, v in method_entry.items() if k != "boxplot_samples"},
                "boxplot_samples": {k: v.tolist() for k, v in boxplot_samples.items()}}
    json_path = os.path.join(out_dir, f"results_paper_{method_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"  results JSON:     {json_path}")

    # --- Paper tables 1-5 + box plots ---
    paper_outputs.write_all_tables_and_boxplot(
        methods=[method_entry], attacks=SELECTED_ATTACKS,
        out_dir=out_dir, suffix=method_name)
    ext_box_path = os.path.join(out_dir, f"boxplot_opinions_extended_{method_name}.png")
    _render_extended_boxplot(boxplot_samples, ext_box_path, title_suffix=f"({method_name})")
    print(f"  extended boxplot: {ext_box_path}")

    print(f"\n  Macro F1: {macro['macro_f1']:.4f}   AURC: {macro['macro_aurc']:.4f}   "
          f"MAUROC: {macro['macro_misclass_auroc']:.4f}   delta_u: {macro['macro_delta_u']:.4f}   "
          f"BeliefCorr: {macro['macro_belief_correctness']:.4f}")
    return method_entry


def run_final_test(ctx, best_state, method_name, out_dir, write_per_sample_csv=False):
    """
    Evaluate ``best_state`` on the full test set (threshold DECISION_THRESHOLD,
    same as in the sweep). Also saves the model.
    """
    print(f"\n{'=' * 70}\nFINAL TEST RUN: {method_name}\n{'=' * 70}")
    device = ctx.device
    w_b = best_state["w_b_joint"].to(device)
    w_u = best_state["w_u_joint"].to(device)
    t_star = DECISION_THRESHOLD
    print(f"  Decision threshold: t* = {t_star:.3f}")

    p_te, (b_te, d_te, u_te), _ = forward_prob_chunked(
        ctx.Phi_te_by_f, w_b, w_u, device=device)
    p_all, b_all, d_all, u_all = p_te.numpy(), b_te.numpy(), d_te.numpy(), u_te.numpy()
    y_true = ctx.y_te.reshape(-1)

    method_entry = build_paper_outputs_final(
        p_all=p_all, u_all=u_all, b_all=b_all, d_all=d_all,
        y_pred_all=(p_all >= t_star).astype(int), y_true_all=y_true,
        attack_all=ctx.attack_te, method_name=method_name, out_dir=out_dir,
        write_per_sample_csv=write_per_sample_csv)

    model = {
        "splines":         ctx.splines,
        "w_b_joint":       best_state["w_b_joint"],
        "w_u_joint":       best_state["w_u_joint"],
        "feature_offsets": ctx.feature_offsets,
        "threshold":       t_star,
        "features":        FEATURE_NAMES,
    }
    model_path = os.path.join(out_dir, f"bspline_trust_model_{method_name}.pt")
    torch.save(model, model_path)
    print(f"\n  Model: {model_path}")
    return method_entry


# ============================================================================
# Optuna study
# ============================================================================

def create_or_load_study(study_name, storage, resume, population_size, seed):
    sampler = optuna.samplers.NSGAIISampler(
        population_size=population_size,
        crossover=optuna.samplers.nsgaii.UniformCrossover(),
        crossover_prob=NSGA_CROSSOVER_PROB,
        swapping_prob=NSGA_SWAPPING_PROB,
        seed=seed,
    )
    if resume:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
            print(f"[Optuna] Study loaded - {len(study.trials)} trials so far.")
        except KeyError:
            print("[Optuna] No existing study found, creating a new one.")
            study = None
    else:
        study = None
    if study is None:
        study = optuna.create_study(study_name=study_name, storage=storage,
                                    directions=OBJECTIVE_DIRECTION, sampler=sampler,
                                    load_if_exists=True)
    if len(study.directions) != len(OBJECTIVE_DIRECTION):
        raise RuntimeError(
            f"Schema mismatch: study '{study_name}' has {len(study.directions)} objectives, "
            f"this script uses {len(OBJECTIVE_DIRECTION)}. Use a different --study-name.")
    return study


def suggest_hyperparameters(trial):
    hp = {}
    for name, (low, high, step) in SEARCH_SPACE.items():
        hp[name] = trial.suggest_float(name, low, high, step=step)
    return hp


# ============================================================================
# Sweep analysis outputs
# ============================================================================

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
            **t.params,
        }
        row.update(zip(OBJECTIVE_NAMES, t.values))
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {path}")


def plot_pareto_2d(study, obj_x, obj_y, out_path):
    idx_x, idx_y = OBJECTIVE_NAMES.index(obj_x), OBJECTIVE_NAMES.index(obj_y)
    arrow = {"maximize": "↑", "minimize": "↓"}
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    pareto_set = {t.number for t in study.best_trials}
    dominated = [t for t in completed if t.number not in pareto_set]
    pareto = sorted((t for t in completed if t.number in pareto_set),
                    key=lambda t: t.values[idx_x])

    fig, ax = plt.subplots(figsize=(9, 6))
    if dominated:
        ax.scatter([t.values[idx_x] for t in dominated], [t.values[idx_y] for t in dominated],
                   c="#888888", s=60, alpha=0.5, label=f"dominated (n={len(dominated)})")
    if pareto:
        px = [t.values[idx_x] for t in pareto]
        py = [t.values[idx_y] for t in pareto]
        ax.scatter(px, py, c="#D85A30", s=120, alpha=0.95, edgecolors="black",
                   linewidths=1.2, label=f"Pareto (n={len(pareto)})", zorder=3)
        ax.plot(px, py, c="#D85A30", alpha=0.4, linestyle="--", zorder=2)
    for t in completed:
        ax.annotate(str(t.number), (t.values[idx_x], t.values[idx_y]),
                    fontsize=7, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"{obj_x}  ({arrow[OBJECTIVE_DIRECTION[idx_x]]})")
    ax.set_ylabel(f"{obj_y}  ({arrow[OBJECTIVE_DIRECTION[idx_y]]})")
    ax.set_title(f"Pareto: {obj_x} vs {obj_y}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_argument_group("sweep")
    g.add_argument("--n-trials", type=int, default=1000,
                   help="Total number of completed trials to reach (default: 1000)")
    g.add_argument("--num-epochs", type=int, default=NUM_EPOCHS_PER_TRIAL,
                   help=f"Max. training epochs per trial (default: {NUM_EPOCHS_PER_TRIAL})")
    g.add_argument("--population-size", type=int, default=NSGA_POPULATION_SIZE,
                   help=f"NSGA-II population size (default: {NSGA_POPULATION_SIZE})")
    g.add_argument("--seed", type=int, default=SEED, help=f"Sampler seed (default: {SEED})")
    g.add_argument("--study-name", default=STUDY_NAME_DEFAULT)
    g.add_argument("--storage", default=None,
                   help="Optuna storage URL (default: sqlite:///<out-dir>/optuna.db)")

    g = parser.add_argument_group("paths")
    g.add_argument("--out-dir", default=OUT_DIR_DEFAULT,
                   help=f"Output directory (default: {OUT_DIR_DEFAULT}, env MBD_SWEEP_DIR)")
    g.add_argument("--cache-dir", default=dc.CACHE_DIR_DEFAULT,
                   help=f"Feature cache directory (default: {dc.CACHE_DIR_DEFAULT})")
    g.add_argument("--data-root", default=dc.DATA_ROOT_DEFAULT,
                   help="Raw JSON dataset, only used if the cache must be built")
    g.add_argument("--num-workers", type=int, default=dc.NUM_WORKERS_DEFAULT,
                   help="Worker processes for building the cache")

    g = parser.add_argument_group("mode")
    g.add_argument("--resume", action="store_true", help="Continue an existing study")
    g.add_argument("--analyze-only", action="store_true",
                   help="Skip the sweep: analysis, knee selection and final test only")
    g.add_argument("--final-test-only", action="store_true",
                   help="Skip sweep and analysis plots: knee selection and final test only")
    g.add_argument("--per-sample-csv", action="store_true",
                   help="Also write per-sample opinions of the test set (large)")
    g.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{out_dir / 'optuna.db'}"
    trial_states_file = str(out_dir / "trial_states.pt")

    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[Sweep] Device:     {device}")
    if use_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Sweep] Objectives: {OBJECTIVE_NAMES}")
    print(f"[Sweep] Directions: {OBJECTIVE_DIRECTION}")
    print(f"[Sweep] Output:     {out_dir}/")

    ctx = build_data_context(device, args.cache_dir, args.data_root, args.num_workers)

    skip_sweep = args.analyze_only or args.final_test_only
    study = create_or_load_study(args.study_name, storage,
                                 resume=args.resume or skip_sweep,
                                 population_size=args.population_size, seed=args.seed)
    trial_states = load_trial_states(trial_states_file)

    # ---------- Phase 1: sweep ----------
    if not skip_sweep:
        n_completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        n_remaining = args.n_trials - n_completed
        if n_remaining <= 0:
            print(f"\n[Sweep] Target ({args.n_trials}) already reached ({n_completed} trials).")
        else:
            print(f"\n[Sweep] {n_completed} done, running {n_remaining} more trials")

            def objective(trial):
                hp = suggest_hyperparameters(trial)
                print(f"\n{'─' * 60}\nTrial {trial.number}: "
                      + ", ".join(f"{k}={v:.4g}" for k, v in hp.items()))
                t_start = time.time()
                best_state, val_metrics = train_single_trial(ctx, hp, num_epochs=args.num_epochs)
                dt = time.time() - t_start

                trial_states[trial.number] = best_state
                save_trial_states(trial_states, trial_states_file)
                trial.set_user_attr("duration_min", dt / 60)
                for k, v in val_metrics.items():
                    trial.set_user_attr(k, v)
                print(f"  done in {dt / 60:.1f} min:  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

                for name in OBJECTIVE_NAMES:
                    if np.isnan(val_metrics.get(name, float("nan"))):
                        print(f"  [WARN] {name} is NaN -> trial pruned")
                        raise optuna.TrialPruned()
                return tuple(val_metrics[n] for n in OBJECTIVE_NAMES)

            t_sweep = time.time()
            study.optimize(objective, n_trials=n_remaining,
                           gc_after_trial=True, show_progress_bar=False)
            print(f"\n[Sweep] Total sweep time: {(time.time() - t_sweep) / 60:.1f} min")

    # ---------- Phase 2: analysis (CSV + Pareto plots) ----------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[ERROR] No completed trials - aborting.")
        return
    print(f"\n{'=' * 70}\nSWEEP ANALYSIS  ({len(completed)} completed trials)\n{'=' * 70}")
    print(f"  Pareto front: {len(study.best_trials)} trials")
    if not args.final_test_only:
        write_summary_csv(study, str(out_dir / "sweep_summary.csv"))
        for i, obj_x in enumerate(OBJECTIVE_NAMES):
            for obj_y in OBJECTIVE_NAMES[i + 1:]:
                plot_pareto_2d(study, obj_x, obj_y, str(out_dir / f"pareto_{obj_x}_vs_{obj_y}.png"))

    # ---------- Phase 3: balanced trial ----------
    winner, info = select_balanced_trial(study)
    balanced_path = out_dir / "balanced_trial.json"
    with open(balanced_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"  balanced JSON: {balanced_path}")

    # ---------- Phase 4: final test ----------
    trial_states = load_trial_states(trial_states_file)
    if winner not in trial_states:
        print(f"\n[ERROR] Best state for trial {winner} missing in {trial_states_file}.")
        print("        Was the sweep run with the same --out-dir?")
        return
    method_name = f"BSpline_balanced_trial{winner:03d}"
    final_dir = str(out_dir / "final_test")
    run_final_test(ctx, trial_states[winner], method_name=method_name, out_dir=final_dir,
                   write_per_sample_csv=args.per_sample_csv)
    print(f"\n[Sweep] Done. Final outputs in {final_dir}/")


if __name__ == "__main__":
    main()
