import math
from enum import Enum

import numpy as np


class Attribute(Enum):
    pos_dist = 0
    speed_dist = 1
    acc_dist = 2
    road_edge = 3
    time_err = 4
    heading = 5
    truth = 6


_FEATURE_IDX = [
    Attribute.pos_dist.value,
    Attribute.speed_dist.value,
    Attribute.acc_dist.value,
    Attribute.road_edge.value,
    Attribute.time_err.value,
    Attribute.heading.value,
]


# ============================================================================
# SCALAR per-feature SL opinion (backward compat / debug)
# ============================================================================

def getSLRelPosX(thr, alpB, betB, alpD, betD, X):
    """
    Convention:
      b = belief in 'benign'
      d = disbelief = belief in 'attacker'
      u = uncertainty mass
    """
    if X <= thr:
        b = alpB * (1.0 - math.exp(-betB * (math.fabs(X - thr) ** 2)))
        d = 0.0
        u = 1.0 - b
    else:
        b = 0.0
        d = alpD * (1.0 - math.exp(-betD * (math.fabs(X - thr) ** 2)))
        u = 1.0 - d
    return b, d, u


# ============================================================================
# SCALAR fusion operators (Josang) — backward compat
# ============================================================================

def fuse_two_cbf(b1, d1, u1, b2, d2, u2, eps=1e-12):
    """Cumulative Belief Fusion. Assumes independent sources."""
    k = u1 + u2 - u1 * u2
    if k < eps:
        k = eps
    b = (b1 * u2 + b2 * u1) / k
    d = (d1 * u2 + d2 * u1) / k
    u = (u1 * u2) / k
    s = b + d + u
    if s < eps:
        return 0.0, 0.0, 1.0
    return b / s, d / s, u / s


def fuse_two_avg(b1, d1, u1, b2, d2, u2, eps=1e-12):
    """Averaging Belief Fusion. Assumes dependent/redundant sources."""
    k = u1 + u2
    if k < eps:
        return (b1 + b2) / 2.0, (d1 + d2) / 2.0, 0.0
    b = (b1 * u2 + b2 * u1) / k
    d = (d1 * u2 + d2 * u1) / k
    u = (2.0 * u1 * u2) / k
    s = b + d + u
    if s < eps:
        return 0.0, 0.0, 1.0
    return b / s, d / s, u / s


def fuse_two_wbf(b1, d1, u1, b2, d2, u2, eps=1e-12):
    """Weighted Belief Fusion. Sources with lower u get more weight."""
    c1 = 1.0 - u1
    c2 = 1.0 - u2
    k = u1 + u2 - 2.0 * u1 * u2
    if k < eps:
        gamma_total = c1 + c2
        if gamma_total < eps:
            return 0.0, 0.0, 1.0
        return (c1 * b1 + c2 * b2) / gamma_total, (c1 * d1 + c2 * d2) / gamma_total, 0.0
    b = (b1 * c1 * u2 + b2 * c2 * u1) / k
    d = (d1 * c1 * u2 + d2 * c2 * u1) / k
    u = ((2.0 - u1 - u2) * u1 * u2) / k
    s = b + d + u
    if s < eps:
        return 0.0, 0.0, 1.0
    return b / s, d / s, u / s


_FUSE_TWO = {
    "cbf": fuse_two_cbf,
    "avg": fuse_two_avg,
    "wbf": fuse_two_wbf,
}


def fuse_many(opinions, fusion_op="cbf"):
    """Sequential pairwise fusion (scalar). Default: CBF."""
    fuse = _FUSE_TWO[fusion_op]
    b, d, u = opinions[0]
    for (b2, d2, u2) in opinions[1:]:
        b, d, u = fuse(b, d, u, b2, d2, u2)
    return b, d, u


def fuse_two(b1, d1, u1, b2, d2, u2, eps=1e-8):
    return fuse_two_cbf(b1, d1, u1, b2, d2, u2, eps=eps)


# ============================================================================
# Trust discount (scalar + vectorized)
# ============================================================================

def discount_opinion(b, d, u, w, eps=1e-12):
    """
    Scalar trust discount (Josang).
        w = 1: opinion unchanged.
        w = 0: fully discounted (u = 1).
    """
    if w >= 1.0 - eps:
        return b, d, u
    if w <= eps:
        return 0.0, 0.0, 1.0
    return w * b, w * d, 1.0 - w * (b + d)


# ============================================================================
# VECTORIZED versions (for compute_opinions)
# ============================================================================

def _compute_feature_opinion_vec(X, thr, alpB, betB, alpD, betD):
    """
    Vectorized per-feature SL opinion.

    Parameters
    ----------
    X    : (N,) numpy array — feature values
    thr, alpB, betB, alpD, betD : scalars — per-feature HPs

    Returns
    -------
    b, d, u : (N,) numpy arrays
    """
    delta_sq = (X - thr) ** 2
    mask_b = X <= thr
    # Numerical stability: clipping for very large bet * delta_sq
    b = np.where(mask_b, alpB * (1.0 - np.exp(-betB * delta_sq)), 0.0)
    d = np.where(~mask_b, alpD * (1.0 - np.exp(-betD * delta_sq)), 0.0)
    u = 1.0 - b - d
    return b, d, u


def _apply_trust_vec(b, d, u, w, eps=1e-12):
    """Vectorized trust discount for a single (scalar) w."""
    if w >= 1.0 - eps:
        return b, d, u
    if w <= eps:
        return np.zeros_like(b), np.zeros_like(d), np.ones_like(u)
    b_new = w * b
    d_new = w * d
    u_new = 1.0 - b_new - d_new
    return b_new, d_new, u_new


def _normalize_bdu(b, d, u, eps=1e-12):
    """Element-wise normalization (b+d+u=1) with handling of degenerate cases."""
    s = b + d + u
    valid = s > eps
    s_safe = np.where(valid, s, 1.0)
    return (
        np.where(valid, b / s_safe, 0.0),
        np.where(valid, d / s_safe, 0.0),
        np.where(valid, u / s_safe, 1.0),
    )


def _fuse_two_cbf_vec(b1, d1, u1, b2, d2, u2, eps=1e-12):
    """Vectorized Cumulative Belief Fusion (element-wise)."""
    k = u1 + u2 - u1 * u2
    k_safe = np.maximum(k, eps)
    b = (b1 * u2 + b2 * u1) / k_safe
    d = (d1 * u2 + d2 * u1) / k_safe
    u = (u1 * u2) / k_safe
    return _normalize_bdu(b, d, u, eps)


def _fuse_two_avg_vec(b1, d1, u1, b2, d2, u2, eps=1e-12):
    """Vectorized Averaging Belief Fusion (element-wise)."""
    k = u1 + u2
    k_safe = np.maximum(k, eps)
    b_std = (b1 * u2 + b2 * u1) / k_safe
    d_std = (d1 * u2 + d2 * u1) / k_safe
    u_std = (2.0 * u1 * u2) / k_safe
    # Degenerate: both u = 0 -> plain mean
    deg = k < eps
    b_deg = 0.5 * (b1 + b2)
    d_deg = 0.5 * (d1 + d2)
    b = np.where(deg, b_deg, b_std)
    d = np.where(deg, d_deg, d_std)
    u = np.where(deg, 0.0, u_std)
    return _normalize_bdu(b, d, u, eps)


def _fuse_two_wbf_vec(b1, d1, u1, b2, d2, u2, eps=1e-12):
    """Vectorized Weighted Belief Fusion (element-wise)."""
    c1 = 1.0 - u1
    c2 = 1.0 - u2
    k = u1 + u2 - 2.0 * u1 * u2
    k_safe = np.maximum(k, eps)
    b_std = (b1 * c1 * u2 + b2 * c2 * u1) / k_safe
    d_std = (d1 * c1 * u2 + d2 * c2 * u1) / k_safe
    u_std = ((2.0 - u1 - u2) * u1 * u2) / k_safe
    # Degenerate: u1 = u2 = 0 -> mean weighted by c_i
    gamma = c1 + c2
    gamma_safe = np.maximum(gamma, eps)
    b_deg = (c1 * b1 + c2 * b2) / gamma_safe
    d_deg = (c1 * d1 + c2 * d2) / gamma_safe
    deg = k < eps
    b = np.where(deg, b_deg, b_std)
    d = np.where(deg, d_deg, d_std)
    u = np.where(deg, 0.0, u_std)
    return _normalize_bdu(b, d, u, eps)


_FUSE_TWO_VEC = {
    "cbf": _fuse_two_cbf_vec,
    "avg": _fuse_two_avg_vec,
    "wbf": _fuse_two_wbf_vec,
}


# ============================================================================
# Main function: VECTORIZED computation of the fused opinions
# ============================================================================

def compute_opinions(msgs, alpB, betB, alpD, betD, thr, fusion_op="cbf", trust=None):
    """
    Vectorized computation of the fused SL opinions over ALL samples.

    Does not keep 6 per-feature arrays at once — fuses cumulatively, feature
    by feature, so peak memory stays at ~6*N*8 bytes.

    Parameters
    ----------
    msgs       : (N, >=7) array-like — feature matrix (last column = truth)
    alpB, betB, alpD, betD, thr : list/array of 6 scalars — per-feature HPs
    fusion_op  : "cbf" | "avg" | "wbf"
    trust      : optional list/array of 6 scalars in [0, 1] — trust discount
                 None = no discount (default)

    Returns
    -------
    b, d, u, truth : (N,) numpy arrays
    """
    msgs_arr = np.asarray(msgs, dtype=np.float64)
    truth_idx = Attribute.truth.value
    t_arr = msgs_arr[:, truth_idx].astype(np.int64)

    fuse_vec = _FUSE_TWO_VEC[fusion_op]
    use_trust = trust is not None

    # Initialize with feature 0
    f0 = _FEATURE_IDX[0]
    b_acc, d_acc, u_acc = _compute_feature_opinion_vec(
        msgs_arr[:, f0], thr[f0], alpB[f0], betB[f0], alpD[f0], betD[f0]
    )
    if use_trust:
        b_acc, d_acc, u_acc = _apply_trust_vec(b_acc, d_acc, u_acc, float(trust[0]))

    # Sequential fusion with the remaining features
    for k in range(1, len(_FEATURE_IDX)):
        fk = _FEATURE_IDX[k]
        b_k, d_k, u_k = _compute_feature_opinion_vec(
            msgs_arr[:, fk], thr[fk], alpB[fk], betB[fk], alpD[fk], betD[fk]
        )
        if use_trust:
            b_k, d_k, u_k = _apply_trust_vec(b_k, d_k, u_k, float(trust[k]))
        b_acc, d_acc, u_acc = fuse_vec(b_acc, d_acc, u_acc, b_k, d_k, u_k)

    return b_acc, d_acc, u_acc, t_arr


def getF1(msgs, alpB, betB, alpD, betD, thr, decisionThr, fusion_op="cbf", trust=None):
    """Backward compat. Returns (f1, tp, tn, fp, fn) with attacker = positive class."""
    b, _d, u, truth = compute_opinions(
        msgs, alpB, betB, alpD, betD, thr, fusion_op=fusion_op, trust=trust
    )
    p_benign = b + 0.5 * u
    y_pred = (p_benign > decisionThr).astype(np.int64)

    tp = int(np.sum((truth == 0) & (y_pred == 0)))
    fp = int(np.sum((truth == 1) & (y_pred == 0)))
    tn = int(np.sum((truth == 1) & (y_pred == 1)))
    fn = int(np.sum((truth == 0) & (y_pred == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, tp, tn, fp, fn
