"""
Closed-form Subjective Logic (SL) opinion model for GNSS spoofing detection.

Each of the four grouped Mahalanobis distances (d_power, d_shape, d_track,
d_dyn), normalized to [0, 1) by data_cache_gnss.py, is mapped to a per-feature
SL opinion (b, d, u). The four opinions are then fused into one opinion per
sample with CBF, AVG or WBF.

Convention:
    b = belief in 'benign'
    d = disbelief = belief in 'attacker' (spoofed)
    u = uncertainty mass
    truth: 1 = benign, 0 = attacker
"""
from enum import Enum

import numpy as np

# Fixed operating point: p = b + 0.5*u >= 0.5  <=>  b >= d.
DECISION_THR = 0.5


class Attribute(Enum):
    d_power = 0
    d_shape = 1
    d_track = 2
    d_dyn = 3
    truth = 4


_FEATURE_IDX = [
    Attribute.d_power.value,
    Attribute.d_shape.value,
    Attribute.d_track.value,
    Attribute.d_dyn.value,
]


# ============================================================================
# Per-feature opinion mapping and fusion operators (vectorized)
# ============================================================================

def _compute_feature_opinion_vec(X, thr, alpB, betB, alpD, betD):
    """
    Vectorized per-feature SL opinion.

        X <= thr:  b = alpB * (1 - exp(-betB * (X - thr)^2)),  d = 0
        X >  thr:  d = alpD * (1 - exp(-betD * (X - thr)^2)),  b = 0
        u = 1 - b - d

    Parameters
    ----------
    X    : (N,) numpy array — feature values
    thr, alpB, betB, alpD, betD : scalars — per-feature hyperparameters

    Returns
    -------
    b, d, u : (N,) numpy arrays
    """
    delta_sq = (X - thr) ** 2
    mask_b = X <= thr
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
    # Degenerate: both u = 0 -> plain average
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
    # Degenerate: u1 = u2 = 0 -> average weighted by c_i
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
# Main entry point: vectorized computation of the fused opinions
# ============================================================================

def compute_opinions(msgs, alpB, betB, alpD, betD, thr, fusion_op="cbf", trust=None):
    """
    Vectorized computation of the fused SL opinions over ALL samples.

    Fuses cumulatively feature by feature instead of holding all per-feature
    arrays at once, which keeps peak memory low.

    Parameters
    ----------
    msgs       : (N, 5) array-like — feature matrix, last column = truth
    alpB, betB, alpD, betD, thr : list/array of 4 scalars — per-feature HPs
    fusion_op  : "cbf" | "avg" | "wbf"
    trust      : optional list/array of 4 scalars in [0, 1] — trust discount;
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

    # Sequentially fuse the remaining features
    for k in range(1, len(_FEATURE_IDX)):
        fk = _FEATURE_IDX[k]
        b_k, d_k, u_k = _compute_feature_opinion_vec(
            msgs_arr[:, fk], thr[fk], alpB[fk], betB[fk], alpD[fk], betD[fk]
        )
        if use_trust:
            b_k, d_k, u_k = _apply_trust_vec(b_k, d_k, u_k, float(trust[k]))
        b_acc, d_acc, u_acc = fuse_vec(b_acc, d_acc, u_acc, b_k, d_k, u_k)

    return b_acc, d_acc, u_acc, t_arr
