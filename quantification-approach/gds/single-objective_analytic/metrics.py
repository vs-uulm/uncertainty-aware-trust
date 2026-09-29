"""
Evaluation metrics — signature and semantics identical to the sl_metrics
module of the B-spline approach, so the numbers are directly comparable.

Convention (same as the B-spline code):
    y_true: 0 = attacker, 1 = benign
    y_prob: P(benign) = b + 0.5*u     (positive class for AUC = benign / 1)
    y_pred: 0/1, same convention as y_true
    u:      uncertainty mass
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# numpy >= 2 renamed trapz -> trapezoid; older versions only have trapz.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz   # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _brier(y_prob, y_true):
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def _ece(y_prob, y_true, n_bins=15):
    """
    Standard binary ECE.
    Bins by max-class confidence; accuracy at the 0.5 cutoff.
    """
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    n = len(p)
    if n == 0:
        return float("nan")

    y_pred = (p >= 0.5).astype(np.int64)
    conf = np.where(y_pred == 1, p, 1.0 - p)
    correct = (y_pred == y).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def _aurc(confidence, correct):
    """
    Area Under the Risk-Coverage curve.
    confidence: higher = more certain    correct: 1 = correct, 0 = wrong
    """
    n = len(confidence)
    if n == 0:
        return float("nan")
    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="mergesort")
    err_sorted = (1 - np.asarray(correct, dtype=np.int64))[order].astype(np.float64)
    risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    return float(_TRAPZ(risk, coverage))


def _risk_at_coverage(confidence, correct, coverage=0.7):
    n = len(confidence)
    if n == 0:
        return float("nan")
    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="mergesort")
    correct_sorted = np.asarray(correct, dtype=np.int64)[order]
    k = max(1, int(round(coverage * n)))
    return float(1.0 - correct_sorted[:k].mean())


def _misclass_auroc(uncertainty, correct):
    """
    AUROC with uncertainty as the score, positive class = 'misclassified'.
    Mann-Whitney U via ranks with tie handling.
    """
    u = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    wrong = (1 - np.asarray(correct, dtype=np.int64)).astype(np.int64)
    n_pos = int(wrong.sum())
    n_neg = int(len(wrong) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    n = len(u)
    order = np.argsort(u, kind="mergesort")
    sorted_u = u[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_u[j + 1] == sorted_u[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[wrong == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _delta_u(uncertainty, correct, statistic="median"):
    """
    Uncertainty gap between misclassified and correctly classified samples:
        stat(u | wrong) - stat(u | correct)

    statistic : "median" (robust, 50% breakdown point) or "mean"
                (sensitive to confident-wrong tails, but also to outliers).

    Higher is better. Negative values mean the model tends to be more certain
    on its errors than on its correct predictions.
    """
    assert statistic in ("median", "mean"), statistic
    agg = np.median if statistic == "median" else np.mean
    u = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.int64).reshape(-1)
    if len(u) == 0:
        return float("nan")
    u_wrong = u[correct == 0]
    u_correct = u[correct == 1]
    if len(u_wrong) == 0 or len(u_correct) == 0:
        return float("nan")
    return float(agg(u_wrong) - agg(u_correct))


def _macro_delta_u(uncertainty, correct, attack_idx, n_attacks, statistic="median"):
    """Mean of the per-attack _delta_u. Skips single-class groups."""
    deltas = []
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    attack_idx = np.asarray(attack_idx, dtype=np.int64)
    for k in range(n_attacks):
        mask = attack_idx == k
        if int(mask.sum()) < 2:
            continue
        d = _delta_u(uncertainty[mask], correct[mask], statistic=statistic)
        if not np.isnan(d):
            deltas.append(d)
    return float(np.mean(deltas)) if deltas else float("nan")


def _delta_u_median(uncertainty, correct):
    """
    Median uncertainty gap:
        median(u | misclassified) - median(u | correctly classified)

    Higher is better, because misclassifications should ideally carry higher
    uncertainty than correctly classified samples. Negative values mean the
    model tends to be more certain on its errors.
    """
    return _delta_u(uncertainty, correct, statistic="median")


# ---------------------------------------------------------------------------
# Macro variants: mean over per-attack values
# ---------------------------------------------------------------------------

def _macro_aurc(confidence, correct, attack_idx, n_attacks):
    """
    Mean of the per-attack AURC (every attack weighted equally).
    Skips attacks without samples or with only one sample.
    """
    aurcs = []
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    attack_idx = np.asarray(attack_idx, dtype=np.int64)
    for k in range(n_attacks):
        mask = attack_idx == k
        n_k = int(mask.sum())
        if n_k < 2:
            continue
        aurcs.append(_aurc(confidence[mask], correct[mask]))
    return float(np.mean(aurcs)) if aurcs else float("nan")


def _macro_misclass_auroc(uncertainty, correct, attack_idx, n_attacks):
    """
    Mean of the per-attack misclassification AUROC (every attack weighted
    equally). Skips attacks with only correct or only wrong predictions
    (n_pos/n_neg = 0).
    """
    aurocs = []
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int64)
    attack_idx = np.asarray(attack_idx, dtype=np.int64)
    for k in range(n_attacks):
        mask = attack_idx == k
        n_k = int(mask.sum())
        if n_k < 2:
            continue
        a = _misclass_auroc(uncertainty[mask], correct[mask])
        if not np.isnan(a):
            aurocs.append(a)
    return float(np.mean(aurocs)) if aurocs else float("nan")


def _macro_delta_u_median(uncertainty, correct, attack_idx, n_attacks):
    """
    Mean of the per-attack median Δu:
        median(u_wrong) - median(u_correct)

    Skips attacks without any correct or without any wrong predictions.
    """
    return _macro_delta_u(uncertainty, correct, attack_idx, n_attacks,
                          statistic="median")


# ---------------------------------------------------------------------------
# Sensoy evidential loss from an SL opinion (diagnostic: the MLP-equivalent
# training loss, evaluated on the closed-form model)
# ---------------------------------------------------------------------------

def evidential_loss_from_opinion(b, d, u, truth, kl_weight=1.0, eps=1e-9):
    """
    Computes Sensoy's MSE + lambda*KL evidential loss directly from
    Subjective Logic opinions (b, d, u).

    Convention:
        truth = 1  -> benign   (class "b")
        truth = 0  -> attacker (class "d")

    Conversion SL -> Dirichlet (binary, K=2, uniform prior W=2):
        S       = 2 / u           (Dirichlet concentration)
        alpha_b = 1 + b * S
        alpha_d = 1 + d * S

    Loss definition (Sensoy et al., NeurIPS 2018):
        L_i   = MSE_i + kl_weight * KL_i
        MSE_i = sum_k [(y_k - p_k)^2 + p_k(1-p_k)/(S+1)]
        KL_i  = KL(Dir(alpha_tilde_i) || Dir(1, 1))
        alpha_tilde_k = y_k + (1 - y_k) * alpha_k
            -> evidence on the wrong class is pulled toward 0,
               evidence on the correct class is set to the prior 1

    Returns
    -------
    float  (averaged over all samples; lower = better)
    """
    from scipy.special import gammaln, digamma

    b = np.asarray(b, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    truth = np.asarray(truth)

    # Clip u to avoid division by zero
    u_safe = np.maximum(u, eps)

    # SL -> Dirichlet
    S = 2.0 / u_safe                  # = alpha_b + alpha_d
    alpha_b = 1.0 + b * S
    alpha_d = 1.0 + d * S

    # Expected class probabilities
    p_b = alpha_b / S
    p_d = alpha_d / S

    # One-hot labels (y_b = 1 if benign)
    y_b = (truth == 1).astype(np.float64)
    y_d = 1.0 - y_b

    # MSE term (expected value + variance of the Dirichlet)
    var_factor = 1.0 / (S + 1.0)
    mse_term = (
        (y_b - p_b) ** 2 + p_b * (1.0 - p_b) * var_factor
        + (y_d - p_d) ** 2 + p_d * (1.0 - p_d) * var_factor
    )

    # KL term: alpha_tilde = y + (1 - y) * alpha
    alpha_tilde_b = y_b + (1.0 - y_b) * alpha_b
    alpha_tilde_d = y_d + (1.0 - y_d) * alpha_d
    S_tilde = alpha_tilde_b + alpha_tilde_d

    # KL(Dir(alpha_tilde) || Dir(1, 1))
    #   = log Gamma(S_tilde) - log Gamma(2)
    #     - sum log Gamma(alpha_tilde_k)
    #     + sum (alpha_tilde_k - 1) * (psi(alpha_tilde_k) - psi(S_tilde))
    # Note: log Gamma(2) = 0
    kl_term = (
        gammaln(S_tilde)
        - gammaln(alpha_tilde_b)
        - gammaln(alpha_tilde_d)
        + (alpha_tilde_b - 1.0) * (digamma(alpha_tilde_b) - digamma(S_tilde))
        + (alpha_tilde_d - 1.0) * (digamma(alpha_tilde_d) - digamma(S_tilde))
    )

    total = mse_term + kl_weight * kl_term
    return float(np.mean(total))


# ---------------------------------------------------------------------------
# Public API — analogous to sl_metrics from the B-spline project
# ---------------------------------------------------------------------------

def risk_coverage_curve(confidence, correct):
    """(coverages, risks) arrays for plotting."""
    n = len(confidence)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-np.asarray(confidence, dtype=np.float64), kind="mergesort")
    err_sorted = (1 - np.asarray(correct, dtype=np.int64))[order].astype(np.float64)
    risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    return coverage, risk


def evaluate_all(y_true, y_prob, y_pred, uncertainty):
    """
    Bundle of the standard evaluation metrics.

    Signature identical to sl_metrics.evaluate_all from the B-spline project.

    Parameters
    ----------
    y_true       : array (N,), 0=attacker, 1=benign
    y_prob       : array (N,), P(benign) — positive class for AUC = benign (1)
    y_pred       : array (N,), 0/1
    uncertainty  : array (N,), u in [0, 1]
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    u = np.asarray(uncertainty).reshape(-1)

    correct = (y_pred == y_true).astype(np.int64)
    confidence = 1.0 - u

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": _brier(y_prob, y_true),
        "ece": _ece(y_prob, y_true),
        "aurc": _aurc(confidence, correct),
        "misclass_auroc": _misclass_auroc(u, correct),
        "delta_u_median": _delta_u_median(u, correct),
        "risk_at_70": _risk_at_coverage(confidence, correct, 0.70),
        "risk_at_80": _risk_at_coverage(confidence, correct, 0.80),
        "risk_at_90": _risk_at_coverage(confidence, correct, 0.90),
        "mean_uncertainty": float(u.mean()) if len(u) else float("nan"),
    }


# ---------------------------------------------------------------------------
# SL-specific convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_all_from_opinion(b, d, u, truth, decision_thr=0.5):
    """
    Convenience wrapper: takes the SL convention (b, d, u, truth) and returns
    the same dict as evaluate_all, plus F1/confusion with attacker = positive
    class.

    truth:        0 = attacker, 1 = benign
    decision_thr: fixed operating point; (b + 0.5*u) >= decision_thr
                  -> y_pred = 1 (benign)
    """
    b = np.asarray(b, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64)

    p_benign = b + 0.5 * u
    y_pred   = (p_benign >= decision_thr).astype(np.int64)

    base = evaluate_all(truth, p_benign, y_pred, u)

    # F1 with attacker = positive class
    tp = int(np.sum((truth == 0) & (y_pred == 0)))
    fp = int(np.sum((truth == 1) & (y_pred == 0)))
    tn = int(np.sum((truth == 1) & (y_pred == 1)))
    fn = int(np.sum((truth == 0) & (y_pred == 1)))
    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    base.update({
        "n": int(len(truth)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_p_benign": float(p_benign.mean()) if len(p_benign) else float("nan"),
    })
    return base
