"""
Metrics for evaluating Subjective Logic opinions in V2X misbehavior detection.

Label convention:
    y_true: 0 = attacker, 1 = benign
    p:       probability of benign  (= b_benign + base_rate * u)
    y_pred:  0/1 binary prediction (1 = benign)
    u:       fused uncertainty mass

All metrics return Python floats (or dicts thereof) for easy CSV logging.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------- Proper scoring rules -----------------------------

def brier_score(y_true, y_prob):
    """Brier = MSE between predicted probability and label.
    Lower is better. Strictly proper scoring rule — rewards calibration AND sharpness.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def nll(y_true, y_prob, eps=1e-12):
    """Negative log-likelihood (binary cross-entropy on the test set).
    Lower is better. Strictly proper, but more sensitive to overconfident errors than Brier.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_prob = np.clip(np.asarray(y_prob).reshape(-1), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


# ----------------------------- Calibration -----------------------------

def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Expected calibration error (Guo et al., 2017).

    Samples are binned by MAX-CLASS CONFIDENCE, accuracy is taken at the 0.5
    cutoff, and the result is the weighted L1 difference |confidence - accuracy|.
    Lower is better.

    This implementation is intentionally identical to the one used for the
    other detection approaches, so ECE values are comparable across methods.
    """
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true).reshape(-1).astype(np.int64)
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


def reliability_diagram(y_true, y_prob, n_bins=15):
    """Returns (bin_centers, bin_acc, bin_conf, bin_count) for plotting."""
    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        centers.append(0.5 * (lo + hi))
        if mask.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
        else:
            accs.append(float(y_true[mask].mean()))
            confs.append(float(y_prob[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)


# ----------------------------- Threshold-free classification -----------------------------

def pr_auc_attacker(y_true, y_prob):
    """Area under Precision-Recall curve, with attacker (y=0) as the positive class.
    Score = 1 - p (higher = more likely attacker). Robust to class imbalance.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    y_attacker = (y_true == 0).astype(int)
    if y_attacker.sum() == 0 or y_attacker.sum() == len(y_attacker):
        return float("nan")
    return float(average_precision_score(y_attacker, 1.0 - y_prob))


def roc_auc_attacker(y_true, y_prob):
    """ROC-AUC. Threshold-free — independent of t_star choice."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    y_attacker = (y_true == 0).astype(int)
    if y_attacker.sum() == 0 or y_attacker.sum() == len(y_attacker):
        return float("nan")
    return float(roc_auc_score(y_attacker, 1.0 - y_prob))


def matthews_corrcoef(y_true, y_pred, positive_class=0):
    """MCC — balanced over the full confusion matrix, robust to imbalance."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    yt = (y_true == positive_class).astype(int)
    yp = (y_pred == positive_class).astype(int)
    TP = int(((yt == 1) & (yp == 1)).sum())
    TN = int(((yt == 0) & (yp == 0)).sum())
    FP = int(((yt == 0) & (yp == 1)).sum())
    FN = int(((yt == 1) & (yp == 0)).sum())
    denom = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    if denom == 0:
        return 0.0
    return float((TP * TN - FP * FN) / denom)


# ----------------------------- Uncertainty-aware metrics (the SL value-add) -----------------------------

def misclassification_auroc(y_true, y_pred, uncertainty):
    """AUROC for: does u rank misclassified samples higher than correct ones?
    Higher is better. Answers: 'is my uncertainty informative?'
    A value near 0.5 means uncertainty is useless. >0.8 means strongly informative.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    u = np.asarray(uncertainty).reshape(-1).astype(float)
    is_wrong = (y_true != y_pred).astype(int)
    if is_wrong.sum() == 0 or is_wrong.sum() == len(is_wrong):
        return float("nan")
    return float(roc_auc_score(is_wrong, u))


def risk_coverage_curve(y_true, y_pred, uncertainty):
    """Risk-Coverage curve.
    Sort samples by ascending u, then compute cumulative error rate.
    Returns (coverage_array, risk_array, AURC).
    AURC: lower is better — model knows when it doesn't know.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    u = np.asarray(uncertainty).reshape(-1).astype(float)
    n = len(y_true)
    order = np.argsort(u)
    errors = (y_true[order] != y_pred[order]).astype(float)
    cum_errors = np.cumsum(errors)
    coverage = np.arange(1, n + 1) / n
    risk = cum_errors / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, coverage))
    return coverage, risk, aurc


def selective_f1(y_true, y_pred, uncertainty,
                 coverages=(0.7, 0.8, 0.9, 0.95), positive_class=0):
    """F1 (attacker = positive class) at given coverage levels.
    For each coverage c, drop the (1-c) most uncertain samples, then compute F1.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    u = np.asarray(uncertainty).reshape(-1).astype(float)
    order = np.argsort(u)
    yt_s = y_true[order]
    yp_s = y_pred[order]
    n = len(y_true)

    out = {}
    for c in coverages:
        k = int(c * n)
        if k == 0:
            out[c] = float("nan")
            continue
        yt = yt_s[:k]
        yp = yp_s[:k]
        TP = int(((yt == positive_class) & (yp == positive_class)).sum())
        FP = int(((yt != positive_class) & (yp == positive_class)).sum())
        FN = int(((yt == positive_class) & (yp != positive_class)).sum())
        denom = 2 * TP + FP + FN
        out[c] = 0.0 if denom == 0 else (2 * TP) / denom
    return out


def delta_u_mean(y_true, y_pred, uncertainty):
    """
    mean(u | misclassified) - mean(u | correct), computed over samples.

    The MEAN is used instead of the median: the median has a 50 % breakdown
    point, so a minority of misclassifications with low u ("confident-wrong")
    stays invisible to it, although that is exactly the population of interest.
    The mean weights it proportionally to its share. Since u lies in [0, 1],
    outlier sensitivity is not an issue: a single sample moves the mean by at
    most 1/n.
    """
    u = np.asarray(uncertainty).reshape(-1).astype(float)
    correct = (np.asarray(y_pred).reshape(-1) == np.asarray(y_true).reshape(-1))
    u_w, u_c = u[~correct], u[correct]
    if len(u_w) == 0 or len(u_c) == 0:
        return float("nan")
    return float(u_w.mean() - u_c.mean())


def delta_u_median(y_true, y_pred, uncertainty):
    """median(u | misclassified) - median(u | correct). Descriptive only.

    Its gap to delta_u_mean reflects the skew of the misclassified
    distribution, i.e. the size of the confident-wrong tail.
    """
    u = np.asarray(uncertainty).reshape(-1).astype(float)
    correct = (np.asarray(y_pred).reshape(-1) == np.asarray(y_true).reshape(-1))
    u_w, u_c = u[~correct], u[correct]
    if len(u_w) == 0 or len(u_c) == 0:
        return float("nan")
    return float(np.median(u_w) - np.median(u_c))


def delta_u_cohens_d(y_true, y_pred, uncertainty):
    """
    Effect size: (mean_wrong - mean_correct) / sigma_pooled.

    delta_u is scale-dependent: a method with a wide u distribution gets larger
    absolute gaps without separating better. Cohen's d normalizes this away
    and is therefore comparable across methods.
    """
    u = np.asarray(uncertainty).reshape(-1).astype(float)
    correct = (np.asarray(y_pred).reshape(-1) == np.asarray(y_true).reshape(-1))
    u_w, u_c = u[~correct], u[correct]
    n_w, n_c = len(u_w), len(u_c)
    if n_w < 2 or n_c < 2:
        return float("nan")
    sp_sq = ((n_w - 1) * u_w.var(ddof=1)
             + (n_c - 1) * u_c.var(ddof=1)) / (n_w + n_c - 2)
    if sp_sq <= 0:
        return float("nan")
    return float((u_w.mean() - u_c.mean()) / np.sqrt(sp_sq))


def uncertainty_split_stats(y_true, y_pred, uncertainty):
    """Mean uncertainty for correct vs. incorrect predictions.
    A useful quick sanity check. Should differ noticeably."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    u = np.asarray(uncertainty).reshape(-1).astype(float)
    correct = (y_true == y_pred)
    return {
        "u_mean_correct":   float(u[correct].mean())   if correct.sum()       else float("nan"),
        "u_mean_incorrect": float(u[~correct].mean())  if (~correct).sum()    else float("nan"),
        "u_std_correct":    float(u[correct].std())    if correct.sum()       else float("nan"),
        "u_std_incorrect":  float(u[~correct].std())   if (~correct).sum()    else float("nan"),
    }


# ----------------------------- Convenience aggregator -----------------------------

def evaluate_all(y_true, y_prob, y_pred, uncertainty):
    """Compute the full panel. Returns a flat dict suitable for CSV logging."""
    out = {
        "brier":          brier_score(y_true, y_prob),
        "nll":            nll(y_true, y_prob),
        "ece":            expected_calibration_error(y_true, y_prob),
        "pr_auc":         pr_auc_attacker(y_true, y_prob),
        "roc_auc":        roc_auc_attacker(y_true, y_prob),
        "mcc":            matthews_corrcoef(y_true, y_pred),
        "misclass_auroc": misclassification_auroc(y_true, y_pred, uncertainty),
        "delta_u_mean":     delta_u_mean(y_true, y_pred, uncertainty),
        "delta_u_median":   delta_u_median(y_true, y_pred, uncertainty),
        "delta_u_cohens_d": delta_u_cohens_d(y_true, y_pred, uncertainty),
    }
    _, _, aurc = risk_coverage_curve(y_true, y_pred, uncertainty)
    out["aurc"] = aurc
    sel = selective_f1(y_true, y_pred, uncertainty)
    out.update({f"sel_f1@{int(c*100)}": v for c, v in sel.items()})
    out.update(uncertainty_split_stats(y_true, y_pred, uncertainty))
    return out