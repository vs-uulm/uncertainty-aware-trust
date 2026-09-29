"""
Writes the five paper tables and the opinion box plot from evaluation results.

Expects one dict per method (as produced by
sweep_optuna.build_paper_outputs_final):
{
    "method_name":   str,
    "macro":         {macro_f1, macro_aurc, macro_misclass_auroc, macro_ece,
                      macro_brier, macro_delta_u},
    "overall_micro": {f1, aurc, misclass_auroc, ece, brier,
                      risk_at_70, risk_at_80, risk_at_90, risk_at_100,
                      delta_u_mean, delta_u_median},
    "per_attack":    {<attack>: {f1,
                                 u_correct_mean, u_correct_median,
                                 u_wrong_mean,   u_wrong_median,
                                 b_correct_mean, b_correct_median,
                                 b_wrong_mean,   b_wrong_median,
                                 d_correct_mean, d_correct_median,
                                 d_wrong_mean,   d_wrong_median, ...}},
    "boxplot_samples": {
        "u_correct", "u_wrong",
        "b_correct", "b_wrong",
        "d_correct", "d_wrong"
    },
}

Usage:
    from paper_outputs import write_all_tables_and_boxplot
    methods = [method_a_result, method_b_result]
    write_all_tables_and_boxplot(methods, attacks, out_dir=".")
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Table 1 — method × {F1, AURC, MAUROC, ECE, Brier, Δu} × {macro, micro}
# ============================================================================

def write_table_1_method_metrics(methods, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "method",
            "f1_macro", "f1_micro",
            "aurc_macro", "aurc_micro",
            "misclass_auroc_macro", "misclass_auroc_micro",
            "ece_macro", "ece_micro",
            "brier_macro", "brier_micro",
            "delta_u_macro", "delta_u_micro_median",
        ])
        for r in methods:
            m = r["macro"]
            om = r["overall_micro"]
            w.writerow([
                r["method_name"],
                _fmt(m.get("macro_f1")),                _fmt(om.get("f1")),
                _fmt(m.get("macro_aurc")),              _fmt(om.get("aurc")),
                _fmt(m.get("macro_misclass_auroc")),    _fmt(om.get("misclass_auroc")),
                _fmt(m.get("macro_ece")),               _fmt(om.get("ece")),
                _fmt(m.get("macro_brier")),             _fmt(om.get("brier")),
                _fmt(m.get("macro_delta_u")),           _fmt(om.get("delta_u_median")),
            ])


# ============================================================================
# Table 2 — attack × F1 per method
# ============================================================================

def write_table_2_per_attack_f1(methods, attacks, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["attack"] + [f"f1_{r['method_name']}" for r in methods]
        w.writerow(header)
        for atk in attacks:
            row = [atk]
            for r in methods:
                f1 = r["per_attack"].get(atk, {}).get("f1")
                row.append(_fmt(f1))
            w.writerow(row)
        # macro summary row
        w.writerow(["__macro__"] + [_fmt(r["macro"].get("macro_f1")) for r in methods])


# ============================================================================
# Table 3 — method × risk@coverage
# ============================================================================

def write_table_3_risk_coverage(methods, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "risk_at_100", "risk_at_90", "risk_at_80", "risk_at_70"])
        for r in methods:
            om = r["overall_micro"]
            w.writerow([
                r["method_name"],
                _fmt(om.get("risk_at_100")),
                _fmt(om.get("risk_at_90")),
                _fmt(om.get("risk_at_80")),
                _fmt(om.get("risk_at_70")),
            ])


# ============================================================================
# Table 4 — attack × {b,d,u}_{misclass,correct} per method (median or mean)
# ============================================================================

def write_table_4_uncertainty_per_attack(methods, attacks, path, statistic="median"):
    """
    statistic : "median" (default, robust) or "mean"

    Columns per method (in order):
        u_misclass, u_correct, b_misclass, b_correct, d_misclass, d_correct
    """
    assert statistic in ("median", "mean")
    keys = {
        "u_wrong":   f"u_wrong_{statistic}",
        "u_correct": f"u_correct_{statistic}",
        "b_wrong":   f"b_wrong_{statistic}",
        "b_correct": f"b_correct_{statistic}",
        "d_wrong":   f"d_wrong_{statistic}",
        "d_correct": f"d_correct_{statistic}",
    }

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# statistic: {statistic}"])
        header = ["attack"]
        for r in methods:
            mn = r["method_name"]
            header.extend([
                f"u_misclass_{mn}", f"u_correct_{mn}",
                f"b_misclass_{mn}", f"b_correct_{mn}",
                f"d_misclass_{mn}", f"d_correct_{mn}",
            ])
        w.writerow(header)
        for atk in attacks:
            row = [atk]
            for r in methods:
                pa = r["per_attack"].get(atk, {})
                row.extend([
                    _fmt(pa.get(keys["u_wrong"],   pa.get("u_wrong_mean"))),
                    _fmt(pa.get(keys["u_correct"], pa.get("u_correct_mean"))),
                    _fmt(pa.get(keys["b_wrong"],   pa.get("b_wrong_mean"))),
                    _fmt(pa.get(keys["b_correct"], pa.get("b_correct_mean"))),
                    _fmt(pa.get(keys["d_wrong"],   pa.get("d_wrong_mean"))),
                    _fmt(pa.get(keys["d_correct"], pa.get("d_correct_mean"))),
                ])
            w.writerow(row)


# ============================================================================
# Table 5 — attack × Δu (per-attack difference u_wrong - u_correct)
# ============================================================================

def write_table_5_delta_u_per_attack(methods, attacks, path, statistic="median"):
    """Per-Attack Δu = u_wrong_<stat> - u_correct_<stat>."""
    assert statistic in ("median", "mean")
    wrong_key   = f"u_wrong_{statistic}"
    correct_key = f"u_correct_{statistic}"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# statistic: {statistic}  (delta_u = u_wrong - u_correct)"])
        header = ["attack"] + [f"delta_u_{r['method_name']}" for r in methods]
        w.writerow(header)
        for atk in attacks:
            row = [atk]
            for r in methods:
                pa = r["per_attack"].get(atk, {})
                uw = pa.get(wrong_key,   pa.get("u_wrong_mean"))
                uc = pa.get(correct_key, pa.get("u_correct_mean"))
                if uw is None or uc is None or _is_nan(uw) or _is_nan(uc):
                    row.append("")
                else:
                    row.append(_fmt(uw - uc))
            w.writerow(row)
        # macro summary row
        w.writerow(["__macro__"] + [_fmt(r["macro"].get("macro_delta_u")) for r in methods])


# ============================================================================
# Box plot — per method: b, d, u distribution, misclassified vs correct (6 boxes)
# ============================================================================

def write_boxplot(methods, path):
    """
    Six boxes per method, side by side:
        u_wrong, u_correct, b_wrong, b_correct, d_wrong, d_correct

    Methods are separated by a gap. Missing components are left empty.
    """
    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=(max(10, 4.0 * n_methods + 2), 7))

    # colours: one (wrong, correct) pair each for u, b, d
    color_u_wrong, color_u_correct = "#D85A30", "#1D9E75"
    color_b_wrong, color_b_correct = "#A03E80", "#3C7DD9"
    color_d_wrong, color_d_correct = "#7A5400", "#888888"

    positions = []
    data = []
    box_colors = []
    xticks = []
    xticklabels = []

    GAP_BETWEEN_METHODS = 1.5
    BOX_SPACING = 1.0

    cursor = 1.0
    for r in methods:
        bs = r.get("boxplot_samples", {})
        u_w = np.asarray(bs.get("u_wrong", []),   dtype=np.float64)
        u_c = np.asarray(bs.get("u_correct", []), dtype=np.float64)
        b_w = np.asarray(bs.get("b_wrong", []),   dtype=np.float64)
        b_c = np.asarray(bs.get("b_correct", []), dtype=np.float64)
        d_w = np.asarray(bs.get("d_wrong", []),   dtype=np.float64)
        d_c = np.asarray(bs.get("d_correct", []), dtype=np.float64)

        method_box_data = [
            (u_w, color_u_wrong),
            (u_c, color_u_correct),
            (b_w, color_b_wrong),
            (b_c, color_b_correct),
            (d_w, color_d_wrong),
            (d_c, color_d_correct),
        ]
        method_positions = []
        for arr, col in method_box_data:
            if arr.size == 0:
                # no data: keep the slot so the layout stays aligned
                arr = np.array([np.nan])
            data.append(arr)
            box_colors.append(col)
            positions.append(cursor)
            method_positions.append(cursor)
            cursor += BOX_SPACING

        xticks.append(np.mean(method_positions))
        xticklabels.append(r["method_name"])
        cursor += GAP_BETWEEN_METHODS

    # drop NaN-only entries to avoid matplotlib warnings
    valid_data = []
    valid_positions = []
    valid_colors = []
    for arr, pos, col in zip(data, positions, box_colors):
        if np.isnan(arr).all():
            continue
        valid_data.append(arr)
        valid_positions.append(pos)
        valid_colors.append(col)

    bp = ax.boxplot(
        valid_data, positions=valid_positions, widths=0.7,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    for patch, color in zip(bp["boxes"], valid_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=10)
    ax.set_ylabel("SL opinion component value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Subjective Logic opinion (b, d, u) distribution: misclassified vs correctly classified")

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_u_wrong,   alpha=0.75, label="u — misclassified"),
        Patch(facecolor=color_u_correct, alpha=0.75, label="u — correct"),
        Patch(facecolor=color_b_wrong,   alpha=0.75, label="b — misclassified"),
        Patch(facecolor=color_b_correct, alpha=0.75, label="b — correct"),
        Patch(facecolor=color_d_wrong,   alpha=0.75, label="d — misclassified"),
        Patch(facecolor=color_d_correct, alpha=0.75, label="d — correct"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.95, ncol=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    pdf_path = path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Convenience
# ============================================================================

def _fmt(val, prec=4):
    if val is None or (isinstance(val, float) and (val != val)):  # None or NaN
        return ""
    if isinstance(val, (int, float)):
        return f"{val:.{prec}f}"
    return str(val)


def _is_nan(v):
    return isinstance(v, float) and v != v


def write_all_tables_and_boxplot(methods, attacks, out_dir=".", suffix=""):
    """
    Write all five tables and the box plot to out_dir.

    Parameters
    ----------
    methods : list of dict — see module docstring
    attacks : list of str  — attack order for tables 2, 4 and 5
    out_dir : str          — output directory
    suffix  : str          — optional file-name suffix (e.g. "balanced")
    """
    os.makedirs(out_dir, exist_ok=True)

    s = f"_{suffix}" if suffix else ""
    write_table_1_method_metrics(methods, os.path.join(out_dir, f"table_1_method_metrics{s}.csv"))
    write_table_2_per_attack_f1(methods, attacks, os.path.join(out_dir, f"table_2_per_attack_f1{s}.csv"))
    write_table_3_risk_coverage(methods, os.path.join(out_dir, f"table_3_risk_coverage{s}.csv"))
    write_table_4_uncertainty_per_attack(methods, attacks, os.path.join(out_dir, f"table_4_opinions_per_attack{s}.csv"))
    write_table_5_delta_u_per_attack(methods, attacks, os.path.join(out_dir, f"table_5_delta_u_per_attack{s}.csv"))
    write_boxplot(methods, os.path.join(out_dir, f"boxplot_opinions{s}.png"))

    print(f"Written to {out_dir}/:")
    print(f"  table_1_method_metrics{s}.csv")
    print(f"  table_2_per_attack_f1{s}.csv")
    print(f"  table_3_risk_coverage{s}.csv")
    print(f"  table_4_opinions_per_attack{s}.csv")
    print(f"  table_5_delta_u_per_attack{s}.csv")
    print(f"  boxplot_opinions{s}.png + .pdf")