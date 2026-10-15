"""Evaluation utilities for the CAISO project-realization classifier.

Role in the pipeline
--------------------
The notebook trains several models; this module scores them consistently so the comparison
is fair and the results are easy to read. It covers three kinds of evaluation:

1. **Ranking quality** (threshold-free): ROC-AUC and PR-AUC. These answer "does the model
   put projects that reached operation ahead of projects that withdrew?" PR-AUC is the more
   informative of the two when, as here, only about 15% of projects are positive.
2. **Probability quality**: Brier score, log loss, and calibration curves. These answer
   "when the model says 30%, do about 30% of such projects reach operation?" This matters
   because the agent reports the probability itself, not just a yes/no label.
3. **Decision quality** at a chosen probability threshold: accuracy, precision, recall,
   F1, balanced accuracy, MCC, confusion matrices, expected cost, and top-k precision/lift.

All functions take ``y_true`` (0/1 array) and ``y_prob`` (predicted probability of the
positive class, ``reached_operation = 1``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Metric tables
# ---------------------------------------------------------------------------


def probability_metrics(y_true, y_prob) -> dict:
    """Threshold-free metrics.

    * ``roc_auc``: chance that a random operational project scores above a random withdrawn
      one (0.5 = random, 1.0 = perfect).
    * ``pr_auc``: average precision across all thresholds. The no-skill value equals the
      positive rate, so it should always be read next to ``base_rate``.
    * ``brier``: mean squared error of the probabilities (lower is better). Predicting the
      base rate for everyone gives ``base_rate * (1 - base_rate)``.
    * ``log_loss``: penalizes confident wrong probabilities (lower is better).
    """
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "base_rate": float(y_true.mean()),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def classification_metrics(y_true, y_prob, threshold: float) -> dict:
    """Metrics for hard predictions made by flagging projects with ``y_prob >= threshold``.

    Precision: of projects flagged as likely to reach operation, the share that did.
    Recall: of projects that reached operation, the share that were flagged.
    MCC: a balanced correlation between prediction and outcome that stays honest under
    class imbalance (0 = no better than chance).
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged_share": float(y_pred.mean()),
    }


def top_k_metrics(y_true, y_prob, fractions=(0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Precision and lift when only the top-scoring fraction of projects is selected.

    This mirrors portfolio triage: "if we can only pursue the top 10% of projects, how many
    of them actually reach operation?" Lift is that precision divided by the base rate
    (lift 2.0 = twice as likely as picking projects at random).
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob))
    base = y_true.mean()
    rows = []
    for frac in fractions:
        k = max(1, int(round(frac * len(y_true))))
        hits = y_true[order[:k]]
        rows.append({
            "top_fraction": frac, "projects_selected": k,
            "operational_in_selection": int(hits.sum()),
            "precision_at_k": hits.mean(),
            "recall_at_k": hits.sum() / max(1, y_true.sum()),
            "lift": hits.mean() / base if base > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for any metric ``metric_fn(y_true, y_prob)``.

    The test set has only a few dozen operational projects, so single metric values are
    noisy. Resampling the test projects with replacement shows how much a metric could move
    with a slightly different sample. Resamples with no positives are skipped.

    Returns
    -------
    (point_estimate, lower, upper)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    lower, upper = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric_fn(y_true, y_prob), lower, upper


def bootstrap_table(y_true, y_prob, threshold: float, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap 95% confidence intervals for the headline metrics at one threshold."""
    metric_fns = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
        "precision": lambda y, p: precision_score(y, (p >= threshold).astype(int), zero_division=0),
        "recall": lambda y, p: recall_score(y, (p >= threshold).astype(int), zero_division=0),
        "f1": lambda y, p: f1_score(y, (p >= threshold).astype(int), zero_division=0),
    }
    rows = []
    for name, fn in metric_fns.items():
        point, lo, hi = bootstrap_ci(y_true, y_prob, fn, n_boot=n_boot, seed=seed)
        rows.append({"metric": name, "estimate": point, "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Thresholds and decision costs
# ---------------------------------------------------------------------------


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Evaluate every candidate threshold and report counts and rate metrics.

    Includes Youden's J (TPR - FPR), which is the formal version of "maximize true
    positives while minimizing false positives", plus F1 and the precision-weighted F0.5.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.81, 0.01), 2)
    positives, negatives = y_true.sum(), (1 - y_true).sum()
    rows = []
    for t in thresholds:
        pred = y_prob >= t
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        tn = int((~pred & (y_true == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "f0_5": 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0,
            "youden_j": recall - fpr,
            "flagged_share": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def expected_cost(sweep: pd.DataFrame, fp_cost: float, fn_cost: float = 1.0) -> pd.Series:
    """Total decision cost at each threshold: ``fp_cost * FP + fn_cost * FN``.

    A false positive is backing a project that later withdraws (sunk development or
    diligence spending). A false negative is passing on a project that would have reached
    operation (a missed opportunity). Only the *ratio* of the two costs matters.
    """
    return fp_cost * sweep["fp"] + fn_cost * sweep["fn"]


def choose_thresholds(sweep: pd.DataFrame, cost_ratios=(1, 2, 3, 5), recall_floor: float = 0.90) -> pd.DataFrame:
    """Pick thresholds under several decision criteria from a threshold sweep.

    Criteria
    --------
    youden_j : maximize TPR - FPR (the "max TP, min FP" objective).
    max_f1 : best balance of precision and recall.
    max_f0_5 : weights precision twice as much as recall (investor screening).
    cost_FP:FN=r : minimize expected cost when a false positive costs ``r`` times a false negative.
    recall_{recall_floor} : highest threshold that still keeps recall at or above the floor
        (projects below it rarely reach operation; used for the "lower likelihood" tier).
    """
    rows = []

    def pick(name, idx):
        row = sweep.loc[idx]
        rows.append({"criterion": name, **row.to_dict()})

    pick("youden_j", sweep["youden_j"].idxmax())
    pick("max_f1", sweep["f1"].idxmax())
    pick("max_f0_5", sweep["f0_5"].idxmax())
    for ratio in cost_ratios:
        pick(f"cost_FP:FN={ratio}:1", expected_cost(sweep, fp_cost=ratio).idxmin())
    meets_floor = sweep[sweep["recall"] >= recall_floor]
    if len(meets_floor):
        pick(f"recall_{int(recall_floor * 100)}", meets_floor["threshold"].idxmax())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_roc_pr(results: dict, title_suffix: str = "") -> plt.Figure:
    """ROC curve (false positive rate vs. true positive rate) and precision-recall curve.

    ``results`` maps a model name to ``(y_true, y_prob)``. The dashed lines show a model
    with no skill: the diagonal for ROC, and the positive rate for precision-recall.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    base_rate = None
    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, y_prob):.3f})")
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[1].plot(recall, precision, label=f"{name} (AP {average_precision_score(y_true, y_prob):.3f})")
        base_rate = np.mean(y_true)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="No skill")
    axes[0].set(title=f"ROC curve{title_suffix}", xlabel="False positive rate", ylabel="True positive rate (recall)")
    axes[1].axhline(base_rate, color="k", ls="--", lw=1, label=f"No skill ({base_rate:.3f})")
    axes[1].set(title=f"Precision-recall curve{title_suffix}", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_calibration(results: dict, n_bins: int = 8, title_suffix: str = "") -> plt.Figure:
    """Reliability diagram: predicted probability vs. observed operational share.

    Points on the diagonal mean the probabilities can be read literally. Quantile bins are
    used so each point contains a similar number of projects.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, (y_true, y_prob) in results.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    upper = max(0.6, max(np.max(p) for _, p in results.values()))
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Mean predicted probability",
           ylabel="Observed share reaching operation", title=f"Calibration{title_suffix}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(results: dict, thresholds: dict) -> plt.Figure:
    """Confusion matrix per model at that model's chosen threshold."""
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        y_true, y_prob = results[name]
        t = thresholds[name]
        cm = confusion_matrix(y_true, (np.asarray(y_prob) >= t).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                    color="white" if value > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred withdrawn", "Pred operational"])
        ax.set_yticks([0, 1], ["Withdrawn", "Operational"])
        ax.set_title(f"{name}\nthreshold {t:.2f}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_threshold_tradeoffs(sweep: pd.DataFrame, chosen: pd.DataFrame, title: str) -> plt.Figure:
    """Precision, recall, F1, F0.5, and Youden's J across thresholds, with chosen cutoffs marked."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                       ("f0_5", "F0.5"), ("youden_j", "Youden's J")]:
        ax.plot(sweep["threshold"], sweep[col], label=label)
    for _, row in chosen.iterrows():
        if row["criterion"] in {"youden_j", "max_f1", "max_f0_5"}:
            ax.axvline(row["threshold"], ls=":", color="gray")
            ax.text(row["threshold"], 1.01, row["criterion"], rotation=90, fontsize=8, va="bottom")
    ax.set(xlabel="Probability threshold", ylabel="Metric value", title=title, ylim=(0, 1.1))
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def plot_cost_curves(sweep: pd.DataFrame, cost_ratios=(1, 2, 3, 5), title: str = "") -> plt.Figure:
    """Expected decision cost by threshold for several false-positive:false-negative cost ratios.

    Each curve is normalized by its value when no project is flagged, so 1.0 means "no
    better than passing on everything". The minimum of each curve is the cost-optimal threshold.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for ratio in cost_ratios:
        cost = expected_cost(sweep, fp_cost=ratio)
        reference = sweep["fn"].max() + 1e-9
        ax.plot(sweep["threshold"], cost / reference, label=f"FP costs {ratio}x an FN")
        best = cost.idxmin()
        ax.scatter(sweep.loc[best, "threshold"], cost.loc[best] / reference, s=40)
    ax.set(xlabel="Probability threshold", ylabel="Relative expected cost", title=title)
    ax.legend()
    fig.tight_layout()
    return fig
