"""Evaluation metrics for FraudShield AI (binary signup + multi-class payment)."""


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary fraud/abuse classification metrics.

    Used for the Signup Trust Model and for the binary "abuse vs. legit" view of
    the Payment Abuse Model (label > 0 against ``payment_risk_score / 100``).

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted abuse probabilities in [0, 1].
        threshold: Decision threshold for discrete classification.

    Returns:
        Dictionary of metric names to values.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {}

    # Probability ranking metrics
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    # Binary decision metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Top-K precision (precision@decile-style operating points)
    for k in [1, 5, 10]:
        n_top = max(1, int(len(y_prob) * (k / 100.0)))
        top_indices = np.argsort(y_prob)[::-1][:n_top]
        metrics[f"precision_at_{k}pct"] = float(np.mean(y_true[top_indices]))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tp"] = float(tp)
    metrics["fp"] = float(fp)
    metrics["tn"] = float(tn)
    metrics["fn"] = float(fn)

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Compute multi-class metrics for the Payment Abuse Model.

    Args:
        y_true: Ground-truth integer class labels.
        y_proba: Class-probability matrix of shape (n_samples, n_classes),
            with columns ordered by ascending class label.
        class_names: Optional mapping of class index to a readable name, used to
            label the per-class metric keys.

    Returns:
        Dictionary containing overall accuracy, macro-averaged precision/recall/
        F1, macro one-vs-rest ROC-AUC, and per-class precision/recall/F1.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_classes = y_proba.shape[1]
    labels = list(range(n_classes))
    y_pred = np.argmax(y_proba, axis=1)

    metrics: dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    metrics["macro_precision"] = float(precision)
    metrics["macro_recall"] = float(recall)
    metrics["macro_f1"] = float(f1)

    try:
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=labels
            )
        )
    except ValueError:
        metrics["macro_roc_auc_ovr"] = float("nan")

    # Per-class precision / recall / F1
    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    for idx in labels:
        name = class_names.get(idx, str(idx)) if class_names else str(idx)
        metrics[f"precision_{name}"] = float(p_c[idx])
        metrics[f"recall_{name}"] = float(r_c[idx])
        metrics[f"f1_{name}"] = float(f_c[idx])

    return metrics


def multiclass_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[list[int]]:
    """Return the multi-class confusion matrix as a nested list of ints."""
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm.astype(int).tolist()


def optimal_threshold_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 5.0,
    cost_fn: float = 50.0,
    n_steps: int = 100,
) -> tuple[float, float]:
    """Find the decision threshold minimizing total financial loss.

    Balances false positives (customer friction / review cost) against false
    negatives (abuse loss).

    Returns:
        (best_threshold, min_cost)
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    costs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        _tn, fp, fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = (fp * cost_fp) + (fn * cost_fn)
        costs.append(cost)

    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
