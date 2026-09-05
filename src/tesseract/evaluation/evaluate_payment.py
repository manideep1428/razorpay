"""Evaluation pipeline for the multi-class Payment Abuse Model."""

from pathlib import Path

import numpy as np
import pandas as pd

from tesseract.config import PAYMENT_LABELS
from tesseract.models.payment_model import PaymentAbuseModel
from tesseract.utils.metrics import (
    compute_classification_metrics,
    compute_multiclass_metrics,
    optimal_threshold_cost,
)
from tesseract.utils.model_io import load_tabular_model


def evaluate_payment_model(
    model_or_path: PaymentAbuseModel | str | Path,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cost_fp: float = 5.0,
    cost_fn: float = 50.0,
) -> dict[str, float]:
    """Evaluate multi-class performance and the binary abuse-vs-legit cost view.

    Args:
        model_or_path: Trained PaymentAbuseModel instance or filepath.
        X: Feature matrix.
        y: Ground-truth integer class labels.
        cost_fp: Cost of a false positive ($).
        cost_fn: Cost of a false negative ($).

    Returns:
        Dictionary of performance metrics.
    """
    if isinstance(model_or_path, (str, Path)):
        model = load_tabular_model(model_or_path)
    else:
        model = model_or_path

    y_true = np.asarray(y)
    risk = model.predict_risk(X)

    metrics = compute_multiclass_metrics(
        y_true, risk["probabilities"], class_names=PAYMENT_LABELS
    )

    # Binary "abuse vs. legit" view from the 0-100 payment risk score.
    y_true_bin = (y_true > 0).astype(int)
    risk_prob = risk["payment_risk_score"] / 100.0
    binary = compute_classification_metrics(y_true_bin, risk_prob)
    metrics["abuse_roc_auc"] = binary["roc_auc"]
    metrics["abuse_pr_auc"] = binary["pr_auc"]

    best_thresh, min_cost = optimal_threshold_cost(
        y_true_bin, risk_prob, cost_fp=cost_fp, cost_fn=cost_fn
    )
    metrics["optimal_threshold"] = best_thresh
    metrics["min_cost"] = min_cost

    return metrics
