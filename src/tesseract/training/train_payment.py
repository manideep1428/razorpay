"""Training pipeline for the multi-class Payment Abuse Model."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tesseract.config import PAYMENT_LABELS, PaymentModelConfig
from tesseract.models.payment_model import PaymentAbuseModel
from tesseract.utils.metrics import (
    compute_classification_metrics,
    compute_multiclass_metrics,
    optimal_threshold_cost,
)
from tesseract.utils.model_io import save_tabular_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_payment_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    config: PaymentModelConfig | None = None,
    test_size: float = 0.2,
    calibrate: bool = True,
    categorical_features: list[str] | None = None,
) -> tuple[PaymentAbuseModel, dict[str, float]]:
    """Train, validate, calibrate, and persist the multi-class PaymentAbuseModel.

    Args:
        X: Feature matrix.
        y: Integer class labels (0=legit, 1=trial, 2=discount, 3=fraud).
        config: Configuration dataclass.
        test_size: Test holdout fraction.
        calibrate: Whether to apply probability calibration.
        categorical_features: Optional categorical column names.

    Returns:
        (trained_model, eval_metrics_dict)
    """
    cfg = config or PaymentModelConfig()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=cfg.random_state, stratify=y
        )
    except ValueError:
        # Fallback for very small datasets where a class may have < 2 members
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=cfg.random_state, stratify=None
        )

    logger.info(
        "Training PaymentAbuseModel on %d samples, testing on %d samples...",
        len(X_train),
        len(X_test),
    )

    model = PaymentAbuseModel(config=cfg)
    model.fit(
        X_train, y_train, calibrate=calibrate, categorical_features=categorical_features
    )

    risk = model.predict_risk(X_test)
    y_test_arr = np.asarray(y_test)

    # Multi-class metrics
    metrics = compute_multiclass_metrics(
        y_test_arr, risk["probabilities"], class_names=PAYMENT_LABELS
    )

    # Binary "abuse vs. legit" view driven by the payment risk score
    y_true_bin = (y_test_arr > 0).astype(int)
    risk_prob = risk["payment_risk_score"] / 100.0
    binary = compute_classification_metrics(y_true_bin, risk_prob)
    metrics["abuse_roc_auc"] = binary["roc_auc"]
    metrics["abuse_pr_auc"] = binary["pr_auc"]

    best_thresh, min_cost = optimal_threshold_cost(y_true_bin, risk_prob)
    metrics["optimal_threshold"] = best_thresh
    metrics["min_expected_cost"] = min_cost

    logger.info(
        "Evaluation -> Accuracy: %.4f | Macro-F1: %.4f | Macro ROC-AUC (OVR): %.4f | "
        "Abuse ROC-AUC: %.4f",
        metrics.get("accuracy", 0.0),
        metrics.get("macro_f1", 0.0),
        metrics.get("macro_roc_auc_ovr", 0.0),
        metrics.get("abuse_roc_auc", 0.0),
    )

    save_tabular_model(model, cfg.model_path)
    logger.info("Payment model saved to %s", cfg.model_path)

    return model, metrics


if __name__ == "__main__":
    # Synthetic self-test using the Tesseract payment schema.
    from tesseract.config import FeatureConfig
    from tesseract.utils.synthetic import synthesize_payment_dataset

    cfg = FeatureConfig()
    df = synthesize_payment_dataset(n=3000, seed=7)
    X_syn = df[cfg.payment_features].copy()
    for col in cfg.payment_categorical_features:
        X_syn[col] = X_syn[col].astype("category")
    y_syn = df["label"]
    train_payment_model(X_syn, y_syn, categorical_features=cfg.payment_categorical_features)
