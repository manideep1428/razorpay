"""Tests for decisioning, score conversions, synthetic schema, and metrics."""

import numpy as np

from trust_radar.config import (
    PAYMENT_IDENTIFIER_COLUMNS,
    PAYMENT_LABELS,
    SIGNUP_IDENTIFIER_COLUMNS,
    FeatureConfig,
)
from trust_radar.decisioning import (
    payment_decision,
    probability_to_risk_score,
    requires_payment_scoring,
    risk_level_from_score,
    risk_score_to_trust_score,
    signup_decision,
)
from trust_radar.utils.metrics import compute_multiclass_metrics, multiclass_confusion
from trust_radar.utils.synthetic import (
    synthesize_payment_dataset,
    synthesize_signup_dataset,
)


def test_signup_decision_tiers_at_boundaries():
    assert signup_decision(0) == "ALLOW"
    assert signup_decision(40) == "ALLOW"
    assert signup_decision(41) == "ALLOW_FLAG_REVIEW"
    assert signup_decision(70) == "ALLOW_FLAG_REVIEW"
    assert signup_decision(71) == "ALLOW_HIGH_PRIORITY_REVIEW"
    assert signup_decision(94) == "ALLOW_HIGH_PRIORITY_REVIEW"
    assert signup_decision(95) == "TEMP_SUSPEND_MANUAL_REVIEW"
    assert signup_decision(100) == "TEMP_SUSPEND_MANUAL_REVIEW"


def test_risk_level_tiers():
    assert risk_level_from_score(40) == "low"
    assert risk_level_from_score(70) == "medium"
    assert risk_level_from_score(94) == "high"
    assert risk_level_from_score(95) == "critical"


def test_payment_decision_tiers_for_trial():
    assert payment_decision(40, is_trial=True) == "ALLOW"
    assert payment_decision(70, is_trial=True) == "ALLOW_FLAG_REVIEW"
    assert payment_decision(94, is_trial=True) == "ALLOW_HIGH_PRIORITY_REVIEW"
    assert payment_decision(95, is_trial=True) == "BLOCK"


def test_payment_full_price_is_always_allowed():
    # Even a maximum risk score is allowed when the plan is full price.
    assert payment_decision(100, plan_type="full_price") == "ALLOW"
    assert payment_decision(100, plan_type="standard") == "ALLOW"


def test_requires_payment_scoring_gating():
    assert requires_payment_scoring(is_trial=True) is True
    assert requires_payment_scoring(is_discounted=True) is True
    assert requires_payment_scoring(plan_type="trial") is True
    assert requires_payment_scoring(plan_type="discounted") is True
    assert requires_payment_scoring(plan_type="full_price") is False
    assert requires_payment_scoring(plan_type="standard") is False
    # No plan signal at all -> treated as full price (no scoring).
    assert requires_payment_scoring() is False


def test_score_conversions_match_spec_examples():
    # Spec example: risk_score 9 -> trust_score 91.
    assert risk_score_to_trust_score(9) == 91
    # Spec example: payment_risk_score 87.
    assert probability_to_risk_score(0.874) == 87
    # Clamping.
    assert probability_to_risk_score(1.5) == 100
    assert probability_to_risk_score(-0.2) == 0


def test_signup_synthetic_schema_is_complete():
    cfg = FeatureConfig()
    df = synthesize_signup_dataset(n=80, seed=1)
    expected = set(
        SIGNUP_IDENTIFIER_COLUMNS
        + cfg.signup_features
        + ["label", "admin_reviewed", "review_result"]
    )
    assert expected.issubset(set(df.columns))
    assert set(df["label"].unique()).issubset({0, 1})


def test_payment_synthetic_schema_and_labels():
    cfg = FeatureConfig()
    df = synthesize_payment_dataset(n=200, seed=1)
    expected = set(PAYMENT_IDENTIFIER_COLUMNS + cfg.payment_features + ["label"])
    assert expected.issubset(set(df.columns))
    assert set(df["label"].unique()).issubset(set(PAYMENT_LABELS.keys()))


def test_compute_multiclass_metrics_keys():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 200)
    logits = rng.random((200, 4))
    proba = logits / logits.sum(axis=1, keepdims=True)

    metrics = compute_multiclass_metrics(y_true, proba, class_names=PAYMENT_LABELS)
    for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_roc_auc_ovr"):
        assert key in metrics
    # Per-class keys use the label names.
    assert "f1_trial_abuse" in metrics

    cm = multiclass_confusion(y_true, np.argmax(proba, axis=1), n_classes=4)
    assert len(cm) == 4 and len(cm[0]) == 4
