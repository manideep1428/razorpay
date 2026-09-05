"""Tests for the multi-class Payment Abuse Model: fit, scoring, gating, IO."""

import numpy as np

from tesseract.config import (
    PAYMENT_LABELS,
    FeatureConfig,
    PaymentModelConfig,
)
from tesseract.decisioning import PAYMENT_ACTIONS
from tesseract.inference.predict_payment import PaymentPredictor
from tesseract.models.payment_model import PaymentAbuseModel
from tesseract.utils.model_io import load_tabular_model, save_tabular_model
from tesseract.utils.synthetic import synthesize_payment_dataset

_FAST_CFG = PaymentModelConfig(n_estimators=60)


def _make_xy(n=600, seed=11):
    cfg = FeatureConfig()
    df = synthesize_payment_dataset(n=n, seed=seed)
    X = df[cfg.payment_features].copy()
    for col in cfg.payment_categorical_features:
        X[col] = X[col].astype("category")
    return df, X, df["label"], cfg


def test_payment_multiclass_fit_and_risk_score():
    _df, X, y, cfg = _make_xy()
    model = PaymentAbuseModel(config=_FAST_CFG)
    model.fit(X, y, calibrate=False, categorical_features=cfg.payment_categorical_features)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    assert proba.shape[1] == 4  # all four classes present in the synthetic data

    preds = model.predict(X)
    assert set(np.unique(preds)).issubset(set(PAYMENT_LABELS.keys()))

    scores = model.payment_risk_score(X)
    assert np.all(scores >= 0) and np.all(scores <= 100)

    risk = model.predict_risk(X)
    assert set(risk["abuse_type"]).issubset(set(PAYMENT_LABELS.values()))
    assert risk["probabilities"].shape[0] == len(X)


def test_payment_feature_importances_calibrated():
    _df, X, y, cfg = _make_xy()
    model = PaymentAbuseModel(config=PaymentModelConfig(n_estimators=40))
    model.fit(X, y, calibrate=True)

    importances = model.get_feature_importances()
    assert len(importances) == len(cfg.payment_features)
    assert "feature" in importances.columns and "importance" in importances.columns


def test_payment_predictor_plan_gating_and_actions():
    df, X, y, cfg = _make_xy()
    model = PaymentAbuseModel(config=_FAST_CFG)
    model.fit(X, y, calibrate=False, categorical_features=cfg.payment_categorical_features)
    predictor = PaymentPredictor(model_or_path=model)

    row = df.iloc[[0]]

    # Full-price plans are scored by the model and always allowed.
    full = predictor.score_transaction(row, plan_type="full_price")
    assert full["scored"] is True
    assert full["decision"] == "ALLOW"

    # Trial plans are scored and return a valid tiered decision.
    trial = predictor.score_transaction(row, plan_type="trial")
    assert trial["scored"] is True
    assert 0 <= trial["payment_risk_score"] <= 100
    assert trial["decision"] in PAYMENT_ACTIONS
    assert trial["abuse_type"] in PAYMENT_LABELS.values()


def test_payment_predictor_persistence_and_batch(tmp_path):
    df, X, y, cfg = _make_xy()
    model = PaymentAbuseModel(config=_FAST_CFG)
    model.fit(X, y, calibrate=False, categorical_features=cfg.payment_categorical_features)

    path = tmp_path / "payment_model.joblib"
    save_tabular_model(model, path)
    loaded = load_tabular_model(path)
    predictor = PaymentPredictor(model_or_path=loaded)

    batch = predictor.score_batch(df.head(25))
    for col in ("payment_risk_score", "risk_level", "abuse_type", "decision"):
        assert col in batch.columns
    assert set(batch["decision"]).issubset(set(PAYMENT_ACTIONS))
