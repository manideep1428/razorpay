"""Multi-class Payment Abuse Model for FraudShield AI.

A calibrated LightGBM classifier that detects trial abuse, discount / coupon
abuse, shared-card abuse, promo farming, and payment fraud. It consumes the
upstream ``trust_score`` alongside payment, card, device, IP, velocity, and
graph features.

Classes (see :data:`trust_radar.config.PAYMENT_LABELS`)::

    0 = legit
    1 = trial_abuse
    2 = discount_abuse
    3 = payment_fraud

The model exposes a ``payment_risk_score`` on a 0-100 scale, defined as the
probability that a transaction is *not* legit.
"""


import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from trust_radar.config import PaymentModelConfig
from trust_radar.decisioning import abuse_type_name


class PaymentAbuseModel:
    """Gradient-boosted multi-class model for payment abuse detection."""

    def __init__(self, config: PaymentModelConfig | None = None) -> None:
        self.config = config or PaymentModelConfig()
        self.model: lgb.LGBMClassifier | CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_calibrated: bool = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_set: list[tuple] | None = None,
        calibrate: bool = True,
        categorical_features: list[str] | None = None,
    ) -> "PaymentAbuseModel":
        """Train the multi-class LightGBM payment abuse model.

        Args:
            X: Feature matrix or DataFrame. Categorical columns should use the
                pandas ``category`` dtype so LightGBM handles them natively.
            y: Integer class labels (0=legit, 1=trial, 2=discount, 3=fraud).
            eval_set: Optional list of (X_val, y_val) tuples for early stopping
                (only used when ``calibrate=False``).
            calibrate: Whether to wrap the estimator in probability calibration.
            categorical_features: Optional explicit categorical column names.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        base_lgbm = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            verbosity=-1,
        )

        if calibrate:
            self.model = CalibratedClassifierCV(
                estimator=base_lgbm,
                method="isotonic",
                cv=3,
            )
            self.model.fit(X, y)
            self.is_calibrated = True
        else:
            base_lgbm.fit(
                X,
                y,
                eval_set=eval_set,
                categorical_feature=categorical_features or "auto",
            )
            self.model = base_lgbm
            self.is_calibrated = False

        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the full class-probability matrix of shape (n_samples, n_classes)."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the predicted class label (argmax) for each sample."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _legit_probability(self, proba: np.ndarray) -> np.ndarray:
        """Extract the probability mass on the legit class (label 0)."""
        if self.classes_.size and 0 in self.classes_:
            legit_idx = int(np.where(self.classes_ == 0)[0][0])
            return proba[:, legit_idx]
        return np.zeros(proba.shape[0])

    def payment_risk_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return the 0-100 payment risk score = P(not legit) * 100."""
        proba = self.predict_proba(X)
        risk = np.clip(1.0 - self._legit_probability(proba), 0.0, 1.0)
        return np.rint(risk * 100).astype(int)

    def predict_risk(
        self, X: pd.DataFrame | np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute risk scores, predicted class, and abuse-type names.

        Returns a dict with:
            * ``payment_risk_score`` -- integer 0-100 (higher = riskier).
            * ``predicted_class``    -- integer class label.
            * ``abuse_type``         -- class name (e.g. ``"trial_abuse"``).
            * ``probabilities``      -- full class-probability matrix.
        """
        proba = self.predict_proba(X)
        risk = np.clip(1.0 - self._legit_probability(proba), 0.0, 1.0)
        predicted_class = self.classes_[np.argmax(proba, axis=1)]
        abuse_types = np.array([abuse_type_name(c) for c in predicted_class])
        return {
            "payment_risk_score": np.rint(risk * 100).astype(int),
            "predicted_class": predicted_class,
            "abuse_type": abuse_types,
            "probabilities": proba,
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature-importance ranking (averaged across CV folds if calibrated)."""
        if self.model is None:
            raise ValueError("Model has not been fitted.")

        if self.is_calibrated:
            importances = np.mean(
                [
                    self._estimator_importances(cal)
                    for cal in self.model.calibrated_classifiers_
                ],
                axis=0,
            )
        else:
            importances = self.model.feature_importances_

        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def _estimator_importances(calibrated_clf) -> np.ndarray:
        """Extract LightGBM importances from a calibrated sub-classifier."""
        for attr in ("estimator", "base_estimator"):
            est = getattr(calibrated_clf, attr, None)
            if est is not None and hasattr(est, "feature_importances_"):
                return est.feature_importances_
        raise AttributeError("Could not locate base estimator feature importances.")
