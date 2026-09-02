"""Inference engine for the multi-class Payment Abuse Model."""

from pathlib import Path

import numpy as np
import pandas as pd

from trust_radar.config import PAYMENT_CATEGORICAL_FEATURES, PAYMENT_LABELS
from trust_radar.decisioning import (
    abuse_type_name,
    payment_block_target,
    payment_decision,
    requires_payment_scoring,
    risk_level_from_score,
)
from trust_radar.models.payment_model import PaymentAbuseModel
from trust_radar.utils.model_io import load_tabular_model


class PaymentPredictor:
    """Real-time scoring service for incoming payment transactions.

    Full-price plans are allowed without scoring; trial and discounted plans are
    run through the model, which emits a 0-100 ``payment_risk_score``, the most
    likely ``abuse_type``, a ``risk_level``, and the 4-tier ``decision``.
    """

    def __init__(
        self,
        model_or_path: PaymentAbuseModel | str | Path,
    ) -> None:
        if isinstance(model_or_path, (str, Path)):
            self.model = load_tabular_model(model_or_path)
        else:
            self.model = model_or_path

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the model's feature columns (dropping identifiers) and coerce
        categorical dtypes so LightGBM receives exactly what it was trained on.
        """
        df = df.copy()
        names = getattr(self.model, "feature_names", None)
        if names:
            for col in names:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[names]
        for col in PAYMENT_CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    @staticmethod
    def _plan_signal(source: dict, key: str):
        val = source.get(key)
        if val is None:
            return None
        if key in ("is_trial", "is_discounted"):
            return bool(val)
        return val

    def score_transaction(
        self,
        transaction_features: dict[str, float] | pd.DataFrame,
        plan_type: str | None = None,
        is_trial: bool | None = None,
        is_discounted: bool | None = None,
    ) -> dict[str, float | int | str | None]:
        """Score a single transaction and return the payment decision.

        Plan signals (``plan_type`` / ``is_trial`` / ``is_discounted``) may be
        passed explicitly or read from ``transaction_features``.
        """
        if isinstance(transaction_features, dict):
            row = dict(transaction_features)
            df = pd.DataFrame([transaction_features])
        else:
            df = transaction_features.iloc[[0]] if len(transaction_features) else transaction_features
            row = df.iloc[0].to_dict() if len(df) else {}

        # Resolve plan signals (explicit args win over row values).
        plan_type = plan_type if plan_type is not None else self._plan_signal(row, "plan_type")
        is_trial = is_trial if is_trial is not None else self._plan_signal(row, "is_trial")
        is_discounted = (
            is_discounted
            if is_discounted is not None
            else self._plan_signal(row, "is_discounted")
        )

        scored = requires_payment_scoring(
            plan_type=plan_type, is_trial=is_trial, is_discounted=is_discounted
        )

        if not scored:
            # Full-price plans skip the model entirely.
            return {
                "scored": False,
                "payment_risk_score": 0,
                "risk_level": "low",
                "abuse_type": "legit",
                "predicted_class": 0,
                "decision": "ALLOW",
                "block_target": None,
            }

        risk = self.model.predict_risk(self._prepare_features(df))
        score = int(risk["payment_risk_score"][0])
        predicted_class = int(risk["predicted_class"][0])
        decision = payment_decision(
            score, plan_type=plan_type, is_trial=is_trial, is_discounted=is_discounted
        )
        block_target = (
            payment_block_target(is_trial=is_trial, is_discounted=is_discounted)
            if decision == "BLOCK"
            else None
        )

        return {
            "scored": True,
            "payment_risk_score": score,
            "risk_level": risk_level_from_score(score),
            "abuse_type": abuse_type_name(predicted_class),
            "predicted_class": predicted_class,
            "decision": decision,
            "block_target": block_target,
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch-score a transaction DataFrame.

        Reads ``plan_type`` / ``is_trial`` / ``is_discounted`` from the frame
        (when present) to apply plan-type gating per row.
        """
        risk = self.model.predict_risk(self._prepare_features(df))
        scores = risk["payment_risk_score"]
        predicted_class = risk["predicted_class"]

        plan_type = df["plan_type"] if "plan_type" in df.columns else [None] * len(df)
        is_trial = df["is_trial"] if "is_trial" in df.columns else [None] * len(df)
        is_discounted = (
            df["is_discounted"] if "is_discounted" in df.columns else [None] * len(df)
        )

        decisions, risk_levels, abuse_types = [], [], []
        for i in range(len(df)):
            pt = plan_type[i] if not isinstance(plan_type, pd.Series) else plan_type.iloc[i]
            it = is_trial[i] if not isinstance(is_trial, pd.Series) else is_trial.iloc[i]
            idsc = (
                is_discounted[i]
                if not isinstance(is_discounted, pd.Series)
                else is_discounted.iloc[i]
            )
            decisions.append(
                payment_decision(
                    scores[i],
                    plan_type=pt,
                    is_trial=None if it is None else bool(it),
                    is_discounted=None if idsc is None else bool(idsc),
                )
            )
            risk_levels.append(risk_level_from_score(scores[i]))
            abuse_types.append(abuse_type_name(int(predicted_class[i])))

        result = df.copy()
        result["payment_risk_score"] = scores
        result["risk_level"] = risk_levels
        result["abuse_type"] = abuse_types
        result["decision"] = decisions
        return result


# Expose the class label map for convenience in downstream code / notebooks.
PAYMENT_CLASS_LABELS = PAYMENT_LABELS
