"""Tesseract AI -- signup trust scoring and payment abuse detection.

The importable package is ``tesseract`` (the project's original name and
editable-install entry point); the product is branded **Tesseract AI**.

Only lightweight, dependency-free symbols (config + decisioning) are re-exported
here so ``import tesseract`` stays fast. Import the models, training,
evaluation, and inference subpackages directly when you need PyTorch / LightGBM.
"""

from tesseract.config import (
    PAYMENT_LABELS,
    SIGNUP_LABELS,
    DecisionConfig,
    FeatureConfig,
    PaymentModelConfig,
    SignupGNNConfig,
    SignupModelConfig,
)
from tesseract.decisioning import (
    abuse_type_name,
    payment_decision,
    probability_to_risk_score,
    requires_payment_scoring,
    risk_level_from_score,
    risk_score_to_trust_score,
    signup_decision,
)

__version__ = "0.2.0"

__all__ = [
    "PAYMENT_LABELS",
    "SIGNUP_LABELS",
    "DecisionConfig",
    "FeatureConfig",
    "PaymentModelConfig",
    "SignupGNNConfig",
    "SignupModelConfig",
    "abuse_type_name",
    "payment_decision",
    "probability_to_risk_score",
    "requires_payment_scoring",
    "risk_level_from_score",
    "risk_score_to_trust_score",
    "signup_decision",
]
