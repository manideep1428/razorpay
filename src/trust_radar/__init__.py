"""FraudShield AI -- signup trust scoring and payment abuse detection.

The importable package is ``trust_radar`` (the project's original name and
editable-install entry point); the product is branded **FraudShield AI**.

Only lightweight, dependency-free symbols (config + decisioning) are re-exported
here so ``import trust_radar`` stays fast. Import the models, training,
evaluation, and inference subpackages directly when you need PyTorch / LightGBM.
"""

from trust_radar.config import (
    PAYMENT_LABELS,
    SIGNUP_LABELS,
    DecisionConfig,
    FeatureConfig,
    PaymentModelConfig,
    SignupGNNConfig,
    SignupModelConfig,
)
from trust_radar.decisioning import (
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
