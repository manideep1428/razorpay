"""Score-to-decision logic for Tesseract AI.

All functions here are pure and deterministic so they can be unit-tested in
isolation and reused across inference, evaluation, and the notebooks.

Scores are expressed on a **0-100 risk scale** where higher means riskier.
For the signup model we also expose ``trust_score = 100 - risk_score`` (the
value stored on ``user.trust_score``).

Business rules encoded here:

* Disposable email, VPN, proxy and Tor **never** trigger an automatic
  rejection. They only raise the model's risk score. Every decision below is a
  pure function of that final score.
* Full-price plans skip the Payment Abuse Model and are always allowed. Only
  trial and discounted plans are scored.
"""


from tesseract.config import PAYMENT_LABELS, DecisionConfig

# Canonical action vocabularies -------------------------------------------------
SIGNUP_ACTIONS = (
    "ALLOW",
    "ALLOW_FLAG_REVIEW",
    "ALLOW_HIGH_PRIORITY_REVIEW",
    "TEMP_SUSPEND_MANUAL_REVIEW",
)
PAYMENT_ACTIONS = (
    "ALLOW",
    "ALLOW_FLAG_REVIEW",
    "ALLOW_HIGH_PRIORITY_REVIEW",
    "BLOCK",
)
RISK_LEVELS = ("low", "medium", "high", "critical")

_DEFAULT_CONFIG = DecisionConfig()


def _clamp_unit(value: float) -> float:
    """Clamp a probability-like value into the closed unit interval [0, 1]."""
    return min(max(float(value), 0.0), 1.0)


def probability_to_risk_score(probability: float) -> int:
    """Convert a 0-1 abuse probability into an integer 0-100 risk score."""
    return round(_clamp_unit(probability) * 100)


def risk_score_to_trust_score(risk_score: float) -> int:
    """Return the complementary trust score (100 - risk) as an int in [0, 100]."""
    score = round(min(max(float(risk_score), 0.0), 100.0))
    return 100 - score


def risk_level_from_score(
    risk_score: float, config: DecisionConfig | None = None
) -> str:
    """Map a 0-100 risk score to one of ``RISK_LEVELS``."""
    cfg = config or _DEFAULT_CONFIG
    if risk_score <= cfg.low_max:
        return "low"
    if risk_score <= cfg.medium_max:
        return "medium"
    if risk_score <= cfg.high_max:
        return "high"
    return "critical"


def signup_decision(
    risk_score: float, config: DecisionConfig | None = None
) -> str:
    """Apply the 4-tier signup decision logic to a 0-100 risk score.

    * 0-40   -> ``ALLOW``
    * 41-70  -> ``ALLOW_FLAG_REVIEW``
    * 71-94  -> ``ALLOW_HIGH_PRIORITY_REVIEW``
    * 95-100 -> ``TEMP_SUSPEND_MANUAL_REVIEW``
    """
    cfg = config or _DEFAULT_CONFIG
    if risk_score <= cfg.low_max:
        return "ALLOW"
    if risk_score <= cfg.medium_max:
        return "ALLOW_FLAG_REVIEW"
    if risk_score <= cfg.high_max:
        return "ALLOW_HIGH_PRIORITY_REVIEW"
    return "TEMP_SUSPEND_MANUAL_REVIEW"


def requires_payment_scoring(
    plan_type: str | None = None,
    is_trial: bool | None = None,
    is_discounted: bool | None = None,
    config: DecisionConfig | None = None,
) -> bool:
    """Every payment transaction is evaluated inside the model without skipping."""
    return True


def payment_decision(
    payment_risk_score: float,
    plan_type: str | None = None,
    is_trial: bool | None = None,
    is_discounted: bool | None = None,
    config: DecisionConfig | None = None,
) -> str:
    """Apply the payment decision logic to the 0-100 payment risk score.

    * Standard / Full-price plans are ALWAYS allowed (capturing legitimate customer revenue).
    * For trial abuse: when a card is reused across multiple accounts and risk >= 50, BLOCK trial.
    * 0-40   -> ``ALLOW``
    * 41-70  -> ``ALLOW_FLAG_REVIEW``
    * 71-94  -> ``ALLOW_HIGH_PRIORITY_REVIEW``
    * 95-100 -> ``BLOCK``
    """
    cfg = config or _DEFAULT_CONFIG

    # Business Rule: Standard / Full-Price plans are ALWAYS ALLOWED
    plan_str = str(plan_type).strip().lower() if plan_type is not None else ""
    if plan_str in cfg.full_price_plan_types or plan_str in ("paid", "standard", "enterprise", "full_price"):
        return "ALLOW"
    if not is_trial and not is_discounted and plan_str != "trial":
        return "ALLOW"

    # Business Rule: Trial Abuse Prevention (Block free trial if risk >= 50)
    if is_trial and payment_risk_score >= 50:
        return "BLOCK"

    if payment_risk_score <= cfg.low_max:
        return "ALLOW"
    if payment_risk_score <= cfg.medium_max:
        return "ALLOW_FLAG_REVIEW"
    if payment_risk_score <= cfg.high_max:
        return "ALLOW_HIGH_PRIORITY_REVIEW"
    return "BLOCK"


def payment_block_target(
    is_trial: bool | None = None, is_discounted: bool | None = None
) -> str | None:
    """Return the concrete block target for the top payment tier."""
    if is_trial:
        return "BLOCK_TRIAL"
    if is_discounted:
        return "BLOCK_DISCOUNT"
    return "BLOCK_PAYMENT"


def abuse_type_name(class_index: int) -> str:
    """Return the human-readable payment abuse type for a class index."""
    return PAYMENT_LABELS.get(int(class_index), "unknown")
