"""FraudShield AI demo entrypoint."""

from trust_radar.config import PAYMENT_LABELS, SIGNUP_LABELS
from trust_radar.decisioning import (
    payment_decision,
    risk_level_from_score,
    risk_score_to_trust_score,
    signup_decision,
)


def main() -> None:
    print("FraudShield AI initialized successfully.")
    print(f"Signup labels : {SIGNUP_LABELS}")
    print(f"Payment labels: {PAYMENT_LABELS}")

    # Example signup outcome (risk score 9 -> high trust).
    signup_risk = 9
    print(
        f"\nSignup example  -> risk_score={signup_risk} "
        f"trust_score={risk_score_to_trust_score(signup_risk)} "
        f"risk_level={risk_level_from_score(signup_risk)} "
        f"decision={signup_decision(signup_risk)}"
    )

    # Example payment outcome on a trial plan (risk score 87 -> review).
    payment_risk = 87
    print(
        f"Payment example -> payment_risk_score={payment_risk} "
        f"risk_level={risk_level_from_score(payment_risk)} "
        f"decision={payment_decision(payment_risk, is_trial=True)}"
    )


if __name__ == "__main__":
    main()
