"""Global configuration for FraudShield AI models, feature schema, and decisioning.

FraudShield AI ships two models:

* **Signup Trust Model** -- a GraphSAGE / LightGBM classifier that emits a
  ``trust_score`` (0-100) and ``risk_level`` for every signup.
* **Payment Abuse Model** -- a multi-class LightGBM classifier that emits a
  ``payment_risk_score`` (0-100) plus the most likely abuse type.

The importable package remains ``trust_radar`` (the project's original name and
editable-install entry point); only the product branding is FraudShield AI.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------
# Signup Trust Model: binary abuse detection.
SIGNUP_LABELS: dict[int, str] = {
    0: "legit_user",
    1: "abuse_user",
}

# Payment Abuse Model: multi-class abuse typing.
PAYMENT_LABELS: dict[int, str] = {
    0: "legit",
    1: "trial_abuse",
    2: "discount_abuse",
    3: "payment_fraud",
}
PAYMENT_NUM_CLASSES = len(PAYMENT_LABELS)


# ---------------------------------------------------------------------------
# Signup feature schema (Streamlined, production-focused fraud detection)
# ---------------------------------------------------------------------------
SIGNUP_FEATURE_GROUPS: dict[str, list[str]] = {
    "identity": [
        "signup_method",
        "oauth_provider",
        "account_type",
    ],
    "email": [
        "plus_alias_used",
    ],
    "device_fingerprint": [
        "screen_width",
        "screen_height",
        "cpu_cores",
        "device_memory_gb",
        "platform",
        "timezone",
        "private_mode_detected",
        "device_type",
        "browser_family",
        "os_family",
    ],
    "device_reputation": [
        "device_age_days",
        "accounts_per_device_7d",
        "banned_accounts_per_device",
        "shared_device_count",
        "device_trust_score",
    ],
    "ip_intelligence": [
        "ip_country",
        "vpn_flag",
        "proxy_flag",
        "tor_flag",
        "datacenter_ip_flag",
        "accounts_per_ip_7d",
    ],
    "behavior": [
        "session_duration_seconds",
    ],
}

# Raw identifiers that key the records and graph.
SIGNUP_IDENTIFIER_COLUMNS: list[str] = [
    "user_id",
    "created_at",
    "email_address",
    "device_fingerprint",
    "ip_address",
    "user_agent",
]

# Categorical (string) signup features requiring encoding.
SIGNUP_CATEGORICAL_FEATURES: list[str] = [
    "signup_method",
    "oauth_provider",
    "account_type",
    "platform",
    "timezone",
    "device_type",
    "browser_family",
    "os_family",
    "ip_country",
]

# Admin / label bookkeeping columns.
SIGNUP_LABEL_COLUMNS: list[str] = ["label", "admin_reviewed", "review_result"]


# ---------------------------------------------------------------------------
# Payment feature schema (grouped exactly as in the FraudShield AI spec)
# ---------------------------------------------------------------------------
PAYMENT_FEATURE_GROUPS: dict[str, list[str]] = {
    "payment_identity": [
        "trust_score",  # upstream signal from the Signup Trust Model
        "account_age_days",
        "days_since_signup",
        "plan_type",
        "payment_type",
    ],
    "payment": [
        "amount",
        "currency",
        "country",
        "payment_provider",
        "is_trial",
        "is_discounted",
        "discount_percentage",
        "coupon_used",
        "coupon_usage_count",
        "promotion_used",
    ],
    "card": [
        "card_country",
        "card_brand",
        "card_type",
        "card_prepaid",
        "card_debit",
        "card_credit",
        "card_bin_risk_score",
        "card_age_days",
    ],
    # Most important section for shared-card / card-testing abuse.
    "card_reputation": [
        "users_per_card_1d",
        "users_per_card_7d",
        "users_per_card_30d",
        "organizations_per_card_30d",
        "trials_per_card_30d",
        "discounts_per_card_30d",
        "successful_payments_per_card",
        "failed_payments_per_card",
        "refunds_per_card",
        "chargebacks_per_card",
        "abuse_rate_per_card",
        "card_trust_score",
    ],
    "device_reputation": [
        "users_per_device_30d",
        "organizations_per_device_30d",
        "trials_per_device_30d",
        "discounts_per_device_30d",
        "successful_payments_per_device",
        "failed_payments_per_device",
        "chargebacks_per_device",
        "abuse_rate_per_device",
    ],
    "ip_reputation": [
        "vpn_flag",
        "proxy_flag",
        "tor_flag",
        "hosting_provider_flag",
        "accounts_per_ip_30d",
        "payments_per_ip_30d",
        "trials_per_ip_30d",
        "discounts_per_ip_30d",
        "chargebacks_per_ip",
        "abuse_rate_per_ip",
    ],
    # Extremely important section for burst / farming detection.
    "velocity": [
        "payments_last_5m",
        "payments_last_1h",
        "payments_last_24h",
        "trials_last_24h",
        "discounts_last_24h",
        "cards_seen_last_24h",
        "devices_seen_last_24h",
        "organizations_created_last_24h",
    ],
    "account_history": [
        "successful_payments_count",
        "failed_payments_count",
        "refund_count",
        "chargeback_count",
        "previous_trial_count",
        "previous_discount_count",
        "previous_bans",
    ],
    "relationship": [
        "shared_card_count",
        "shared_device_count",
        "shared_ip_count",
        "linked_accounts_count",
        "linked_organizations_count",
        "high_risk_neighbor_count",
        "abusive_neighbors",
    ],
    "graph": [
        "cluster_size",
        "community_risk_score",
        "avg_neighbor_risk",
        "max_neighbor_risk",
        "distance_to_known_abuser",
        "card_centrality",
        "device_centrality",
        "ip_centrality",
    ],
}

PAYMENT_IDENTIFIER_COLUMNS: list[str] = [
    "transaction_id",
    "user_id",
    "organization_id",
    "card_fingerprint",
    "signup_device_fingerprint",
    "payment_device_fingerprint",
    "ip_address",
]

PAYMENT_CATEGORICAL_FEATURES: list[str] = [
    "plan_type",
    "payment_type",
    "currency",
    "country",
    "payment_provider",
    "card_country",
    "card_brand",
    "card_type",
]

PAYMENT_LABEL_COLUMNS: list[str] = ["label"]


def _flatten_unique(groups: dict[str, list[str]]) -> list[str]:
    """Flatten grouped feature lists into a de-duplicated, order-preserving list."""
    seen: dict[str, None] = {}
    for cols in groups.values():
        for col in cols:
            if col not in seen:
                seen[col] = None
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Decision thresholds (shared by the signup and payment 4-tier logic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionConfig:
    """Risk-tier boundaries applied to a 0-100 *risk* score (higher = riskier).

    Tiers (identical shape for signup and payment):
        0-40   -> tier 1 (allow)
        41-70  -> tier 2 (allow + flag review)
        71-94  -> tier 3 (allow + high priority review)
        95-100 -> tier 4 (temp suspend / block)

    Note: disposable email, VPN, proxy and Tor are *risk-increasing signals
    only*. They must never trigger an automatic rejection on their own -- the
    decision is always a pure function of the final score.
    """

    low_max: int = 40
    medium_max: int = 70
    high_max: int = 94
    # 95-100 is the top (critical) tier.

    # Plan types that are billed at full price and therefore skip the
    # Payment Abuse Model entirely (they are always allowed).
    full_price_plan_types: tuple = ("full_price", "standard", "paid", "enterprise")


# ---------------------------------------------------------------------------
# Model hyper-parameter configs
# ---------------------------------------------------------------------------
@dataclass
class SignupGNNConfig:
    """Hyper-parameters for the GraphSAGE Signup Trust Model."""

    in_channels: int = 32
    hidden_channels: int = 64
    out_channels: int = 1
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 256
    pos_weight: float = 5.0
    checkpoint_path: Path = field(
        default_factory=lambda: ARTIFACTS_DIR / "signup_graphsage.pt"
    )


@dataclass
class SignupModelConfig:
    """Config for the tabular (LightGBM) variant of the Signup Trust Model."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 7
    num_leaves: int = 48
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    class_weight: str = "balanced"
    random_state: int = 42
    model_path: Path = field(
        default_factory=lambda: ARTIFACTS_DIR / "signup_trust_lgbm.joblib"
    )


@dataclass
class PaymentModelConfig:
    """Hyper-parameters for the multi-class Payment Abuse Model (LightGBM).

    Classes follow :data:`PAYMENT_LABELS`:
    ``0=legit, 1=trial_abuse, 2=discount_abuse, 3=payment_fraud``.
    """

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 7
    num_leaves: int = 48
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    class_weight: str = "balanced"
    num_classes: int = PAYMENT_NUM_CLASSES
    random_state: int = 42
    model_path: Path = field(
        default_factory=lambda: ARTIFACTS_DIR / "payment_abuse_lgbm.joblib"
    )


@dataclass
class FeatureConfig:
    """Resolved feature schema for both FraudShield AI models.

    The grouped dictionaries (:data:`SIGNUP_FEATURE_GROUPS` /
    :data:`PAYMENT_FEATURE_GROUPS`) are the single source of truth; the flat
    ``*_features`` lists (model inputs) are derived from them at construction.
    """

    signup_feature_groups: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in SIGNUP_FEATURE_GROUPS.items()}
    )
    payment_feature_groups: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in PAYMENT_FEATURE_GROUPS.items()}
    )

    signup_categorical_features: list[str] = field(
        default_factory=lambda: list(SIGNUP_CATEGORICAL_FEATURES)
    )
    payment_categorical_features: list[str] = field(
        default_factory=lambda: list(PAYMENT_CATEGORICAL_FEATURES)
    )

    signup_identifier_columns: list[str] = field(
        default_factory=lambda: list(SIGNUP_IDENTIFIER_COLUMNS)
    )
    payment_identifier_columns: list[str] = field(
        default_factory=lambda: list(PAYMENT_IDENTIFIER_COLUMNS)
    )

    # Populated in __post_init__.
    signup_features: list[str] = field(default_factory=list)
    payment_features: list[str] = field(default_factory=list)
    signup_numeric_features: list[str] = field(default_factory=list)
    payment_numeric_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.signup_features = _flatten_unique(self.signup_feature_groups)
        self.payment_features = _flatten_unique(self.payment_feature_groups)
        self.signup_numeric_features = [
            c for c in self.signup_features if c not in self.signup_categorical_features
        ]
        self.payment_numeric_features = [
            c
            for c in self.payment_features
            if c not in self.payment_categorical_features
        ]
