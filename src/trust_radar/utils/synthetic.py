"""Schema-faithful synthetic data generators for FraudShield AI.

These produce pandas DataFrames whose columns match the feature schema declared
in :mod:`trust_radar.config`, so notebooks and tests can exercise the full
pipeline without a real dataset. A handful of "driver" features are correlated
with the label so the models have genuine signal to learn; the long tail of
features is filled with plausible noise.
"""


import numpy as np
import pandas as pd

from trust_radar.config import (
    PAYMENT_IDENTIFIER_COLUMNS,
    SIGNUP_IDENTIFIER_COLUMNS,
    FeatureConfig,
)

# ---------------------------------------------------------------------------
# Categorical value pools (kept small for realistic, low-cardinality columns)
# ---------------------------------------------------------------------------
_CATEGORICAL_POOLS: dict[str, list[str]] = {
    # signup
    "signup_method": ["email", "google", "apple", "github", "facebook"],
    "oauth_provider": ["none", "google", "apple", "github", "facebook"],
    "account_type": ["personal", "business", "developer"],
    "phone_country": ["US", "IN", "GB", "DE", "NG", "RU", "none"],
    "browser_family": ["Chrome", "Firefox", "Safari", "Edge", "HeadlessChrome"],
    "language": ["en", "en-US", "ru", "hi", "de", "zh"],
    "device_type": ["desktop", "mobile", "tablet"],
    "os_family": ["Windows", "macOS", "Linux", "Android", "iOS"],
    "os_version": ["10", "11", "13", "14", "15"],
    "platform": ["Win32", "MacIntel", "Linux x86_64", "iPhone", "Android"],
    "timezone": [
        "America/New_York",
        "Asia/Kolkata",
        "Europe/London",
        "Europe/Berlin",
        "Europe/Moscow",
    ],
    "ip_country": ["US", "IN", "GB", "DE", "RU", "NG"],
    "ip_region": ["CA", "NY", "MH", "LDN", "BE", "MOW"],
    "ip_city": ["San Francisco", "New York", "Mumbai", "London", "Berlin", "Moscow"],
    "asn": ["AS15169", "AS55836", "AS2856", "AS3320", "AS14061", "AS37963"],
    "isp": ["Comcast", "Jio", "BT", "Deutsche Telekom", "DigitalOcean", "OVH"],
    # payment
    "plan_type": ["trial", "discounted", "full_price", "standard"],
    "payment_type": ["card", "paypal", "apple_pay", "google_pay"],
    "currency": ["USD", "EUR", "GBP", "INR"],
    "country": ["US", "IN", "GB", "DE", "RU"],
    "payment_provider": ["stripe", "paypal", "adyen", "braintree"],
    "card_country": ["US", "IN", "GB", "DE", "RU"],
    "card_brand": ["visa", "mastercard", "amex", "discover"],
    "card_type": ["credit", "debit", "prepaid"],
}

# Substrings that indicate a 0/1 binary flag feature.
_FLAG_HINTS = (
    "flag",
    "detected",
    "enabled",
    "_used",
    "prepaid",
    "debit",
    "credit",
    "free_provider",
    "is_disposable_email",
    "is_trial",
    "is_discounted",
    "promotion_used",
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Realistic-looking value generators (emails, hashes, IPs, user agents, phones)
# ---------------------------------------------------------------------------
_FREE_EMAIL_DOMAINS = [
    "gmail.com",
    "outlook.com",
    "yahoo.com",
    "hotmail.com",
    "icloud.com",
    "proton.me",
    "zoho.com",
    "aol.com",
    "gmx.com",
    "mail.com",
]
_BUSINESS_EMAIL_DOMAINS = [
    "google.com",
    "microsoft.com",
    "amazon.com",
    "apple.com",
    "meta.com",
    "ibm.com",
    "oracle.com",
    "salesforce.com",
]
_DISPOSABLE_EMAIL_DOMAINS = [
    "mailinator.com", "10minutemail.com", "guerrillamail.com",
    "tempmail.com", "yopmail.com", "trashmail.com", "throwawaymail.com",
]
_ROLE_PREFIXES = ["admin", "support", "info", "sales", "contact", "noreply", "billing"]
_FIRST_NAMES = [
    "james", "mary", "robert", "linda", "michael", "patricia", "david", "jennifer",
    "arjun", "priya", "wei", "mei", "olga", "ivan", "carlos", "sofia", "amara",
    "kwame", "yuki", "hiro", "fatima", "omar", "elena", "noah", "ava",
]
_LAST_NAMES = [
    "smith", "johnson", "williams", "brown", "sharma", "patel", "zhang", "li",
    "kim", "park", "ivanov", "garcia", "rodriguez", "okafor", "diallo",
    "tanaka", "suzuki", "khan", "hassan", "novak", "clark", "lewis",
]

# OS build strings keyed by os_family, used to assemble a real user-agent.
_OS_UA_TOKENS = {
    "Windows": ["Windows NT 10.0; Win64; x64", "Windows NT 6.1; Win64; x64"],
    "macOS": ["Macintosh; Intel Mac OS X 10_15_7", "Macintosh; Intel Mac OS X 13_4_1"],
    "Linux": ["X11; Linux x86_64", "X11; Ubuntu; Linux x86_64"],
    "Android": ["Linux; Android 13; Pixel 7", "Linux; Android 12; SM-G991U"],
    "iOS": ["iPhone; CPU iPhone OS 16_5 like Mac OS X", "iPad; CPU OS 16_5 like Mac OS X"],
}

_PHONE_CC = {
    "US": "1", "IN": "91", "GB": "44", "DE": "49", "NG": "234", "RU": "7", "none": None,
}


def _random_hex(rng: np.random.Generator, n: int, n_chars: int) -> np.ndarray:
    """Vectorized random lowercase-hex strings of length ``n_chars`` (like an md5/sha hash)."""
    n_bytes = (n_chars + 1) // 2
    raw = rng.integers(0, 256, size=(n, n_bytes), dtype=np.uint8)
    return np.array([bytes(row).hex()[:n_chars] for row in raw])


def _random_ipv4(rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorized dotted-quad IPv4 addresses, avoiding 0/255 edge octets."""
    octets = rng.integers(1, 255, size=(n, 4))
    return np.array([".".join(map(str, row)) for row in octets])


def _make_email_username(rng: np.random.Generator, n: int, entropy_hint: np.ndarray) -> list[str]:
    """Build a realistic local-part: name-based for low entropy, random-looking for high entropy.

    ``entropy_hint`` (roughly the existing 0-6.5 email_username_entropy scale)
    biases toward bot-like random strings as it increases, matching the
    correlation the rest of the schema already encodes for abuse.
    """
    usernames = []
    bot_like = rng.random(n) < _sigmoid(entropy_hint - 4.0)
    for i in range(n):
        if bot_like[i]:
            length = rng.integers(8, 14)
            chars = rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=length)
            usernames.append("".join(chars))
        else:
            first = rng.choice(_FIRST_NAMES)
            last = rng.choice(_LAST_NAMES)
            sep = rng.choice([".", "_", ""])
            suffix = "" if rng.random() < 0.5 else str(rng.integers(1, 999))
            usernames.append(f"{first}{sep}{last}{suffix}")
    return usernames


def _make_emails(
    rng: np.random.Generator,
    n: int,
    is_disposable: np.ndarray,
    free_provider: np.ndarray,
    role_based: np.ndarray,
    plus_alias: np.ndarray,
    entropy_hint: np.ndarray,
) -> np.ndarray:
    """Assemble full, realistic-looking email addresses consistent with the
    existing disposable/free-provider/role-based/plus-alias flag columns."""
    usernames = _make_email_username(rng, n, entropy_hint)
    domains = np.empty(n, dtype=object)
    disposable_mask = is_disposable == 1
    free_mask = (~disposable_mask) & (free_provider == 1)
    business_mask = ~(disposable_mask | free_mask)

    domains[disposable_mask] = rng.choice(_DISPOSABLE_EMAIL_DOMAINS, size=disposable_mask.sum())
    domains[free_mask] = rng.choice(_FREE_EMAIL_DOMAINS, size=free_mask.sum())
    domains[business_mask] = rng.choice(_BUSINESS_EMAIL_DOMAINS, size=business_mask.sum())

    emails = []
    for i in range(n):
        local = _ROLE_PREFIXES[rng.integers(0, len(_ROLE_PREFIXES))] if role_based[i] else usernames[i]
        if plus_alias[i]:
            local = f"{local}+{rng.integers(1, 9999)}"
        emails.append(f"{local}@{domains[i]}")
    return np.array(emails)


def _make_user_agents(
    rng: np.random.Generator,
    n: int,
    browser_family: np.ndarray,
    browser_major: np.ndarray,
    os_family: np.ndarray,
) -> np.ndarray:
    """Assemble a real-looking UA string consistent with the browser/OS columns."""
    uas = []
    for i in range(n):
        os_token = rng.choice(_OS_UA_TOKENS.get(os_family[i], _OS_UA_TOKENS["Windows"]))
        family = browser_family[i]
        major = int(browser_major[i])
        if family == "Safari":
            uas.append(
                f"Mozilla/5.0 ({os_token}) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                f"Version/{major}.0 Safari/605.1.15"
            )
        elif family == "Firefox":
            uas.append(f"Mozilla/5.0 ({os_token}; rv:{major}.0) Gecko/20100101 Firefox/{major}.0")
        elif family == "HeadlessChrome":
            uas.append(
                f"Mozilla/5.0 ({os_token}) AppleWebKit/537.36 (KHTML, like Gecko) "
                f"HeadlessChrome/{major}.0.0.0 Safari/537.36"
            )
        else:  # Chrome / Edge share the Chromium token shape
            engine = "Edg" if family == "Edge" else "Chrome"
            uas.append(
                f"Mozilla/5.0 ({os_token}) AppleWebKit/537.36 (KHTML, like Gecko) "
                f"Chrome/{major}.0.0.0 {'Safari/537.36' if engine == 'Chrome' else f'Safari/537.36 Edg/{major}.0.0.0'}"
            )
    return np.array(uas)


def _make_phone_numbers(rng: np.random.Generator, n: int, phone_country: np.ndarray, has_phone: np.ndarray) -> np.ndarray:
    """Build E.164-style phone numbers matching ``phone_country``; blank when no phone."""
    numbers = np.full(n, "", dtype=object)
    for i in range(n):
        if not has_phone[i]:
            continue
        cc = _PHONE_CC.get(phone_country[i])
        if not cc:
            continue
        subscriber = "".join(str(d) for d in rng.integers(0, 10, size=9))
        numbers[i] = f"+{cc}{subscriber}"
    return numbers



def _fill_generic(col: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a plausible noise column based on naming heuristics."""
    if col in _CATEGORICAL_POOLS:
        return rng.choice(_CATEGORICAL_POOLS[col], size=n)
    if any(hint in col for hint in _FLAG_HINTS):
        return rng.binomial(1, 0.2, n)
    if "abuse_rate" in col or col.endswith("_rate"):
        return np.round(rng.beta(1.3, 18.0, n), 4)
    if col.endswith("_score"):
        return np.round(rng.uniform(30, 95, n), 2)
    if "centrality" in col or col == "pagerank_score":
        return np.round(rng.beta(1.5, 12.0, n), 4)
    if "days" in col:
        return np.round(rng.uniform(0, 1500, n), 1)
    if col in ("signup_hour",):
        return rng.integers(0, 24, n)
    if col in ("signup_day_of_week",):
        return rng.integers(0, 7, n)
    if "utc_offset" in col:
        return rng.integers(-12, 14, n)
    count_hints = (
        "count",
        "per_",
        "accounts",
        "payments",
        "trials",
        "discounts",
        "seen",
        "views",
        "movements",
        "keystroke",
        "click",
        "neighbors",
        "cluster",
        "component",
        "refund",
        "chargeback",
        "distance",
        "languages_count",
        "font_count",
        "touch_points",
        "paste_events",
    )
    if any(hint in col for hint in count_hints):
        return rng.poisson(2.0, n)
    if col in ("screen_width",):
        return rng.choice([1280, 1366, 1440, 1920, 2560], size=n)
    if col in ("screen_height",):
        return rng.choice([720, 768, 900, 1080, 1440], size=n)
    if col in ("cpu_cores",):
        return rng.choice([2, 4, 6, 8, 12, 16], size=n)
    if col in ("device_memory_gb",):
        return rng.choice([2, 4, 8, 16, 32], size=n)
    if col in ("color_depth",):
        return rng.choice([16, 24, 30, 32], size=n)
    if col in ("pixel_ratio",):
        return np.round(rng.choice([1.0, 1.5, 2.0, 3.0], size=n), 2)
    if col in ("browser_major_version",):
        return rng.integers(80, 130, n)
    if col in ("browser_minor_version",):
        return rng.integers(0, 20, n)
    if col in ("user_agent_length",):
        return rng.integers(90, 240, n)
    if col in ("discount_percentage",):
        return np.round(rng.uniform(0, 90, n), 1)
    if col in ("amount",):
        return np.round(rng.exponential(80.0, n), 2)
    if "seconds" in col:
        return np.round(rng.exponential(60.0, n), 1)
    if col in ("scroll_depth",):
        return np.round(rng.uniform(0, 1, n), 3)
    # Fallback: mild positive continuous.
    return np.round(rng.uniform(0, 10, n), 3)


def synthesize_signup_dataset(
    n: int = 2000, seed: int | None = 42
) -> pd.DataFrame:
    """Generate a synthetic signup dataset with the streamlined FraudShield schema.

    Returns a DataFrame containing identifier columns, every signup feature, and
    the label columns (``label``, ``admin_reviewed``, ``review_result``).
    ``label`` is 1 for abusive signups and 0 for legit users.
    """
    rng = np.random.default_rng(seed)
    cfg = FeatureConfig()
    data: dict[str, np.ndarray] = {}

    # Latent abuse factor -> ~12% abusers.
    z = rng.normal(0.0, 1.0, n)
    label = (z > 1.18).astype(int)
    z_pos = np.maximum(z, 0.0)

    # Driver features correlated with the latent factor -----------------------
    data["accounts_per_device_7d"] = rng.poisson(np.exp(0.1 + 0.7 * z_pos))
    data["accounts_per_ip_7d"] = rng.poisson(np.exp(0.2 + 0.8 * z_pos))
    data["banned_accounts_per_device"] = rng.poisson(np.exp(-0.5 + 0.8 * z_pos))
    data["shared_device_count"] = rng.poisson(np.exp(0.7 * z_pos))
    data["device_trust_score"] = np.round(
        np.clip(72 - 20 * z + rng.normal(0, 6, n), 0, 100), 2
    )
    data["device_age_days"] = np.round(
        np.clip(450 - 120 * z + rng.normal(0, 50, n), 1, 2000), 1
    )
    data["plus_alias_used"] = rng.binomial(1, _sigmoid(-1.8 + 0.8 * z))
    data["vpn_flag"] = rng.binomial(1, _sigmoid(-1.0 + 0.7 * z))
    data["proxy_flag"] = rng.binomial(1, _sigmoid(-1.6 + 0.7 * z))
    data["tor_flag"] = rng.binomial(1, _sigmoid(-2.8 + 0.9 * z))
    data["datacenter_ip_flag"] = rng.binomial(1, _sigmoid(-1.5 + 0.8 * z))
    data["private_mode_detected"] = rng.binomial(1, _sigmoid(-1.2 + 0.6 * z))
    data["session_duration_seconds"] = np.round(
        np.clip(45.0 - 25.0 * _sigmoid(z) + rng.exponential(15, n), 0.5, 300.0), 2
    )

    # Hardware profile pools
    data["screen_width"] = rng.choice([1920, 1440, 1366, 1280, 2560], size=n)
    data["screen_height"] = rng.choice([1080, 900, 768, 720, 1440], size=n)
    data["cpu_cores"] = rng.choice([2, 4, 6, 8, 12, 16], size=n)
    data["device_memory_gb"] = rng.choice([4, 8, 16, 32], size=n)

    # Identifier columns -------------------------------------------------------
    data["user_id"] = np.array([f"u_{i:07d}" for i in range(n)])
    data["created_at"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 240 * 24 * 3600, n), unit="s"
    )

    # Device fingerprint as a real-looking hex hash.
    device_pool = _random_hex(rng, max(2, n // 3), 32)
    data["device_fingerprint"] = device_pool[rng.integers(0, len(device_pool), n)]

    # Real-looking IP address, drawn from a pool so shared-IP signals persist.
    ip_pool = _random_ipv4(rng, max(2, n // 3))
    data["ip_address"] = ip_pool[rng.integers(0, len(ip_pool), n)]

    # Fill every remaining schema column with categorical/generic pools
    all_cols = cfg.signup_features + [
        c for c in SIGNUP_IDENTIFIER_COLUMNS if c not in data
    ]
    for col in all_cols:
        if col not in data:
            data[col] = _fill_generic(col, n, rng)

    # Realistic email address
    is_disp = rng.binomial(1, 0.05, n)
    is_free = rng.binomial(1, 0.85, n)
    role_based = rng.binomial(1, 0.02, n)
    data["email_address"] = _make_emails(
        rng, n,
        is_disposable=is_disp,
        free_provider=is_free,
        role_based=role_based,
        plus_alias=data["plus_alias_used"],
        entropy_hint=np.full(n, 3.0),
    )

    # Real-looking User-Agent string
    browser_major = rng.choice([110, 115, 120, 124, 128], size=n)
    data["user_agent"] = _make_user_agents(
        rng, n,
        browser_family=data["browser_family"],
        browser_major=browser_major,
        os_family=data["os_family"],
    )

    # Label bookkeeping columns ------------------------------------------------
    data["label"] = label
    data["admin_reviewed"] = np.where(
        label == 1, rng.binomial(1, 0.6, n), rng.binomial(1, 0.05, n)
    )
    data["review_result"] = np.where(
        data["admin_reviewed"] == 1,
        np.where(label == 1, "abuse_confirmed", "cleared"),
        "not_reviewed",
    )

    df = pd.DataFrame(data)
    ordered = (
        SIGNUP_IDENTIFIER_COLUMNS
        + cfg.signup_features
        + ["label", "admin_reviewed", "review_result"]
    )
    return df[ordered]


def synthesize_signup_edges(
    num_nodes: int, avg_degree: float = 4.0, seed: int | None = 42
) -> pd.DataFrame:
    """Generate a synthetic shared-device / shared-IP edge list for the graph."""
    rng = np.random.default_rng(seed)
    num_edges = int(num_nodes * avg_degree)
    src = rng.integers(0, num_nodes, num_edges)
    dst = rng.integers(0, num_nodes, num_edges)
    mask = src != dst
    return pd.DataFrame({"src": src[mask], "dst": dst[mask]})


def synthesize_payment_dataset(
    n: int = 4000, seed: int | None = 42
) -> pd.DataFrame:
    """Generate a synthetic payment dataset with the full FraudShield schema.

    ``label`` follows :data:`trust_radar.config.PAYMENT_LABELS`:
    ``0=legit, 1=trial_abuse, 2=discount_abuse, 3=payment_fraud``.
    """
    rng = np.random.default_rng(seed)
    cfg = FeatureConfig()
    data: dict[str, np.ndarray] = {}

    # Independent latent drivers for each abuse type.
    z_trial = rng.normal(0.0, 1.0, n)
    z_disc = rng.normal(0.0, 1.0, n)
    z_fraud = rng.normal(0.0, 1.0, n)
    tp, dp, fp = np.maximum(z_trial, 0), np.maximum(z_disc, 0), np.maximum(z_fraud, 0)

    is_trial = rng.binomial(1, _sigmoid(-0.3 + 0.9 * z_trial))
    is_discounted = np.where(
        is_trial == 1, 0, rng.binomial(1, _sigmoid(-0.5 + 0.9 * z_disc))
    )
    data["is_trial"] = is_trial
    data["is_discounted"] = is_discounted

    # Trial-abuse drivers.
    data["trials_per_card_30d"] = rng.poisson(np.exp(0.2 + 0.9 * tp))
    data["trials_last_24h"] = rng.poisson(np.exp(0.1 + 0.8 * tp))
    data["trials_per_ip_30d"] = rng.poisson(np.exp(0.2 + 0.7 * tp))
    data["users_per_card_30d"] = rng.poisson(np.exp(0.3 + 0.8 * np.maximum(tp, fp)))

    # Discount-abuse drivers.
    data["discounts_per_card_30d"] = rng.poisson(np.exp(0.2 + 0.9 * dp))
    data["discounts_last_24h"] = rng.poisson(np.exp(0.1 + 0.8 * dp))
    data["coupon_usage_count"] = rng.poisson(np.exp(0.2 + 0.9 * dp))
    data["promotion_used"] = np.where(
        is_discounted == 1, 1, rng.binomial(1, _sigmoid(-0.5 + 0.8 * z_disc))
    )
    data["coupon_used"] = np.where(
        is_discounted == 1, rng.binomial(1, 0.85, n), rng.binomial(1, 0.05, n)
    )
    data["discount_percentage"] = np.where(
        is_discounted == 1, np.round(rng.uniform(10, 90, n), 1), 0.0
    )

    # Payment-fraud drivers.
    data["chargebacks_per_card"] = rng.poisson(np.exp(-1.0 + 1.0 * fp))
    data["failed_payments_per_card"] = rng.poisson(np.exp(-0.5 + 0.9 * fp))
    data["chargeback_count"] = rng.poisson(np.exp(-1.0 + 0.9 * fp))
    data["card_bin_risk_score"] = np.round(
        np.clip(30 + 22 * z_fraud + rng.normal(0, 6, n), 0, 100), 2
    )
    data["abuse_rate_per_card"] = np.round(
        np.clip(0.02 + 0.25 * _sigmoid(z_fraud) + rng.normal(0, 0.02, n), 0, 1), 4
    )
    data["vpn_flag"] = rng.binomial(1, _sigmoid(-1.0 + 0.7 * z_fraud))
    data["proxy_flag"] = rng.binomial(1, _sigmoid(-1.6 + 0.7 * z_fraud))
    data["amount"] = np.round(np.exp(rng.normal(4.0 + 0.35 * z_fraud, 1.0)), 2)
    data["payments_last_5m"] = rng.poisson(np.exp(-0.5 + 0.7 * fp))
    data["payments_last_1h"] = rng.poisson(np.exp(0.1 + 0.7 * fp))

    # Upstream signup trust score: lower for any abuse propensity.
    data["trust_score"] = np.round(
        np.clip(
            80 - 12 * tp - 12 * dp - 15 * fp + rng.normal(0, 6, n),
            0,
            100,
        ),
        1,
    )
    data["card_trust_score"] = np.round(
        np.clip(78 - 20 * fp + rng.normal(0, 6, n), 0, 100), 2
    )

    # Consistent plan_type from the trial/discount flags.
    plan_type = np.where(
        is_trial == 1,
        "trial",
        np.where(
            is_discounted == 1,
            "discounted",
            rng.choice(["full_price", "standard"], size=n),
        ),
    )
    data["plan_type"] = plan_type

    # Class assignment: elevated driver wins; otherwise legit ------------------
    score_trial = z_trial + 0.6 * is_trial
    score_disc = z_disc + 0.6 * is_discounted
    score_fraud = z_fraud
    stacked = np.vstack([score_trial, score_disc, score_fraud])
    best_idx = np.argmax(stacked, axis=0)  # 0=trial,1=disc,2=fraud
    best_val = np.max(stacked, axis=0)
    label = np.where(best_val > 0.95, best_idx + 1, 0).astype(int)
    data["label"] = label

    # Identifier columns -------------------------------------------------------
    data["transaction_id"] = np.array([f"tx_{i:08d}" for i in range(n)])
    data["user_id"] = np.array([f"u_{v:07d}" for v in rng.integers(0, max(2, n), n)])
    data["organization_id"] = np.array(
        [f"org_{v:06d}" for v in rng.integers(0, max(2, n // 2), n)]
    )
    # Realistic hex fingerprints / dotted-quad IPs, drawn from a smaller pool so
    # shared-card / shared-device / shared-IP abuse signals still show up.
    card_pool = _random_hex(rng, max(2, n // 3), 32)
    device_pool = _random_hex(rng, max(2, n // 3), 32)
    signup_device_pool = _random_hex(rng, max(2, n // 3), 32)
    ip_pool = _random_ipv4(rng, max(2, n // 3))

    data["card_fingerprint"] = card_pool[rng.integers(0, len(card_pool), n)]
    data["signup_device_fingerprint"] = signup_device_pool[rng.integers(0, len(signup_device_pool), n)]
    data["payment_device_fingerprint"] = np.where(
        rng.random(n) < 0.75,
        data["signup_device_fingerprint"],
        device_pool[rng.integers(0, len(device_pool), n)],
    )
    data["ip_address"] = ip_pool[rng.integers(0, len(ip_pool), n)]

    # Fill remaining schema columns with plausible noise -----------------------
    all_cols = cfg.payment_features + PAYMENT_IDENTIFIER_COLUMNS
    for col in all_cols:
        if col not in data:
            data[col] = _fill_generic(col, n, rng)

    df = pd.DataFrame(data)
    ordered = PAYMENT_IDENTIFIER_COLUMNS + cfg.payment_features + ["label"]
    return df[ordered]
