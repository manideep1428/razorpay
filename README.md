# 🛡️ FraudShield AI

FraudShield AI is an end-to-end Machine Learning + Graph Neural Network system
for signup abuse mitigation and payment abuse detection. It produces
**0-100 scores** and actionable, tiered decisions:

```text
User Signup ──▶ Signup Trust Model  ──▶ trust_score (0-100)  ──▶ stored on user
Payment     ──▶ Payment Abuse Model ──▶ payment_risk_score (0-100) + decision
```

> The importable Python package is `trust_radar` (the project's original name
> and editable-install entry point). Only the product branding is FraudShield AI.

---

## 🧠 Models

### 1. Signup Trust Model (`SignupGraphSAGE`)
Detects fake signups, multi-account abuse, bot signups, device farming, VPN
abuse, trial farmers, and account-creation rings.

- Binary labels: `0 = legit_user`, `1 = abuse_user`.
- Output: `trust_score`, `risk_score`, `risk_level`.
- Built on **PyTorch Geometric** GraphSAGE over shared device / IP / phone /
  email-domain networks.

```json
{ "trust_score": 91, "risk_score": 9, "risk_level": "low" }
```

### 2. Payment Abuse Model (`PaymentAbuseModel`)
A calibrated **multi-class LightGBM** classifier consuming the upstream
`trust_score` plus payment, card, device, IP, velocity, and graph features.

- Classes: `0 = legit`, `1 = trial_abuse`, `2 = discount_abuse`, `3 = payment_fraud`.
- Output: `payment_risk_score` (0-100), `abuse_type`, and a `decision`.

```json
{ "payment_risk_score": 87, "abuse_type": "discount_abuse", "decision": "ALLOW_HIGH_PRIORITY_REVIEW" }
```

---

## 🚦 Decision Logic (both models, on a 0-100 **risk** score)

| Risk score | Signup action | Payment action |
|------------|---------------|----------------|
| 0-40 | `ALLOW` | `ALLOW` |
| 41-70 | `ALLOW_FLAG_REVIEW` | `ALLOW_FLAG_REVIEW` |
| 71-94 | `ALLOW_HIGH_PRIORITY_REVIEW` | `ALLOW_HIGH_PRIORITY_REVIEW` |
| 95-100 | `TEMP_SUSPEND_MANUAL_REVIEW` | `BLOCK` (trial / discount) |

Plan gating: **full-price plans skip the payment model and are always allowed**;
only trial and discounted plans are scored.

> Disposable email, VPN, proxy, and Tor are **risk-increasing signals only** —
> they never trigger an automatic rejection. Every decision is a pure function
> of the final score.

---

## 📂 Project Structure

```
razor-hac/
├── pyproject.toml
├── README.md
├── main.py
├── artifacts/
├── notebooks/
│   ├── 01_signup_gnn.ipynb
│   ├── 02_payment_model.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   └── trust_radar/
│       ├── __init__.py
│       ├── config.py            # full categorized feature schema + configs
│       ├── decisioning.py       # 0-100 scores, risk levels, tiered decisions
│       ├── models/              # signup_gnn.py, payment_model.py
│       ├── training/            # train_signup.py, train_payment.py
│       ├── evaluation/          # evaluate_signup.py, evaluate_payment.py
│       ├── inference/           # predict_signup.py, predict_payment.py
│       └── utils/               # metrics, preprocessing, model_io, synthetic
└── tests/
    ├── test_signup_model.py
    ├── test_payment_model.py
    └── test_decisioning.py
```

---

## 🚀 Getting Started

```bash
uv sync
```

### Signup trust scoring

```python
from trust_radar.inference.predict_signup import SignupPredictor

predictor = SignupPredictor(model_or_path="artifacts/signup_graphsage.pt")
result = predictor.predict_single_node(node_idx=10, data=graph_data)
# {'trust_score': 91, 'risk_score': 9, 'risk_level': 'low', 'decision': 'ALLOW', ...}
```

### Payment abuse scoring

```python
from trust_radar.inference.predict_payment import PaymentPredictor

predictor = PaymentPredictor(model_or_path="artifacts/payment_abuse_lgbm.joblib")
decision = predictor.score_transaction(transaction_row, plan_type="trial")
# {'payment_risk_score': 87, 'abuse_type': 'discount_abuse',
#  'risk_level': 'high', 'decision': 'ALLOW_HIGH_PRIORITY_REVIEW', ...}
```

### Synthetic data (schema-faithful)

```python
from trust_radar.utils.synthetic import (
    synthesize_signup_dataset, synthesize_payment_dataset,
)

signup_df = synthesize_signup_dataset(n=2000)
payment_df = synthesize_payment_dataset(n=4000)
```

---

## 🧪 Tests & Quality

```bash
uv run pytest tests/ -v
uv run ruff check .
```
