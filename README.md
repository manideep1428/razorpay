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
├── requirements.txt
├── README.md
├── train.py                 # Primary training entrypoint (streams from Hugging Face Hub)
├── test.py                  # Primary evaluation entrypoint (held-out test split)
├── artifacts/               # Trained model checkpoints (.pt, .joblib)
├── data/                    # Local scratch directory (.gitkeep)
├── src/
│   └── trust_radar/
│       ├── __init__.py
│       ├── config.py        # Feature schemas, hyperparams, and threshold configs
│       ├── decisioning.py   # 0-100 scores, risk levels, tiered decisions
│       ├── models/          # signup_gnn.py, payment_model.py
│       ├── training/        # train_signup.py, train_payment.py
│       ├── evaluation/      # evaluate_signup.py, evaluate_payment.py
│       ├── inference/       # predict_signup.py, predict_payment.py
│       └── utils/           # metrics, preprocessing, model_io, synthetic
└── tests/
    ├── test_signup_model.py
    ├── test_payment_model.py
    └── test_decisioning.py
```

---

## ⚡ Google Colab & Fast Start

You can train directly in Google Colab using GPU acceleration either with the turnkey notebook [colab_train.ipynb](file:///c:/Users/saima/OneDrive/Desktop/razor-hac/colab_train.ipynb) or from the terminal:

### Option A: Open `colab_train.ipynb`
Open `colab_train.ipynb` in Google Colab for interactive cells, GPU checks, metric plots, and one-click Drive/HF Hub export.

### Option B: Terminal / Cell Quickstart
```bash
# 1. Clone repository
!git clone https://github.com/manideep1428/razorpay.git
%cd razorpay

# 2. Install dependencies (Colab pre-installs PyTorch CUDA & pandas)
!pip install -q torch-geometric lightgbm datasets huggingface-hub pyarrow

# 3. Train models directly from HF dataset (smart shard loading, e.g. 500k rows)
!python train.py --max-rows 500000 --epochs 100 --trees 300 --device cuda

# 4. Evaluate against held-out test split
!python test.py --test-rows 10000
```

---

## 🚀 Local Usage

```bash
uv sync  # or pip install -r requirements.txt
python train.py
python test.py
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
