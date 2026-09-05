# 🛡️ Tesseract AI

**Tesseract AI** is an enterprise-grade Machine Learning + Graph Neural Network system for **signup abuse mitigation** and **multi-class payment abuse detection**. It powers real-time fraud scoring, producing calibrated **0–100 risk scores** and tiered operational decisions.

```text
User Signup  ──▶  Signup Trust Model (GraphSAGE GNN)  ──▶  trust_score (0-100)  ──▶ Stored in signups_db.json
Payment Tx   ──▶  Payment Abuse Model (LightGBM)      ──▶  payment_risk_score   ──▶ Tiered Action in transactions_db.json
```

---

## 📦 Model Artifacts & Hugging Face Download

To keep the Git repository lightweight and adhere to best practices, **large model binary checkpoints (`.joblib`, `.pt`) are NOT tracked in Git** (managed via `.gitignore`).

### Hugging Face Models Hub
All pretrained models are hosted on Hugging Face:
🔗 **[vicky1428/fraudshield-models](https://huggingface.co/vicky1428/fraudshield-models)**

| Model File | Type | Architecture / Framework | Size | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `payment_abuse_lgbm.joblib` | Tabular Classifier | Calibrated Multi-Class LightGBM | ~19 MB | Scores payment & card velocity abuse across 4 distinct classes |
| `signup_graphsage.pt` | Graph Neural Network | PyTorch Geometric GraphSAGE | ~60 KB | Evaluates identity graphs (shared IP, device fingerprint, CIDR) |

### How to Download Models

#### Option 1: Automatic Download
The built-in application server (`app.py`) automatically downloads missing model weights from Hugging Face on startup.

#### Option 2: Download via Python Script
```python
from huggingface_hub import hf_hub_download

# Download Payment Abuse Model
hf_hub_download(
    repo_id="vicky1428/fraudshield-models",
    filename="payment_abuse_lgbm.joblib",
    local_dir="models"
)

# Download Signup GraphSAGE Model
hf_hub_download(
    repo_id="vicky1428/fraudshield-models",
    filename="signup_graphsage.pt",
    local_dir="models"
)
```

#### Option 3: Download via Hugging Face CLI
```bash
pip install huggingface-hub
huggingface-cli download vicky1428/fraudshield-models payment_abuse_lgbm.joblib --local-dir models
huggingface-cli download vicky1428/fraudshield-models signup_graphsage.pt --local-dir models
```

---

## 💻 Local Development & Setup Rules

Follow these rules to set up and run Tesseract locally:

### 1. Prerequisites
- **Python 3.12+**
- Recommended environment tool: `uv` (or standard `venv` + `pip`)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/manideep1428/razorpay.git
cd razorpay

# Create virtual environment with Python 3.12
uv venv --python 3.12
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install the editable package and dependencies
uv pip install -e .
# Alternatively using standard pip:
pip install -e .
pip install -r requirements.txt
```

### 3. Running the Real Model Serve Server
Run the zero-dependency simulation and model-serving web server:
```bash
python app.py 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser:
- **Sign Up (`/signup`)**: Real client telemetry extraction (IP, User-Agent, device fingerprint), passing into GraphSAGE to generate real `trust_score`.
- **Payment Checkout (`/payment`)**: Choose between **Standard Plan ($19.00)** and **Free Trial ($0.00)**. Real card BIN analysis, database velocity detection, and LightGBM model scoring.
- **Audit Database (`/admin`)**: Live inspection of entries logged in `signups_db.json` and `transactions_db.json`.
- **Clear Database (`/clear`)**: Resets both JSON databases to `[]` for clean end-to-end test runs.

### 4. Database Files (`signups_db.json` & `transactions_db.json`)
- `signups_db.json` and `transactions_db.json` start as empty JSON arrays (`[]`).
- Each user signup and payment transaction logs telemetry, extracted signals, and raw model inference outputs directly into these files.

---

## 🚦 Decision Rules, Statuses & How It Works

Tesseract implements a 4-tier decision matrix along with enterprise business rules:

### 1. Signup Decision Statuses (GraphSAGE GNN)
Based on calculated `risk_score` (0 - 100):
- **`ALLOW` (0 - 40)**: Clean, legitimate user profile.
- **`ALLOW_FLAG_REVIEW` (41 - 70)**: Borderline indicators (e.g. shared device or datacenter IP); allowed without friction, queued for asynchronous review.
- **`ALLOW_HIGH_PRIORITY_REVIEW` (71 - 94)**: High risk indicators; queued for urgent investigation.
- **`TEMP_SUSPEND_MANUAL_REVIEW` (95 - 100)**: Coordinated fraud ring or bot farm detected; account temporarily locked.

### 2. Payment Decision Statuses & Rules (LightGBM)
- **Standard / Paid Plans ($19.00) are ALWAYS ALLOWED (`ALLOW`)**: Captures legitimate customer revenue without false positives.
- **Free Trial Card Reuse Abuse (`BLOCK_TRIAL`)**: When a payment card is reused across multiple accounts for free trials, Tesseract detects the velocity and blocks the trial.
- **High Risk Trial Abuse (`BLOCK`)**: Any trial registration with model `payment_risk_score >= 50` is blocked.
- **`BLOCK_CARD_VELOCITY`**: Triggered when the same card attempts >5 transactions in a short window.
- **`BLOCK_IP_VELOCITY`**: Triggered when a single IP floods multiple payment requests.
- **`CHALLENGE_3DS`**: Step-up authentication required for anomalous payment parameters.

### 3. Abuse Classification Categories
LightGBM predicts probabilities across 4 distinct abuse classes:
1. `0 = Legit`: Genuine user transaction.
2. `1 = Trial Abuse`: Card recycling, multi-accounting to obtain recurring free trials.
3. `2 = Card Velocity Fraud`: Automated card testing or rapid sequential transactions.
4. `3 = Promo / Referral Abuse`: Exploiting referral codes and promo discounts.

---

## 📊 How to Generate Model Evaluation Graphs & Charts

Tesseract includes a turnkey visualization generator script: [`generate_graphs.py`](generate_graphs.py).

### Run the Graph Generator
Ensure model weights exist in `models/` (or run `python app.py` once to download them), then run:
```bash
python generate_graphs.py
```

### Generated Artifacts (`artifacts/` directory)
The script evaluates the models and outputs publication-quality visualizations:

1. **`artifacts/payment_confusion_matrix.png`**:
   - Normalized multi-class confusion matrix showing model classification precision across Legit, Trial Abuse, Card Velocity, and Promo Abuse.
2. **`artifacts/payment_roc_curves.png`**:
   - One-vs-Rest (OvR) ROC curves displaying AUC scores for each abuse category.
3. **`artifacts/payment_feature_importance.png`**:
   - Top 15 split gain feature importance rankings from LightGBM (card velocity, account age, proxy flags, trust score).
4. **`artifacts/signup_risk_distribution.png`**:
   - GraphSAGE risk score distribution comparing legitimate users vs coordinated fraud ring clusters.

---

## ⚡ Training Pipeline & Google Colab

Tesseract models can be trained on synthetic data or streamed from the 10M record Hugging Face dataset:
🔗 **[vicky1428/fraudshield-10m](https://huggingface.co/datasets/vicky1428/fraudshield-10m)**

### Train Locally
```bash
# Train both models with smart shard streaming
python train.py --max-rows 500000 --epochs 50 --trees 300
```

### Test Against Held-out Evaluation Split
```bash
python test.py --test-rows 10000
```

### Train in Google Colab (GPU Acceleration)
Open [colab_train.ipynb](colab_train.ipynb) in Google Colab to train with full CUDA acceleration and automatically export trained weights.

---

## 📂 Project Structure

```
razorpay/
├── app.py                      # Real model serve & live simulation server (zero CSS/JS)
├── train.py                    # Training pipeline streaming from Hugging Face Hub
├── test.py                     # Evaluation pipeline against held-out test splits
├── generate_graphs.py          # Generates evaluation charts, ROC curves & confusion matrices
├── pyproject.toml              # Modern Hatchling build configuration (package: tesseract)
├── requirements.txt            # Pinned requirements for pip environments
├── README.md                   # System documentation & setup guide
├── signups_db.json             # Clean local user database (generates on test runs)
├── transactions_db.json        # Clean local transaction database (generates on test runs)
├── models/
│   ├── .gitkeep                # Keeps models directory in git (binaries are git-ignored)
│   ├── payment_abuse_lgbm.joblib  (downloaded from HF Hub)
│   └── signup_graphsage.pt        (downloaded from HF Hub)
├── artifacts/                  # Generated plots and evaluation charts (.png)
├── src/
│   └── tesseract/              # Core Python package
│       ├── __init__.py
│       ├── config.py           # Feature schemas, hyperparams, and threshold configs
│       ├── decisioning.py      # 0-100 scores, risk levels, and 4-tier decisions
│       ├── models/             # signup_gnn.py, payment_model.py
│       ├── training/           # train_signup.py, train_payment.py
│       ├── evaluation/         # evaluate_signup.py, evaluate_payment.py
│       ├── inference/          # predict_signup.py, predict_payment.py
│       └── utils/              # metrics, preprocessing, model_io, synthetic
└── tests/
    ├── test_signup_model.py    # GNN forward pass, shape, and score tests
    ├── test_payment_model.py   # LightGBM training, persistence, and batch tests
    └── test_decisioning.py     # Threshold, plan gating, and metric tests
```

---

## 🧪 Running Tests

Run the full pytest test suite:
```bash
pytest tests/ -v
```
All tests run in-memory and validate decisioning, model scoring, and plan-gating logic.
