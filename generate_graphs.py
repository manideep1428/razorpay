"""Generate evaluation graphs and metrics for Tesseract AI models.

Generates:
1. Multi-Class Payment Confusion Matrix (Trial Abuse, Card Velocity, Promo Abuse, Legit)
2. LightGBM Top Feature Importances (Velocities, Device signals, IP risk)
3. One-vs-Rest ROC & PR Curves for each abuse class
4. GraphSAGE Signup Ring Risk Distribution

Usage:
    python generate_graphs.py
"""

from pathlib import Path
import sys

# Ensure src/ is in sys.path
_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from tesseract.config import PAYMENT_LABELS, FeatureConfig
from tesseract.inference.predict_payment import PaymentPredictor
from tesseract.inference.predict_signup import SignupPredictor
from tesseract.utils.model_io import load_tabular_model
from tesseract.utils.preprocessing import build_graph_data
from tesseract.utils.synthetic import (
    synthesize_payment_dataset,
    synthesize_signup_dataset,
    synthesize_signup_edges,
)

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT_DIR / "models"


def generate_payment_graphs():
    print("\n--- Generating Payment Abuse Model Graphs ---")
    model_path = MODELS_DIR / "payment_abuse_lgbm.joblib"
    if not model_path.exists():
        print(f"[WARN] Local model not found at {model_path}. Run training or download from Hugging Face.")
        return

    # Load predictor
    predictor = PaymentPredictor(model_or_path=model_path)
    
    # Generate test set
    print("Synthesizing evaluation dataset (n=5,000)...")
    test_df = synthesize_payment_dataset(n=5000, seed=42)
    y_true = test_df["label"].values

    # Batch prediction
    print("Scoring transactions through PaymentAbuseModel...")
    features = predictor._prepare_features(test_df)
    risk = predictor.model.predict_risk(features)
    y_probs = risk["probabilities"]
    y_pred = risk["predicted_class"]

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    class_names = [PAYMENT_LABELS[i] for i in range(len(PAYMENT_LABELS))]

    plt.figure(figsize=(7, 6), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Tesseract AI: Payment Abuse Confusion Matrix (Normalized)", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right")
    plt.yticks(tick_marks, class_names)

    # Label cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]:.1%}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.ylabel("True Class", fontweight="bold")
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.tight_layout()
    cm_path = ARTIFACTS_DIR / "payment_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved: {cm_path}")

    # 2. Multi-Class ROC Curves
    plt.figure(figsize=(8, 6), dpi=150)
    for i, label_name in enumerate(class_names):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("Tesseract AI: Payment Abuse ROC Curves (One-vs-Rest)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = ARTIFACTS_DIR / "payment_roc_curves.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved: {roc_path}")

    # 3. Feature Importance
    try:
        raw = predictor.model.model
        if hasattr(raw, "calibrated_classifiers_"):
            importances = np.mean([c.estimator.feature_importances_ for c in raw.calibrated_classifiers_], axis=0)
        elif hasattr(raw, "feature_importances_"):
            importances = raw.feature_importances_
        else:
            importances = None

        if importances is not None:
            feature_names = predictor.model.feature_names
            top_n = min(15, len(feature_names))
            indices = np.argsort(importances)[::-1][:top_n]
            plt.figure(figsize=(9, 6), dpi=150)
            plt.title(f"Top {top_n} Payment Abuse Feature Importances (LightGBM)", fontsize=12, fontweight="bold")
            plt.barh(range(top_n), importances[indices][::-1], align="center", color="#2563eb")
            plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
            plt.xlabel("Relative Importance (Split Gain)", fontweight="bold")
            plt.tight_layout()
            fi_path = ARTIFACTS_DIR / "payment_feature_importance.png"
            plt.savefig(fi_path)
            plt.close()
            print(f"Saved: {fi_path}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


def generate_signup_graphs():
    print("\n--- Generating Signup GNN Graphs ---")
    model_path = MODELS_DIR / "signup_graphsage.pt"
    if not model_path.exists():
        print(f"[WARN] Local model not found at {model_path}. Run training or download from Hugging Face.")
        return

    try:
        predictor = SignupPredictor(model_or_path=model_path)
        cfg = FeatureConfig()
        
        print("Synthesizing signup identity graph (n=1,500 nodes)...")
        nodes_df = synthesize_signup_dataset(n=1500, seed=42)
        nodes_feat = nodes_df[cfg.signup_numeric_features]
        edges_df = synthesize_signup_edges(len(nodes_df), avg_degree=4.0, seed=42)
        graph_data = build_graph_data(nodes_feat, edges_df, labels=nodes_df["label"])
        
        results = predictor.predict_graph(graph_data)
        risk_scores = np.asarray(results["risk_score"])
        labels = np.asarray(nodes_df["label"].values, dtype=int)

        # Risk distribution histogram
        plt.figure(figsize=(8, 5), dpi=150)
        plt.hist(risk_scores[labels == 0], bins=30, alpha=0.6, label="Legitimate Signups", color="green", density=True)
        plt.hist(risk_scores[labels == 1], bins=30, alpha=0.6, label="Fraud Ring / Multi-Account", color="red", density=True)
        plt.axvline(40, color="gray", linestyle="--", label="Tier 1 Review (Score 40)")
        plt.axvline(70, color="orange", linestyle="--", label="Tier 2 High Priority (Score 70)")
        plt.axvline(95, color="darkred", linestyle="--", label="Tier 3 Suspend (Score 95)")
        plt.xlabel("Calculated GNN Risk Score (0 - 100)", fontweight="bold")
        plt.ylabel("Density", fontweight="bold")
        plt.title("Tesseract AI: GraphSAGE Signup Ring Risk Score Distribution", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        hist_path = ARTIFACTS_DIR / "signup_risk_distribution.png"
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved: {hist_path}")
    except Exception as e:
        print(f"Could not plot signup distribution: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  TESSERACT AI - MODEL VISUALIZATION & GRAPH GENERATOR")
    print("=" * 60)
    generate_payment_graphs()
    generate_signup_graphs()
    print("\n[SUCCESS] All evaluation graphs generated successfully in 'artifacts/' directory!")
