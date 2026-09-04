"""Scale training pipeline for 5,000,000 datasets on Google Colab or High-VRAM GPUs."""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trust_radar.config import (
    ARTIFACTS_DIR,
    FeatureConfig,
    PaymentModelConfig,
    SignupGNNConfig,
)
from trust_radar.evaluation.evaluate_signup import evaluate_signup_gnn
from trust_radar.training.train_payment import train_payment_model
from trust_radar.training.train_signup import train_signup_gnn
from trust_radar.utils.preprocessing import build_graph_data
from trust_radar.utils.synthetic import (
    synthesize_payment_dataset,
    synthesize_signup_dataset,
    synthesize_signup_edges,
)


def main():
    print("=" * 75)
    print("FRAUDSHIELD AI: TRAINING ON 5,000,000 RECORDS PER MODEL")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Note: Running on CPU (GPU recommended for 5M scale)")

    feat_cfg = FeatureConfig()
    N_SAMPLES = 5_000_000

    # -----------------------------------------------------------------------
    # STAGE 1: SIGNUP TRUST MODEL (GraphSAGE GNN) - 5,000,000 Nodes
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STAGE 1: GENERATING 5,000,000 SIGNUP NODES & GRAPH EDGES")
    print("-" * 75)
    t0 = time.time()

    # Fast memory-optimized synthesis without string identifiers
    df_signup = synthesize_signup_dataset(n=N_SAMPLES, seed=42, include_identifiers=False)
    nodes = df_signup[feat_cfg.signup_numeric_features]
    edges = synthesize_signup_edges(num_nodes=N_SAMPLES, avg_degree=4.0, seed=42)
    data = build_graph_data(nodes, edges, labels=df_signup["label"])

    print(f"Graph constructed in {time.time() - t0:.2f}s:")
    print(f"  Nodes     : {data.num_nodes:,}")
    print(f"  Edges     : {data.num_edges:,}")
    print(f"  Abuse Rate: {float(df_signup['label'].mean()):.4f}")

    # Free memory before GPU training
    del df_signup, nodes, edges
    gc.collect()

    print("\nTRAINING SIGNUP GRAPHSAGE MODEL (5M Nodes, 100 Epochs)...")
    t_train_gnn = time.time()
    cfg_signup = SignupGNNConfig(
        epochs=100,
        hidden_channels=64,
        learning_rate=0.005,
        pos_weight=5.0,
    )
    model_gnn, history = train_signup_gnn(data, config=cfg_signup, device=device)
    gnn_time = time.time() - t_train_gnn
    print(f"[SUCCESS] GNN training complete in {gnn_time:.2f}s ({(gnn_time/60):.2f} min)!")

    # Evaluate GNN on held-out test mask
    gnn_metrics = evaluate_signup_gnn(model_gnn, data, split="test", device=device)
    print(f"  Test ROC-AUC        : {gnn_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC         : {gnn_metrics['pr_auc']:.4f}")
    print(f"  Test F1             : {gnn_metrics['f1']:.4f}")
    print(f"  Mean Trust Score    : {gnn_metrics['mean_trust_score']:.1f}/100")
    print(f"  Saved Checkpoint    : {cfg_signup.checkpoint_path}")

    # Completely release Stage 1 memory before Stage 2
    del data, model_gnn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # STAGE 2: PAYMENT ABUSE MODEL (LightGBM) - 5,000,000 Rows
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STAGE 2: GENERATING 5,000,000 PAYMENT TRANSACTIONS")
    print("-" * 75)
    t0 = time.time()
    df_payment = synthesize_payment_dataset(n=N_SAMPLES, seed=42, include_identifiers=False)

    X = df_payment[feat_cfg.payment_features].copy()
    for col in feat_cfg.payment_categorical_features:
        X[col] = X[col].astype("category")
    y = df_payment["label"]

    print(f"Payment data generated in {time.time() - t0:.2f}s:")
    print(f"  Rows    : {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print("  Class Distribution:")
    for cls_idx, count in y.value_counts().sort_index().items():
        print(f"    Class {cls_idx}: {count:,} ({count / len(y):.2%})")

    del df_payment
    gc.collect()

    print("\nTRAINING MULTI-CLASS PAYMENT MODEL (LightGBM on 5M Rows)...")
    t_train_pay = time.time()
    cfg_pay = PaymentModelConfig(n_estimators=300, learning_rate=0.05)
    model_pay, pay_metrics = train_payment_model(
        X, y, config=cfg_pay, test_size=0.2, calibrate=True
    )
    pay_time = time.time() - t_train_pay
    print(f"[SUCCESS] Payment model training complete in {pay_time:.2f}s ({(pay_time/60):.2f} min)!")

    print(f"  Accuracy            : {pay_metrics.get('accuracy', 0):.4f}")
    print(f"  Macro-F1            : {pay_metrics.get('macro_f1', 0):.4f}")
    print(f"  Macro ROC-AUC (OVR) : {pay_metrics.get('macro_roc_auc_ovr', 0):.4f}")
    print(f"  Abuse ROC-AUC       : {pay_metrics.get('abuse_roc_auc', 0):.4f}")
    print(f"  Saved Model         : {cfg_pay.model_path}")

    # -----------------------------------------------------------------------
    # FINAL PERSISTENCE SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("ALL 5,000,000-ROW MODELS TRAINED & PERSISTED SUCCESSFULLY!")
    print("=" * 75)
    print(f"1. GNN Checkpoint (100 Epochs) : {cfg_signup.checkpoint_path} ({cfg_signup.checkpoint_path.stat().st_size / 1024:.1f} KB)")
    print(f"2. LightGBM Model (5M Rows)    : {cfg_pay.model_path} ({cfg_pay.model_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"Total Combined Training Time   : {(gnn_time + pay_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
