"""Train FraudShield AI models using datasets streamed/loaded directly from Hugging Face Hub."""

import argparse
import gc
import os
import time

import pandas as pd
import torch
from datasets import load_dataset

from trust_radar.config import (
    FeatureConfig,
    PaymentModelConfig,
    SignupGNNConfig,
)
from trust_radar.evaluation.evaluate_signup import evaluate_signup_gnn
from trust_radar.training.train_payment import train_payment_model
from trust_radar.training.train_signup import train_signup_gnn
from trust_radar.utils.preprocessing import build_graph_data
from trust_radar.utils.synthetic import synthesize_signup_edges


def parse_args():
    parser = argparse.ArgumentParser(description="Train FraudShield AI directly from Hugging Face Datasets.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Dataset repo id, e.g. 'username/fraudshield-10m'")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (for private datasets)")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to load (default: all)")
    parser.add_argument("--epochs", type=int, default=100, help="GNN training epochs (default: 100)")
    parser.add_argument("--trees", type=int, default=300, help="LightGBM boosting trees (default: 300)")
    parser.add_argument("--calibrate", action="store_true", help="Enable 3-fold calibration for LightGBM")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    print("=" * 75)
    print(f"TRAINING FROM HUGGING FACE DATASET: {args.repo_id}")
    print("=" * 75)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch Compute Device: {device}")
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"GPU Name             : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    feat_cfg = FeatureConfig()

    # -----------------------------------------------------------------------
    # STAGE 1: SIGNUP TRUST MODEL (GraphSAGE GNN) FROM HUGGING FACE
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STAGE 1: LOADING SIGNUP DATASET FROM HUGGING FACE")
    print("-" * 75)
    t0 = time.time()

    # Load signup training split from HF
    ds_signup = load_dataset(
        args.repo_id,
        data_dir="signup",
        split="train",
        token=token,
    )
    if args.max_rows and len(ds_signup) > args.max_rows:
        ds_signup = ds_signup.select(range(args.max_rows))

    print(f"Loaded {len(ds_signup):,} signup records from Hugging Face in {time.time() - t0:.1f}s")
    df_signup = ds_signup.to_pandas()
    del ds_signup
    gc.collect()

    nodes = df_signup[feat_cfg.signup_numeric_features]
    edges = synthesize_signup_edges(num_nodes=len(df_signup), avg_degree=4.0, seed=42)
    labels = df_signup["label"]

    data = build_graph_data(nodes, edges, labels=labels)
    del df_signup, nodes, edges, labels
    gc.collect()

    print(f"\nTRAINING SIGNUP GRAPHSAGE MODEL ({data.num_nodes:,} Nodes, {args.epochs} Epochs)...")
    t_train_gnn = time.time()
    cfg_signup = SignupGNNConfig(
        epochs=args.epochs,
        hidden_channels=64,
        learning_rate=0.005,
        pos_weight=5.0,
    )
    model_gnn, history = train_signup_gnn(data, config=cfg_signup, device=device)
    gnn_time = time.time() - t_train_gnn
    print(f"[SUCCESS] GNN training complete in {gnn_time:.2f}s ({(gnn_time/60):.2f} min)!")

    # Evaluate on held-out split
    gnn_metrics = evaluate_signup_gnn(model_gnn, data, split="test", device=device)
    print(f"  Test ROC-AUC        : {gnn_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC         : {gnn_metrics['pr_auc']:.4f}")
    print(f"  Test F1             : {gnn_metrics['f1']:.4f}")
    print(f"  Mean Trust Score    : {gnn_metrics['mean_trust_score']:.1f}/100")
    print(f"  Saved Checkpoint    : {cfg_signup.checkpoint_path}")

    # Release memory before Stage 2
    del data, model_gnn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # STAGE 2: PAYMENT ABUSE MODEL (LightGBM) FROM HUGGING FACE
    # -----------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STAGE 2: LOADING PAYMENT DATASET FROM HUGGING FACE")
    print("-" * 75)
    t0 = time.time()

    ds_payment = load_dataset(
        args.repo_id,
        data_dir="payment",
        split="train",
        token=token,
    )
    if args.max_rows and len(ds_payment) > args.max_rows:
        ds_payment = ds_payment.select(range(args.max_rows))

    print(f"Loaded {len(ds_payment):,} payment records from Hugging Face in {time.time() - t0:.1f}s")
    df_payment = ds_payment.to_pandas()
    del ds_payment
    gc.collect()

    X = df_payment[feat_cfg.payment_features].copy()
    for col in feat_cfg.payment_categorical_features:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df_payment["label"]

    del df_payment
    gc.collect()

    print(f"\nTRAINING MULTI-CLASS PAYMENT MODEL (LightGBM {args.trees} Trees)...")
    t_train_pay = time.time()
    cfg_pay = PaymentModelConfig(n_estimators=args.trees, learning_rate=0.05)
    model_pay, pay_metrics = train_payment_model(
        X, y, config=cfg_pay, test_size=0.2, calibrate=args.calibrate
    )
    pay_time = time.time() - t_train_pay
    print(f"[SUCCESS] Payment model training complete in {pay_time:.2f}s ({(pay_time/60):.2f} min)!")

    print(f"  Accuracy            : {pay_metrics.get('accuracy', 0):.4f}")
    print(f"  Macro-F1            : {pay_metrics.get('macro_f1', 0):.4f}")
    print(f"  Macro ROC-AUC (OVR) : {pay_metrics.get('macro_roc_auc_ovr', 0):.4f}")
    print(f"  Abuse ROC-AUC       : {pay_metrics.get('abuse_roc_auc', 0):.4f}")
    print(f"  Saved Model         : {cfg_pay.model_path}")

    del X, y
    gc.collect()

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("ALL MODELS TRAINED FROM HUGGING FACE DATASET & PERSISTED!")
    print("=" * 75)
    print(f"1. GNN Checkpoint ({args.epochs} Epochs) : {cfg_signup.checkpoint_path}")
    print(f"2. LightGBM Model ({args.trees} Trees)  : {cfg_pay.model_path}")
    print(f"Total Combined Training Time : {(gnn_time + pay_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
