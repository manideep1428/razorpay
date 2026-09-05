"""Test FraudShield AI models against the held-out test split from Hugging Face Hub."""

import argparse
import os
import socket
import time

# Force IPv4 socket resolution to prevent Windows IPv6 DNS timeouts
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    ipv4 = [r for r in res if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else res
socket.getaddrinfo = _ipv4_getaddrinfo

import sys
from pathlib import Path

# Ensure src/ is in sys.path regardless of execution directory
_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pandas as pd
from datasets import load_dataset

from trust_radar.config import FeatureConfig
from trust_radar.inference.predict_payment import PaymentPredictor
from trust_radar.inference.predict_signup import SignupPredictor
from trust_radar.utils.preprocessing import build_graph_data
from trust_radar.utils.synthetic import synthesize_signup_edges


import math
from pathlib import Path


def resolve_test_shards(category: str, test_rows: int | None) -> str | list[str]:
    """Select only the needed test parquet shards based on row count."""
    if not test_rows:
        return f"{category}/test/*.parquet"
    needed = max(1, math.ceil(test_rows / 500_000))
    needed = min(needed, 2)
    if needed == 1:
        return f"{category}/test/shard_000.parquet"
    return [f"{category}/test/shard_{i:03d}.parquet" for i in range(needed)]


def parse_args():
    parser = argparse.ArgumentParser(description="Test FraudShield AI against Hugging Face test splits.")
    parser.add_argument("--repo-id", type=str, default="vicky1428/fraudshield-10m", help="Hugging Face Dataset repo id (default: 'vicky1428/fraudshield-10m')")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (optional for public dataset)")
    parser.add_argument("--test-rows", type=int, default=10_000, help="Number of test samples to score (default: 10,000)")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory where trained models are stored (default: 'artifacts')")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    artifacts_dir = Path(args.artifacts_dir)

    print("=" * 75)
    print(f"EVALUATING MODELS ON HUGGING FACE TEST SPLIT: {args.repo_id}")
    print("=" * 75)

    feat_cfg = FeatureConfig()

    # -----------------------------------------------------------------------
    # TEST 1: SIGNUP TRUST MODEL
    # -----------------------------------------------------------------------
    print("\n--- [TEST 1] Evaluating Signup Trust Model (GraphSAGE) ---")
    signup_shards = resolve_test_shards("signup", args.test_rows)
    print(f"Fetching test shard(s): {signup_shards} from {args.repo_id}...")
    ds_signup_test = load_dataset(
        args.repo_id,
        data_files={"test": signup_shards},
        split="test",
        token=token,
    )
    if args.test_rows and len(ds_signup_test) > args.test_rows:
        ds_signup_test = ds_signup_test.select(range(args.test_rows))

    df_s_test = ds_signup_test.to_pandas()
    print(f"Loaded {len(df_s_test):,} held-out test signup rows.")

    signup_predictor = SignupPredictor(model_or_path=artifacts_dir / "signup_graphsage.pt")

    nodes = df_s_test[feat_cfg.signup_numeric_features]
    edges = synthesize_signup_edges(num_nodes=len(df_s_test), avg_degree=4.0, seed=999)
    test_graph = build_graph_data(nodes, edges, labels=df_s_test["label"])

    t0 = time.time()
    signup_results = signup_predictor.predict_graph(test_graph)
    print(f"Scored {len(df_s_test):,} signup nodes in {time.time() - t0:.2f}s!")

    df_s_eval = pd.DataFrame({
        "trust_score": signup_results["trust_score"],
        "risk_score": signup_results["risk_score"],
        "risk_level": signup_results["risk_level"],
        "decision": signup_results["decision"],
    })
    print("\nSignup Decision Distribution on HF Test Split:")
    print(df_s_eval["decision"].value_counts().to_string())

    # -----------------------------------------------------------------------
    # TEST 2: PAYMENT ABUSE MODEL
    # -----------------------------------------------------------------------
    print("\n\n--- [TEST 2] Evaluating Payment Abuse Model (LightGBM) ---")
    payment_shards = resolve_test_shards("payment", args.test_rows)
    print(f"Fetching test shard(s): {payment_shards} from {args.repo_id}...")
    ds_payment_test = load_dataset(
        args.repo_id,
        data_files={"test": payment_shards},
        split="test",
        token=token,
    )
    if args.test_rows and len(ds_payment_test) > args.test_rows:
        ds_payment_test = ds_payment_test.select(range(args.test_rows))

    df_p_test = ds_payment_test.to_pandas()
    print(f"Loaded {len(df_p_test):,} held-out test payment transactions.")

    payment_predictor = PaymentPredictor(model_or_path=artifacts_dir / "payment_abuse_lgbm.joblib")

    t0 = time.time()
    pay_results = payment_predictor.score_batch(df_p_test)
    print(f"Scored {len(df_p_test):,} payment transactions in {time.time() - t0:.2f}s!")

    df_p_eval = pd.DataFrame({
        "risk_score": pay_results["payment_risk_score"],
        "abuse_type": pay_results["abuse_type"],
        "risk_level": pay_results["risk_level"],
        "decision": pay_results["decision"],
    })
    print("\nPayment Decision Distribution on HF Test Split:")
    print(df_p_eval["decision"].value_counts().to_string())

    print("\n" + "=" * 75)
    print("ALL HUGGING FACE TEST SPLIT EVALUATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
