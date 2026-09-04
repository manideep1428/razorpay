"""Generate a 10,000,000 dataset in chunked Parquet files and upload directly to Hugging Face Hub."""

import argparse
import gc
import os
import shutil
import socket
import time
from pathlib import Path

# Force IPv4 socket resolution to prevent Windows IPv6 DNS timeouts
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    res = _orig_getaddrinfo(*args, **kwargs)
    ipv4 = [r for r in res if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else res
socket.getaddrinfo = _ipv4_getaddrinfo

import pyarrow as pa
import pyarrow.parquet as pq

from trust_radar.utils.synthetic import (
    synthesize_payment_dataset,
    synthesize_signup_dataset,
)


def chunk_ranges(total_rows: int, chunk_size: int):
    """Yield start and size for each chunk."""
    for start in range(0, total_rows, chunk_size):
        yield start, min(chunk_size, total_rows - start)


def generate_and_save_parquet(
    generator_fn,
    total_rows: int,
    output_dir: Path,
    chunk_size: int = 500_000,
    test_ratio: float = 0.1,
    name: str = "dataset",
):
    """Stream dataset generation into train and test Parquet shards (skips if already generated)."""
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    test_rows = int(total_rows * test_ratio)
    train_rows = total_rows - test_rows

    print(f"\n--- Checking/Generating {name.upper()} ({total_rows:,} total: {train_rows:,} train / {test_rows:,} test) ---")

    # 1. Generate Train Shards
    shard_idx = 0
    for start, size in chunk_ranges(train_rows, chunk_size):
        shard_path = train_dir / f"shard_{shard_idx:03d}.parquet"
        if shard_path.exists() and shard_path.stat().st_size > 100_000:
            print(f"  Train shard {shard_idx} already cached on disk ({shard_path.stat().st_size / 1e6:.1f} MB).")
            shard_idx += 1
            continue

        t0 = time.time()
        print(f"  Generating train shard {shard_idx} ({size:,} rows)...", end="", flush=True)
        df = generator_fn(n=size, seed=1000 + shard_idx)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, shard_path, compression="snappy")
        del df, table
        gc.collect()
        print(f" Saved in {time.time() - t0:.1f}s ({shard_path.stat().st_size / 1e6:.1f} MB)")
        shard_idx += 1

    # 2. Generate Test Shards
    test_shard_idx = 0
    for start, size in chunk_ranges(test_rows, chunk_size):
        shard_path = test_dir / f"shard_{test_shard_idx:03d}.parquet"
        if shard_path.exists() and shard_path.stat().st_size > 100_000:
            print(f"  Test shard {test_shard_idx} already cached on disk ({shard_path.stat().st_size / 1e6:.1f} MB).")
            test_shard_idx += 1
            continue

        t0 = time.time()
        print(f"  Generating test shard {test_shard_idx} ({size:,} rows)...", end="", flush=True)
        df = generator_fn(n=size, seed=9000 + test_shard_idx)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, shard_path, compression="snappy")
        del df, table
        gc.collect()
        print(f" Saved in {time.time() - t0:.1f}s ({shard_path.stat().st_size / 1e6:.1f} MB)")
        test_shard_idx += 1


def upload_to_huggingface(local_dir: Path, repo_id: str, token: str | None = None):
    """Upload dataset files shard by shard with automatic retry and skip logic."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install via: pip install huggingface_hub")

    api = HfApi(token=token)
    print(f"\nConnecting to Hugging Face dataset repository: {repo_id} ...")
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)

    try:
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token))
    except Exception as err:
        print(f"Note: Could not fetch remote file list ({err}), will upload all files.")
        remote_files = set()

    all_files = sorted([p for p in local_dir.rglob("*") if p.is_file()])
    total_files = len(all_files)
    print(f"\nFound {total_files} files in {local_dir} to upload to https://huggingface.co/datasets/{repo_id}")

    t_all = time.time()
    for i, file_path in enumerate(all_files, 1):
        rel_path = file_path.relative_to(local_dir).as_posix()
        size_mb = file_path.stat().st_size / 1e6

        if rel_path in remote_files:
            print(f"[{i:02d}/{total_files:02d}] Skipping (already on Hub): {rel_path}")
            continue

        print(f"[{i:02d}/{total_files:02d}] Uploading {rel_path} ({size_mb:.1f} MB)... ", end="", flush=True)
        uploaded = False
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=rel_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print(f"Done ({time.time() - t0:.1f}s)")
                uploaded = True
                break
            except Exception as e:
                print(f"\n  [Retry {attempt}/3] Upload error: {e}. Retrying in 3s...")
                time.sleep(3)
        if not uploaded:
            raise RuntimeError(f"Failed to upload {rel_path} after 3 attempts.")

    print(f"\n[SUCCESS] All {total_files} files uploaded in {(time.time() - t_all)/60:.2f} minutes!")
    print(f"Dataset is now live at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Generate 10M dataset and upload to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo id, e.g. 'username/fraudshield-10m'")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Write Token (or set HF_TOKEN env var)")
    parser.add_argument("--rows", type=int, default=10_000_000, help="Total rows to generate per model (default: 10,000,000)")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Rows per Parquet shard (default: 500,000)")
    parser.add_argument("--local-dir", type=Path, default=Path("hf_data"), help="Local temporary output directory")
    parser.add_argument("--keep-local", action="store_true", help="Keep local parquet files after upload")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Note: No HF token provided. Will use existing huggingface-cli login if available.")

    base_dir = args.local_dir
    signup_dir = base_dir / "signup"
    payment_dir = base_dir / "payment"

    t_start = time.time()

    # Generate Signup Parquet Shards (caches if already on disk)
    generate_and_save_parquet(
        generator_fn=synthesize_signup_dataset,
        total_rows=args.rows,
        output_dir=signup_dir,
        chunk_size=args.chunk_size,
        name="signup",
    )

    # Generate Payment Parquet Shards (caches if already on disk)
    generate_and_save_parquet(
        generator_fn=synthesize_payment_dataset,
        total_rows=args.rows,
        output_dir=payment_dir,
        chunk_size=args.chunk_size,
        name="payment",
    )

    # Write Dataset Card (README.md)
    readme_path = base_dir / "README.md"
    readme_content = f"""---
license: apache-2.0
task_categories:
- tabular-classification
- graph-ml
tags:
- fraud-detection
- graphsage
- lightgbm
- fintech
- trust-radar
size_categories:
- 10M<n<100M
---

# FraudShield AI: 10M Synthetic Fraud & Abuse Dataset

This dataset contains **{args.rows:,}** synthetic signup events and **{args.rows:,}** synthetic payment events modeled after enterprise fintech trust and anti-abuse systems.

## Dataset Structure
- `signup/`: Features for Graph Neural Network (GraphSAGE) signup trust scoring (train: 90%, test: 10%).
- `payment/`: Features for multi-class LightGBM payment abuse classification (train: 90%, test: 10%).

## Quickstart
```python
from datasets import load_dataset

signup_train = load_dataset("{args.repo_id}", data_dir="signup", split="train")
payment_train = load_dataset("{args.repo_id}", data_dir="payment", split="train")
```
"""
    readme_path.write_text(readme_content, encoding="utf-8")

    # Upload to Hugging Face
    upload_to_huggingface(local_dir=base_dir, repo_id=args.repo_id, token=token)

    # Cleanup local disk if not requested to keep
    if not args.keep_local:
        print(f"\nCleaning up local temporary folder {base_dir}...")
        shutil.rmtree(base_dir, ignore_errors=True)
        print("Local shards cleaned up. Disk space reclaimed.")

    print(f"\nTotal Pipeline Completed in {(time.time() - t_start)/60:.2f} minutes!")


if __name__ == "__main__":
    main()
