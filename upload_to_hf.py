"""Generate a 10,000,000 dataset in chunked Parquet files and upload directly to Hugging Face Hub."""

import argparse
import gc
import os
import shutil
import time
from pathlib import Path

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
    """Stream dataset generation into train and test Parquet shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = int(total_rows * test_ratio)
    train_rows = total_rows - test_rows

    print(f"\n--- Generating {name.upper()} ({total_rows:,} total: {train_rows:,} train / {test_rows:,} test) ---")

    # 1. Generate Train Shards
    shard_idx = 0
    for start, size in chunk_ranges(train_rows, chunk_size):
        t0 = time.time()
        shard_path = output_dir / f"train_{shard_idx:03d}.parquet"
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
        t0 = time.time()
        shard_path = output_dir / f"test_{test_shard_idx:03d}.parquet"
        print(f"  Generating test shard {test_shard_idx} ({size:,} rows)...", end="", flush=True)

        df = generator_fn(n=size, seed=9000 + test_shard_idx)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, shard_path, compression="snappy")

        del df, table
        gc.collect()
        print(f" Saved in {time.time() - t0:.1f}s ({shard_path.stat().st_size / 1e6:.1f} MB)")
        test_shard_idx += 1


def upload_to_huggingface(local_dir: Path, repo_id: str, token: str | None = None):
    """Upload data directory directly to Hugging Face Datasets repo."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install via: pip install huggingface_hub")

    api = HfApi(token=token)
    print(f"\nCreating / connecting to Hugging Face dataset repository: {repo_id} ...")
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)

    print(f"Uploading {local_dir} to https://huggingface.co/datasets/{repo_id} ...")
    t0 = time.time()
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    print(f"Upload complete in {(time.time() - t0)/60:.2f} minutes!")
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

    # Generate Signup Parquet Shards
    generate_and_save_parquet(
        generator_fn=synthesize_signup_dataset,
        total_rows=args.rows,
        output_dir=signup_dir,
        chunk_size=args.chunk_size,
        name="signup",
    )

    # Generate Payment Parquet Shards
    generate_and_save_parquet(
        generator_fn=synthesize_payment_dataset,
        total_rows=args.rows,
        output_dir=payment_dir,
        chunk_size=args.chunk_size,
        name="payment",
    )

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
