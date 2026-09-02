"""Generate a large synthetic FraudShield AI *payment* dataset.

Standalone data-generation utility. It reuses the project's schema-faithful
synthesizer (:func:`trust_radar.utils.synthetic.synthesize_payment_dataset`) so
the output carries every column of the FraudShield payment schema (identifiers
+ features + label) with the same multi-class label logic
(0=legit, 1=trial_abuse, 2=discount_abuse, 3=payment_fraud) the model is
trained on.

This is NOT uniform random noise. Every abuse-relevant column is driven by
latent per-row "abuse propensity" factors (trial/discount/fraud), so the
generated dataset has genuine, learnable signal:

    - trials_per_card_30d, trials_last_24h, users_per_card_30d rise with the
      trial-abuse factor.
    - discounts_per_card_30d, coupon_usage_count, discount_percentage rise
      with the discount-abuse factor.
    - chargebacks_per_card, failed_payments_per_card, card_bin_risk_score,
      abuse_rate_per_card rise with the fraud factor.
    - the upstream trust_score falls as any abuse factor rises (mirroring the
      Signup Trust Model handing off a lower score for risky accounts).
    - card_fingerprint / device_fingerprint / ip_address are realistic-looking
      hex hashes / dotted-quad IPs drawn from a smaller pool per block, so
      shared-card / shared-device / shared-IP abuse signals persist.

Why chunked?
    10,000,000 rows x ~90 columns is too large to hold comfortably in memory as
    a single DataFrame. This script generates the data in blocks (default
    100,000 rows) and streams each block to disk, so peak memory stays at
    roughly one block. ``transaction_id`` is re-issued as a globally unique key
    across blocks; ``user_id`` is intentionally left as-is (drawn from a
    smaller pool) so repeat-customer / multi-transaction patterns still show up
    within and across blocks.

Nothing runs on import. Generate via the CLI, e.g.::

    python generate_payment_data.py --rows 10000000 --output data/payment_dataset.csv
    python generate_payment_data.py --rows 10000000 --output data/payment_dataset.parquet

or import and call :func:`generate_payment_data` directly.

CSV note:
    CSV has no per-column type metadata, so numeric-looking identifier strings
    can be misread on reload. Parquet avoids this entirely and is recommended
    for large runs; prefer ``--format parquet`` for anything beyond a quick look.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trust_radar.config import PAYMENT_IDENTIFIER_COLUMNS, FeatureConfig
from trust_radar.utils.synthetic import synthesize_payment_dataset

DEFAULT_ROWS = 1000
DEFAULT_CHUNK_SIZE = 100
DEFAULT_OUTPUT = Path("data") / "payment_dataset.csv"
DEFAULT_SEED = 42


def _infer_format(output_path: Path, file_format: str | None) -> str:
    """Resolve the output format from an explicit flag or the file suffix."""
    if file_format:
        fmt = file_format.lower()
    elif output_path.suffix.lower() in {".parquet", ".pq"}:
        fmt = "parquet"
    else:
        fmt = "csv"
    if fmt not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'csv' or 'parquet'.")
    return fmt


def _chunk_sizes(n_rows: int, chunk_size: int) -> list[int]:
    """Split ``n_rows`` into a list of block sizes of at most ``chunk_size``."""
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    full, remainder = divmod(n_rows, chunk_size)
    sizes = [chunk_size] * full
    if remainder:
        sizes.append(remainder)
    return sizes


def _relabel_transaction_ids(chunk: pd.DataFrame, start: int) -> pd.DataFrame:
    """Reassign ``transaction_id`` to a globally unique, zero-padded key.

    The per-block synthesizer numbers transactions from 0; this offsets them
    by the running row count so identifiers are unique across the whole file.
    ``transaction_id`` is a pure identifier (never a model feature), so
    rewriting it does not affect any feature or label.

    ``user_id`` / ``organization_id`` / ``card_fingerprint`` /
    ``device_fingerprint`` / ``ip_address`` are intentionally left as the
    synthesizer produced them (drawn from a bounded pool) -- collapsing them
    to globally unique values per block would destroy the repeat-customer and
    shared-card/device/IP abuse signal the model needs to learn.
    """
    stop = start + len(chunk)
    chunk["transaction_id"] = [f"tx_{i:09d}" for i in range(start, stop)]
    return chunk


def generate_payment_data(
    n_rows: int = DEFAULT_ROWS,
    output_path: str | Path = DEFAULT_OUTPUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    seed: int = DEFAULT_SEED,
    file_format: str | None = None,
    verbose: bool = True,
) -> Path:
    """Generate a synthetic payment dataset and stream it to disk in chunks.

    Args:
        n_rows: Total number of payment rows to generate (e.g. 10_000_000).
        output_path: Destination file. ``.parquet``/``.pq`` implies Parquet,
            otherwise CSV (unless ``file_format`` is given).
        chunk_size: Rows generated and written per block (controls peak memory).
        seed: Base RNG seed; block ``i`` uses ``seed + i`` for reproducibility.
        file_format: Force ``"csv"`` or ``"parquet"`` (overrides the suffix).
        verbose: Print per-block progress and a final summary.

    Returns:
        The resolved output :class:`~pathlib.Path`.
    """
    output_path = Path(output_path)
    fmt = _infer_format(output_path, file_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sizes = _chunk_sizes(n_rows, chunk_size)
    started = time.perf_counter()
    written = 0
    class_totals = np.zeros(4, dtype=np.int64)  # legit, trial, discount, fraud

    parquet_writer = None  # lazily created for the Parquet branch
    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Parquet output requires 'pyarrow'. Install it (e.g. "
                "`uv add pyarrow`) or use a .csv output path instead."
            ) from exc

    try:
        for block_index, size in enumerate(sizes):
            chunk = synthesize_payment_dataset(n=size, seed=seed + block_index)
            chunk = _relabel_transaction_ids(chunk, start=written)

            counts = chunk["label"].value_counts()
            for cls in range(4):
                class_totals[cls] += int(counts.get(cls, 0))

            if fmt == "csv":
                chunk.to_csv(
                    output_path,
                    mode="w" if block_index == 0 else "a",
                    header=block_index == 0,
                    index=False,
                )
            else:  # parquet
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_path, table.schema)
                parquet_writer.write_table(table)

            written += len(chunk)
            if verbose:
                elapsed = time.perf_counter() - started
                print(
                    f"  block {block_index + 1:>3}/{len(sizes)}  "
                    f"rows={written:>9,}/{n_rows:,}  "
                    f"elapsed={elapsed:6.1f}s",
                    flush=True,
                )
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if verbose:
        cfg = FeatureConfig()
        elapsed = time.perf_counter() - started
        size_mb = output_path.stat().st_size / (1024 * 1024)
        legit, trial, discount, fraud = class_totals
        print(
            "\nDone.\n"
            f"  file        : {output_path}\n"
            f"  format      : {fmt}\n"
            f"  rows        : {written:,}\n"
            f"  columns     : {len(PAYMENT_IDENTIFIER_COLUMNS) + len(cfg.payment_features) + 1} "
            f"({len(PAYMENT_IDENTIFIER_COLUMNS)} ids + {len(cfg.payment_features)} features + 1 label col)\n"
            f"  class 0 legit         : {legit:>9,}  ({legit / written:.4f})\n"
            f"  class 1 trial_abuse   : {trial:>9,}  ({trial / written:.4f})\n"
            f"  class 2 discount_abuse: {discount:>9,}  ({discount / written:.4f})\n"
            f"  class 3 payment_fraud : {fraud:>9,}  ({fraud / written:.4f})\n"
            f"  size on disk: {size_mb:,.1f} MB\n"
            f"  total time  : {elapsed:.1f}s"
        )
        if fmt == "csv":
            print(
                "\nNote: when re-reading this CSV, force identifier columns to string to\n"
                "avoid numeric-looking values being misread as floats, e.g.:\n"
                '  pd.read_csv(path, dtype={"user_id": str, "transaction_id": str,\n'
                '                            "organization_id": str})'
            )
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic FraudShield AI payment dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS, help="Total number of rows."
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output file path."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows generated/written per block (controls peak memory).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Base random seed."
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default=None,
        help="Force output format (default: inferred from the file suffix).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    generate_payment_data(
        n_rows=args.rows,
        output_path=args.output,
        chunk_size=args.chunk_size,
        seed=args.seed,
        file_format=args.format,
    )


if __name__ == "__main__":
    main()
