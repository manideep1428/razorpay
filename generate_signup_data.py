"""Generate a large synthetic FraudShield AI *signup* dataset.

Standalone data-generation utility. It reuses the project's schema-faithful
synthesizer (:func:`trust_radar.utils.synthetic.synthesize_signup_dataset`) so
the output carries every column of the FraudShield signup schema (identifiers
+ features + label bookkeeping) with the same label logic the models are
trained on.

The schema covers Tiers 1-3 (+5) only: IP reputation, email reputation,
browser/device metadata from the request itself, signup-velocity/history, and
optional phone -- no third-party canvas/audio/font fingerprinting SDK is
required or simulated.

Why chunked?
    1,000,000 rows x ~155 columns is too large to hold comfortably in memory as
    a single DataFrame. This script generates the data in blocks (default
    100,000 rows) and streams each block to disk, so peak memory stays at roughly
    one block. ``user_id`` is re-issued as a globally unique key across blocks.

Nothing runs on import. Generate via the CLI, e.g.::

    python generate_signup_data.py --rows 1000000 --output data/signup_dataset.csv
    python generate_signup_data.py --rows 1000000 --output data/signup_dataset.parquet

or import and call :func:`generate_signup_data` directly.

CSV note (``phone_number`` / ``email_address``):
    CSV has no per-column type metadata. Values like ``+7728649473`` *look*
    numeric to pandas' type sniffer and can be misread as scientific-notation
    floats on load. The file on disk is correct; when reading it back with
    pandas, force the column(s) to string, e.g.::

        pd.read_csv(path, dtype={"phone_number": str, "user_id": str})

    Parquet output does not have this problem (column types are stored in the
    file), so prefer ``--format parquet`` if this matters for your workflow.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from trust_radar.config import SIGNUP_IDENTIFIER_COLUMNS, FeatureConfig
from trust_radar.utils.synthetic import synthesize_signup_dataset

DEFAULT_ROWS = 1000
DEFAULT_CHUNK_SIZE = 100
DEFAULT_OUTPUT = Path("data") / "signup_dataset.csv"
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


def _relabel_user_ids(chunk: pd.DataFrame, start: int) -> pd.DataFrame:
    """Reassign ``user_id`` to a globally unique, zero-padded key.

    The per-block synthesizer numbers users from 0; this offsets them by the
    running row count so identifiers are unique across the whole file.
    ``user_id`` is a pure identifier (never a model feature), so rewriting it
    does not affect any feature or label.
    """
    stop = start + len(chunk)
    chunk["user_id"] = [f"u_{i:09d}" for i in range(start, stop)]
    return chunk


def generate_signup_data(
    n_rows: int = DEFAULT_ROWS,
    output_path: str | Path = DEFAULT_OUTPUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    seed: int = DEFAULT_SEED,
    file_format: str | None = None,
    verbose: bool = True,
) -> Path:
    """Generate a synthetic signup dataset and stream it to disk in chunks.

    Args:
        n_rows: Total number of signup rows to generate (e.g. 1_000_000).
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
    abuse_total = 0

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
            chunk = synthesize_signup_dataset(n=size, seed=seed + block_index)
            chunk = _relabel_user_ids(chunk, start=written)

            abuse_total += int(chunk["label"].sum())

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
        print(
            "\nDone.\n"
            f"  file        : {output_path}\n"
            f"  format      : {fmt}\n"
            f"  rows        : {written:,}\n"
            f"  columns     : {len(SIGNUP_IDENTIFIER_COLUMNS) + len(cfg.signup_features) + 3} "
            f"({len(SIGNUP_IDENTIFIER_COLUMNS)} ids + {len(cfg.signup_features)} features + 3 label cols)\n"
            f"  abuse rate  : {abuse_total / written:.4f}\n"
            f"  size on disk: {size_mb:,.1f} MB\n"
            f"  total time  : {elapsed:.1f}s"
        )
        if fmt == "csv":
            print(
                "\nNote: when re-reading this CSV, force text columns to string to avoid\n"
                "numeric-looking values (e.g. phone_number) being misread as floats:\n"
                '  pd.read_csv(path, dtype={"phone_number": str, "user_id": str})'
            )
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic FraudShield AI signup dataset.",
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
    generate_signup_data(
        n_rows=args.rows,
        output_path=args.output,
        chunk_size=args.chunk_size,
        seed=args.seed,
        file_format=args.format,
    )


if __name__ == "__main__":
    main()
