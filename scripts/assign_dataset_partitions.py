# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Assign fixed source partitions to an existing canonical custom dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clumsification_code.data.io import sha256_file, write_json_atomic
from clumsification_code.data.partitioning import (
    apply_partition_plan,
    make_partition_plan,
    partition_manifest,
)
from clumsification_code.data.repository import DatasetRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Canonical custom-dataset name.")
    parser.add_argument("--dataset-root", default="data/custom_datasets")
    parser.add_argument("--dev-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--train-block-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite-partitions",
        action="store_true",
        help="Replace existing partition fields after explicit confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without changing original.jsonl or its manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DatasetRepository.from_root(args.dataset_root, args.dataset)
    originals = repository.read_originals()
    source_hash_before = sha256_file(repository.original_path)
    plan = make_partition_plan(
        originals,
        dev_size=args.dev_size,
        test_size=args.test_size,
        train_block_size=args.train_block_size,
        seed=args.seed,
    )
    manifest_path = repository.dataset_dir / "partition_manifest.json"
    preview = partition_manifest(
        plan,
        dataset_name=repository.dataset_name,
        seed=args.seed,
        dev_size=args.dev_size,
        test_size=args.test_size,
        train_block_size=args.train_block_size,
        original_sha256_before=source_hash_before,
    )
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    if manifest_path.exists() and not args.overwrite_partitions:
        raise FileExistsError(
            f"Partition manifest already exists: {manifest_path}; use "
            "--overwrite-partitions to replace it"
        )

    updated = apply_partition_plan(
        originals, plan, overwrite_partitions=args.overwrite_partitions
    )
    repository.write_originals(updated, overwrite=True)
    completed = partition_manifest(
        plan,
        dataset_name=repository.dataset_name,
        seed=args.seed,
        dev_size=args.dev_size,
        test_size=args.test_size,
        train_block_size=args.train_block_size,
        original_sha256_before=source_hash_before,
        original_sha256_after=sha256_file(repository.original_path),
    )
    write_json_atomic(manifest_path, completed, overwrite=args.overwrite_partitions)
    print(json.dumps(completed["counts"], sort_keys=True))
    print(f"Assigned source partitions: {repository.original_path}")
    print(f"Wrote partition manifest: {manifest_path}")


if __name__ == "__main__":
    main()
