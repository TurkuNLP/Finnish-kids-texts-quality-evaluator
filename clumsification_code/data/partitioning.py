# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Deterministic source-level partitions for staged perturbation experiments."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random
from typing import Iterable

from .io import canonical_json_hash
from .schemas import OriginalRecord


PARTITION_FIELD = "partition"
PARTITION_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class PartitionPlan:
    """A complete, source-disjoint partition assignment."""

    assignments: dict[str, str]
    counts: dict[str, int]


def _partition_labels(
    *,
    total: int,
    dev_size: int,
    test_size: int,
    train_block_size: int,
) -> list[str]:
    if dev_size < 1 or test_size < 1 or train_block_size < 1:
        raise ValueError("dev_size, test_size, and train_block_size must be positive")
    if dev_size + test_size >= total:
        raise ValueError("dev_size plus test_size must leave at least one training source")

    labels = ["dev"] * dev_size + ["test"] * test_size
    remaining = total - len(labels)
    block_index = 1
    while remaining >= train_block_size:
        labels.extend([f"train_{block_index:02d}"] * train_block_size)
        remaining -= train_block_size
        block_index += 1
    if remaining:
        labels.extend(["train_remainder"] * remaining)
    return labels


def _length_stratified_order(
    records: Iterable[OriginalRecord], *, seed: int, strata: int = 10
) -> list[OriginalRecord]:
    """Return a seeded ordering whose prefixes represent every length decile."""
    values = sorted(records, key=lambda record: (len(record.text), record.base_text_id))
    if not values:
        raise ValueError("Cannot partition an empty original dataset")
    bins = [
        values[index * len(values) // strata:(index + 1) * len(values) // strata]
        for index in range(strata)
    ]
    rng = random.Random(seed)
    for value in bins:
        rng.shuffle(value)
    queues = [deque(value) for value in bins]
    ordered: list[OriginalRecord] = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
    return ordered


def make_partition_plan(
    records: Iterable[OriginalRecord],
    *,
    dev_size: int = 10_000,
    test_size: int = 10_000,
    train_block_size: int = 50_000,
    seed: int = 42,
) -> PartitionPlan:
    """Assign every canonical source to a fixed dev/test/train partition."""
    ordered = _length_stratified_order(records, seed=seed)
    labels = _partition_labels(
        total=len(ordered),
        dev_size=dev_size,
        test_size=test_size,
        train_block_size=train_block_size,
    )
    assignments = {
        record.base_text_id: label for record, label in zip(ordered, labels, strict=True)
    }
    return PartitionPlan(assignments=assignments, counts=dict(sorted(Counter(labels).items())))


def apply_partition_plan(
    records: Iterable[OriginalRecord],
    plan: PartitionPlan,
    *,
    overwrite_partitions: bool = False,
) -> tuple[OriginalRecord, ...]:
    """Attach the plan to originals without changing their identities or text."""
    values = tuple(records)
    expected_ids = {record.base_text_id for record in values}
    if set(plan.assignments) != expected_ids:
        raise ValueError("Partition plan IDs do not exactly match original records")
    existing = [
        record.base_text_id for record in values if PARTITION_FIELD in record.metadata
    ]
    if existing and not overwrite_partitions:
        raise ValueError(
            "Original records already have partition assignments; use "
            "--overwrite-partitions to replace them"
        )
    return tuple(
        OriginalRecord(
            dataset_name=record.dataset_name,
            base_text_id=record.base_text_id,
            text=record.text,
            metadata={**record.metadata, PARTITION_FIELD: plan.assignments[record.base_text_id]},
        )
        for record in values
    )


def partition_manifest(
    plan: PartitionPlan,
    *,
    dataset_name: str,
    seed: int,
    dev_size: int,
    test_size: int,
    train_block_size: int,
    original_sha256_before: str,
    original_sha256_after: str | None = None,
) -> dict:
    """Return stable audit metadata without duplicating every assigned ID."""
    ids_by_partition: dict[str, list[str]] = {}
    for source_id, partition in plan.assignments.items():
        ids_by_partition.setdefault(partition, []).append(source_id)
    return {
        "schema_version": PARTITION_MANIFEST_VERSION,
        "dataset_name": dataset_name,
        "partition_field": PARTITION_FIELD,
        "seed": seed,
        "stratification": "document_character_length_decile",
        "dev_size": dev_size,
        "test_size": test_size,
        "train_block_size": train_block_size,
        "counts": plan.counts,
        "partition_id_hashes": {
            partition: canonical_json_hash(sorted(ids))
            for partition, ids in sorted(ids_by_partition.items())
        },
        "original_sha256_before": original_sha256_before,
        "original_sha256_after": original_sha256_after,
    }


__all__ = [
    "PARTITION_FIELD",
    "PARTITION_MANIFEST_VERSION",
    "PartitionPlan",
    "apply_partition_plan",
    "make_partition_plan",
    "partition_manifest",
]
