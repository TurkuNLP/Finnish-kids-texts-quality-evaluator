# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Deterministic sampling for LLM perturbation edit catalogs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence


SEVERITIES = ("weak", "medium", "strong")
ABSOLUTE_MAX_EDIT_COUNT = 5
_REQUIRED_FIELDS = {
    "edit_id",
    "target_dimensions",
    "edit_type",
    "example_edited",
    "example_clean",
}


@dataclass(frozen=True)
class EditCatalogEntry:
    edit_id: str
    target_dimensions: tuple[str, ...]
    edit_type: str
    example_edited: str
    example_clean: str
    instruction: str | None = None
    minimum_realization: str | None = None
    non_examples: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampledEditAssignment:
    target_dimensions: tuple[str, ...]
    edits: tuple[EditCatalogEntry, ...]
    severity: str
    seed: int


def _as_nonempty_string(value: object, field: str, line_no: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Catalog line {line_no}: {field} must be a non-empty string")
    return value


def _as_string_tuple(value: object, field: str, line_no: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Catalog line {line_no}: {field} must be a non-empty array")
    result = tuple(_as_nonempty_string(item, field, line_no) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"Catalog line {line_no}: {field} contains duplicates")
    return result


def _optional_string(value: object, field: str, line_no: int) -> str | None:
    if value is None:
        return None
    return _as_nonempty_string(value, field, line_no)


def _optional_string_tuple(value: object, field: str, line_no: int) -> tuple[str, ...]:
    if value is None:
        return ()
    return _as_string_tuple(value, field, line_no)


def load_edit_catalog(path: str | Path) -> tuple[EditCatalogEntry, ...]:
    """Load and validate a JSONL edit catalog in stable file order."""
    catalog_path = Path(path)
    entries: list[EditCatalogEntry] = []
    seen_ids: set[str] = set()
    with catalog_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Catalog line {line_no} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Catalog line {line_no} must be a JSON object")
            missing = sorted(_REQUIRED_FIELDS - set(value))
            if missing:
                raise ValueError(f"Catalog line {line_no} is missing fields: {missing}")
            edit_id = _as_nonempty_string(value["edit_id"], "edit_id", line_no)
            if edit_id in seen_ids:
                raise ValueError(f"Duplicate edit_id {edit_id!r} at catalog line {line_no}")
            seen_ids.add(edit_id)
            entries.append(
                EditCatalogEntry(
                    edit_id=edit_id,
                    target_dimensions=_as_string_tuple(value["target_dimensions"], "target_dimensions", line_no),
                    edit_type=_as_nonempty_string(value["edit_type"], "edit_type", line_no),
                    example_edited=_as_nonempty_string(value["example_edited"], "example_edited", line_no),
                    example_clean=_as_nonempty_string(value["example_clean"], "example_clean", line_no),
                    instruction=_optional_string(value.get("instruction"), "instruction", line_no),
                    minimum_realization=_optional_string(value.get("minimum_realization"), "minimum_realization", line_no),
                    non_examples=_optional_string_tuple(value.get("non_examples"), "non_examples", line_no),
                    applicability=_optional_string_tuple(value.get("applicability"), "applicability", line_no),
                )
            )
    if not entries:
        raise ValueError(f"Edit catalog is empty: {catalog_path}")
    return tuple(entries)


def _dimension_key(value: str) -> str:
    return " ".join(value.casefold().split())


def sample_edit_count(
    text_length: int,
    *,
    seed: int,
    max_edits: int | None = None,
) -> int:
    """Sample a length-scaled edit count from a truncated normal distribution.

    The range is one through ``min(5, floor(text_length / 500))``. Short
    texts therefore still receive one edit. ``max_edits`` can impose a lower
    caller-specific cap when a method cannot realize the full shared range.
    """
    if text_length < 0:
        raise ValueError("text_length must be non-negative")
    if max_edits is not None and max_edits < 1:
        raise ValueError("max_edits must be at least 1 when provided")

    upper = min(ABSOLUTE_MAX_EDIT_COUNT, max(1, text_length // 500))
    if max_edits is not None:
        upper = min(upper, max_edits)
    if upper == 1:
        return 1

    mean = (1 + upper) / 2
    standard_deviation = (upper - 1) / 6
    rng = random.Random(seed)
    while True:
        draw = rng.normalvariate(mean, standard_deviation)
        if 1 <= draw <= upper:
            return min(upper, max(1, int(draw + 0.5)))


def sample_target_dimensions(
    catalog: Sequence[EditCatalogEntry],
    *,
    n_dimensions: int,
    seed: int,
) -> tuple[str, ...]:
    """Uniformly sample distinct target dimensions represented in a catalog."""
    if n_dimensions < 1:
        raise ValueError("n_dimensions must be at least 1")
    dimensions_by_key: dict[str, str] = {}
    for entry in catalog:
        for dimension in entry.target_dimensions:
            dimensions_by_key.setdefault(_dimension_key(dimension), dimension)
    dimensions = tuple(dimensions_by_key[key] for key in sorted(dimensions_by_key))
    if n_dimensions > len(dimensions):
        raise ValueError(
            f"Requested {n_dimensions} target dimensions but catalog has only "
            f"{len(dimensions)}"
        )
    return tuple(random.Random(seed).sample(dimensions, n_dimensions))


def sample_dimension_count(
    *,
    n_edits: int,
    available_dimensions: int,
    seed: int,
) -> int:
    """Sample how many dimensions a candidate targets for its edit count."""
    if n_edits < 1:
        raise ValueError("n_edits must be at least 1")
    if available_dimensions < 1:
        raise ValueError("available_dimensions must be at least 1")
    return random.Random(seed).randint(1, min(n_edits, available_dimensions))


def _weighted_choice_without_replacement(
    candidates: list[EditCatalogEntry],
    count: int,
    rng: random.Random,
    weights: Mapping[str, float] | None,
) -> list[EditCatalogEntry]:
    remaining = list(candidates)
    selected: list[EditCatalogEntry] = []
    for _ in range(count):
        if not remaining:
            break
        if weights is None:
            index = rng.randrange(len(remaining))
        else:
            values = [float(weights.get(entry.edit_id, 0.0)) for entry in remaining]
            if any(value < 0 for value in values) or not any(values):
                raise ValueError("Operation weights must be non-negative and not all zero")
            threshold = rng.random() * sum(values)
            index = 0
            for index, weight in enumerate(values):
                threshold -= weight
                if threshold <= 0:
                    break
        selected.append(remaining.pop(index))
    return selected


def sample_edit_types(
    catalog: Sequence[EditCatalogEntry],
    *,
    target_dimensions: Sequence[str],
    n_edits: int,
    seed: int,
    weights: Mapping[str, float] | None = None,
    require_dimension_coverage: bool = True,
) -> tuple[EditCatalogEntry, ...]:
    """Sample edit assignments deterministically, optionally covering dimensions.

    Operations are sampled without replacement until the compatible catalog is
    exhausted, then with replacement. This permits long texts to receive more
    edits than one selected dimension has distinct catalog entries.
    """
    if n_edits < 1:
        raise ValueError("n_edits must be at least 1")
    dimensions = tuple(dict.fromkeys(_dimension_key(value) for value in target_dimensions if value.strip()))
    if not dimensions:
        raise ValueError("target_dimensions must contain at least one dimension")
    candidates = sorted(
        (entry for entry in catalog if any(_dimension_key(dim) in dimensions for dim in entry.target_dimensions)),
        key=lambda entry: entry.edit_id,
    )
    rng = random.Random(seed)
    selected: list[EditCatalogEntry] = []
    remaining = list(candidates)
    if require_dimension_coverage:
        uncovered = set(dimensions)
        while uncovered and remaining and len(selected) < n_edits:
            best_score = max(
                len(uncovered & {_dimension_key(dim) for dim in entry.target_dimensions})
                for entry in remaining
            )
            if best_score == 0:
                break
            best = [
                entry for entry in remaining
                if len(uncovered & {_dimension_key(dim) for dim in entry.target_dimensions}) == best_score
            ]
            choice = _weighted_choice_without_replacement(best, 1, rng, weights)[0]
            selected.append(choice)
            remaining.remove(choice)
            uncovered -= {_dimension_key(dim) for dim in choice.target_dimensions}

    while len(selected) < n_edits:
        selected.append(
            _weighted_choice_without_replacement(candidates, 1, rng, weights)[0]
        )
    return tuple(selected)


def sample_severity(
    severity: str | Sequence[str] | None,
    *,
    seed: int,
) -> str:
    """Resolve a fixed severity or sample one deterministically."""
    if severity is None:
        return "medium"
    if isinstance(severity, str):
        value = severity.casefold()
        if value not in SEVERITIES:
            raise ValueError(f"Unknown severity {severity!r}; choose one of {SEVERITIES}")
        return value
    choices = tuple(str(value).casefold() for value in severity)
    if not choices or any(value not in SEVERITIES for value in choices):
        raise ValueError(f"Severity choices must be drawn from {SEVERITIES}")
    return random.Random(seed).choice(choices)


def sample_edit_assignment(
    catalog: Sequence[EditCatalogEntry],
    *,
    target_dimensions: Sequence[str],
    n_edits: int,
    severity: str | Sequence[str] | None = "medium",
    seed: int = 0,
    weights: Mapping[str, float] | None = None,
    require_dimension_coverage: bool = True,
) -> SampledEditAssignment:
    """Sample the complete operation assignment for one candidate."""
    edits = sample_edit_types(
        catalog,
        target_dimensions=target_dimensions,
        n_edits=n_edits,
        seed=seed,
        weights=weights,
        require_dimension_coverage=require_dimension_coverage,
    )
    resolved_dimensions = tuple(dict.fromkeys(str(value) for value in target_dimensions))
    return SampledEditAssignment(
        target_dimensions=resolved_dimensions,
        edits=edits,
        severity=sample_severity(severity, seed=seed + 1),
        seed=seed,
    )


__all__ = [
    "ABSOLUTE_MAX_EDIT_COUNT",
    "SEVERITIES",
    "EditCatalogEntry",
    "SampledEditAssignment",
    "load_edit_catalog",
    "sample_edit_assignment",
    "sample_edit_count",
    "sample_edit_types",
    "sample_dimension_count",
    "sample_severity",
    "sample_target_dimensions",
]
