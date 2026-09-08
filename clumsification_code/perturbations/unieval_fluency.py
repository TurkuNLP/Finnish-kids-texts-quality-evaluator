# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""UniEval's token-span disfluency transformation.

The edit loop is transcribed from ``pseudo_data_summ.py`` in the official
UniEval repository at commit d33e7b6cfebe97b2bafe435adbd818230d5a416a.
Only the surrounding data adapter differs: this module accepts one arbitrary
text instead of extracting the first three sentences of CNN/DailyMail input.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np


UNIEVAL_COMMIT = "d33e7b6cfebe97b2bafe435adbd818230d5a416a"
UNIEVAL_SOURCE = (
    "https://github.com/maszhongming/UniEval/blob/"
    f"{UNIEVAL_COMMIT}/pseudo_data_summ.py#L19-L57"
)

UNIEVAL_OPERATIONS = ("repetition", "deletion", "shuffle")


class UniEvalOperationUnavailable(RuntimeError):
    """A requested UniEval operation cannot make a substantive token edit."""


def _span_length(
    token_count: int,
    *,
    operation: str,
    numpy_rng: np.random.Generator,
) -> int:
    """Use UniEval's Poisson span draw, subject to substantive-edit minima."""
    minimum = 2 if operation == "shuffle" else 1
    if token_count < minimum:
        raise UniEvalOperationUnavailable(
            f"{operation} requires at least {minimum} tokens; found {token_count}"
        )
    return min(token_count, max(minimum, int(numpy_rng.poisson(5))))


def apply_unieval_operation(
    text: str,
    *,
    operation: str,
    python_rng: random.Random | Any | None = None,
    numpy_rng: np.random.Generator | None = None,
    max_shuffle_attempts: int = 10,
) -> tuple[str, dict]:
    """Apply one explicit UniEval corruption with a substantive token change.

    Span lengths, positions, copied-span selection, and token shuffling retain
    UniEval's original mechanics. The only deliberate deviations are minimum
    spans of one token for repetition/deletion, two for shuffle, and resampling
    an unchanged shuffle permutation.
    """
    if operation not in UNIEVAL_OPERATIONS:
        raise ValueError(
            f"Unknown UniEval operation {operation!r}; choose one of {UNIEVAL_OPERATIONS}"
        )
    if max_shuffle_attempts < 1:
        raise ValueError("max_shuffle_attempts must be at least 1")

    py_rng = python_rng or random
    np_rng = numpy_rng or np.random.default_rng()
    tokens = text.split()
    if not tokens:
        raise UniEvalOperationUnavailable("Cannot corrupt an empty token sequence")
    target_len = len(tokens)
    span_len = _span_length(target_len, operation=operation, numpy_rng=np_rng)
    start_idx = py_rng.randint(0, target_len - span_len)
    edit = {
        "transform_type": operation,
        "span_len": span_len,
        "start_idx": start_idx,
    }

    if operation == "repetition":
        copy_idx = py_rng.randint(0, target_len - span_len)
        edit["copy_idx"] = copy_idx
        result = (
            tokens[:start_idx]
            + tokens[copy_idx : copy_idx + span_len]
            + tokens[start_idx:]
        )
    elif operation == "deletion":
        result = tokens[:start_idx] + tokens[start_idx + span_len :]
        if not result:
            raise UniEvalOperationUnavailable(
                "Deletion would remove the entire token sequence"
            )
    else:
        original_span = tokens[start_idx : start_idx + span_len]
        result = None
        for shuffle_attempt in range(max_shuffle_attempts):
            shuffled_span = list(original_span)
            py_rng.shuffle(shuffled_span)
            if shuffled_span != original_span:
                result = tokens[:start_idx] + shuffled_span + tokens[start_idx + span_len :]
                edit["shuffle_attempts"] = shuffle_attempt + 1
                break
        if result is None:
            raise UniEvalOperationUnavailable(
                "Shuffle could not change the selected token span"
            )

    output = " ".join(result)
    if output == " ".join(tokens):
        raise UniEvalOperationUnavailable(
            f"{operation} did not change the token sequence"
        )
    return output, edit
