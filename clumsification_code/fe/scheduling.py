# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Training schedules expressed in examples rather than optimizer steps."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleSchedule:
    """Resolved step schedule and its effective global batch size."""

    global_batch_size: int
    save_steps: int
    eval_steps: int
    effective_save_examples: int
    effective_eval_examples: int


def resolve_example_schedule(
    *,
    save_every_examples: int | None,
    eval_every_examples: int | None,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> ExampleSchedule | None:
    """Convert compatible example intervals to optimizer-step intervals.

    The checkpoint interval must be an integer multiple of the evaluation
    interval, ensuring that every saved checkpoint has fresh validation metrics.
    """
    if save_every_examples is None and eval_every_examples is None:
        return None
    if save_every_examples is None or eval_every_examples is None:
        raise ValueError(
            "save_every_examples and eval_every_examples must be set together"
        )
    if save_every_examples < 1 or eval_every_examples < 1:
        raise ValueError("example intervals must be positive")
    if save_every_examples % eval_every_examples:
        raise ValueError(
            "save_every_examples must be an integer multiple of "
            "eval_every_examples"
        )
    if (
        per_device_train_batch_size < 1
        or gradient_accumulation_steps < 1
        or world_size < 1
    ):
        raise ValueError("batch size, gradient accumulation, and world size must be positive")

    global_batch_size = (
        per_device_train_batch_size * gradient_accumulation_steps * world_size
    )
    eval_steps = max(1, round(eval_every_examples / global_batch_size))
    save_steps = eval_steps * (save_every_examples // eval_every_examples)
    return ExampleSchedule(
        global_batch_size=global_batch_size,
        save_steps=save_steps,
        eval_steps=eval_steps,
        effective_save_examples=save_steps * global_batch_size,
        effective_eval_examples=eval_steps * global_batch_size,
    )
