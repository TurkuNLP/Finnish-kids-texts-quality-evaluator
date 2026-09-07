# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Early stopping that counts only evaluated checkpoint boundaries."""
from __future__ import annotations

import math
from typing import Any

from transformers import TrainerCallback


class CheckpointEarlyStoppingCallback(TrainerCallback):
    """Stop after consecutive non-improving saved checkpoints.

    Validation may run between checkpoints. Those measurements are retained by
    the Trainer, but only the evaluation at a save boundary changes patience.
    """

    def __init__(
        self,
        *,
        metric_name: str,
        patience: int,
        threshold: float,
        min_examples: int,
        global_batch_size: int,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if min_examples < 0:
            raise ValueError("min_examples must be non-negative")
        if global_batch_size < 1:
            raise ValueError("global_batch_size must be at least 1")
        self.metric_name = metric_name
        self.patience = patience
        self.threshold = threshold
        self.min_examples = min_examples
        self.global_batch_size = global_batch_size
        self.best_metric: float | None = None
        self.non_improving_checkpoints = 0
        self._metrics_by_step: dict[int, float] = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, control, kwargs
        value = (metrics or {}).get(self.metric_name)
        if value is not None and math.isfinite(float(value)):
            self._metrics_by_step[state.global_step] = float(value)

    def on_save(self, args, state, control, **kwargs):
        del args, kwargs
        value = self._metrics_by_step.pop(state.global_step, None)
        if value is None:
            return control
        examples_seen = state.global_step * self.global_batch_size
        if examples_seen < self.min_examples:
            return control
        if self.best_metric is None or value > self.best_metric + self.threshold:
            self.best_metric = value
            self.non_improving_checkpoints = 0
            return control

        self.non_improving_checkpoints += 1
        if self.non_improving_checkpoints >= self.patience:
            control.should_training_stop = True
        return control
