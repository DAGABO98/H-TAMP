from __future__ import annotations

import os
from typing import Any

DEFAULT_WANDB_INIT_TIMEOUT_SECONDS = 300.0


def _coerce_positive_float(raw_value: str | None, *, default: float) -> float:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def wandb_init_settings(wandb_module: Any) -> Any:
    init_timeout = _coerce_positive_float(
        os.getenv("WANDB_INIT_TIMEOUT"),
        default=DEFAULT_WANDB_INIT_TIMEOUT_SECONDS,
    )
    return wandb_module.Settings(init_timeout=init_timeout)
