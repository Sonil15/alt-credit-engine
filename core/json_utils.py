"""Helpers for JSON-safe numeric values."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/inf and invalid inputs with default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        # numpy floats subclass float; coerce to a plain float so strict JSON
        # encoders (json.dumps) accept the result.
        return float(value) if math.isfinite(value) else default
    if isinstance(value, Real):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return number if math.isfinite(number) else default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_round(value: Any, ndigits: int = 0, default: float = 0.0) -> float:
    """Round a JSON-safe float."""
    return round(safe_float(value, default=default), ndigits)


def sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-finite floats so strict JSON encoders succeed."""
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_json(item) for item in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return safe_float(value)
    if isinstance(value, Real):
        return safe_float(value)
    return value
