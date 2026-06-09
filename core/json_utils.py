"""Helpers for JSON-safe numeric values."""

from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/inf and invalid inputs with default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number
