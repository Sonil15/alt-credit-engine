"""Cohort-aware imputation profile for missing data sources.

A thin-file borrower — or one who has revoked consent for a source — arrives with
whole feature groups absent. The models need *some* value in every column, but the
value we substitute is not neutral: it is what the model reads as fact. Filling an
absent source with ``0.0`` is directional — ``monthly_income_mean = 0`` reads as "no
income" (unfairly punishing), while ``missed_payments_count = 0`` reads as "perfect
history" (unfairly rewarding). Neither is what "we don't know" should mean.

This module instead learns, at training time, the *typical applicant* value for each
feature and persists it as an artifact. At serve time a missing feature is filled with
the median of the borrower's **own cohort**, so "unknown" resolves to "typical for
someone like you" rather than a biased extreme.

Two kinds of absence are distinguished automatically by the per-cohort median:

* **Applicable but not collected** (e.g. a genuine thin file missing cashflow) — the
  cohort has observed values, so we fill the cohort median.
* **Structurally not applicable** (e.g. business vintage for a salaried individual) —
  the cohort has *no* observed values (all-NaN), so the median is undefined and we
  fall back to ``0.0``, matching the correct real-world meaning ("no business").

Because the training population is dense, retraining is unaffected by this change: the
fill branch is only ever exercised at serve time for genuinely absent data, so the
committed models and their scores are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models_ai.constants import FEATURE_COLUMNS

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
IMPUTATION_PATH = ARTIFACT_DIR / "imputation_stats.json"

# Module-level cache; keyed on the artifact so a rebuild during the same process
# (e.g. a retrain) is picked up via ``save_imputation_stats``.
_CACHE: dict[str, Any] | None = None


def _cohort_key(code: Any) -> str | None:
    """Normalise a cohort_code cell to the string key used in the artifact."""
    try:
        val = float(code)
    except (TypeError, ValueError):
        return None
    if np.isnan(val):
        return None
    return str(int(val))


def _column_medians(frame: pd.DataFrame) -> dict[str, float | None]:
    """Median per FEATURE_COLUMN over a frame; ``None`` when a column is all-NaN."""
    out: dict[str, float | None] = {}
    for feat in FEATURE_COLUMNS:
        if feat not in frame.columns:
            out[feat] = None
            continue
        series = pd.to_numeric(frame[feat], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        median = series.median()
        out[feat] = None if pd.isna(median) else float(median)
    return out


def build_imputation_stats(wide: pd.DataFrame) -> dict[str, Any]:
    """Compute the global + per-cohort typical-applicant profile from training data."""
    stats: dict[str, Any] = {"global": _column_medians(wide), "by_cohort": {}}
    if "cohort_code" in wide.columns:
        for code, group in wide.groupby("cohort_code"):
            key = _cohort_key(code)
            if key is not None:
                stats["by_cohort"][key] = _column_medians(group)
    return stats


def save_imputation_stats(stats: dict[str, Any]) -> Path:
    """Persist the imputation profile and refresh the in-process cache."""
    global _CACHE
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    IMPUTATION_PATH.write_text(json.dumps(stats, indent=2))
    _CACHE = stats
    return IMPUTATION_PATH


def load_imputation_stats() -> dict[str, Any]:
    """Load the imputation profile, or an empty profile if the artifact is absent.

    An empty profile makes ``fill_missing_features`` fall back to ``0.0`` for every
    column — i.e. the historical behaviour — so nothing breaks before the artifact
    is built.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        _CACHE = json.loads(IMPUTATION_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        _CACHE = {"global": {}, "by_cohort": {}}
    return _CACHE


def invalidate_cache() -> None:
    """Drop the cached profile (call after an out-of-process rebuild)."""
    global _CACHE
    _CACHE = None


def imputation_fill_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-row fill value for every FEATURE_COLUMN, aligned to ``df.index``.

    Each row is imputed from *its own* cohort profile — this matters for mixed-cohort
    batches (training, population baselines), where a single shared profile would leak
    one cohort's typical value onto another. Resolution per (row, feature):

    * Row has a known cohort with a defined median → that cohort median.
    * Row has a known cohort but the median is undefined (feature structurally not
      applicable to the cohort, e.g. business vintage for a salaried applicant) → NaN,
      which the caller resolves to ``0.0`` — the correct real-world meaning. We do *not*
      borrow the global median here, which would fabricate a value.
    * Row has no identifiable cohort → the global median (best available fallback).

    NaN cells are left for the caller's ``0.0`` safety fill.
    """
    stats = load_imputation_stats()
    global_profile: dict[str, Any] = stats.get("global", {})
    by_cohort: dict[str, Any] = stats.get("by_cohort", {})

    if "cohort_code" in df.columns:
        keys = [_cohort_key(code) for code in df["cohort_code"].tolist()]
    else:
        keys = [None] * len(df)

    columns: dict[str, list[float]] = {}
    for feat in FEATURE_COLUMNS:
        values: list[float] = []
        for key in keys:
            if key is None:
                resolved = global_profile.get(feat)
            else:
                cohort_profile = by_cohort.get(key)
                resolved = cohort_profile.get(feat) if cohort_profile is not None else None
            values.append(np.nan if resolved is None else float(resolved))
        columns[feat] = values

    return pd.DataFrame(columns, index=df.index)
