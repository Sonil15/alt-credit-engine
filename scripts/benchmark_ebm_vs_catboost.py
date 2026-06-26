"""Pre-decision benchmark: EBM (glass-box) vs CatBoost (current model).

Standalone — touches NOTHING in the live scoring path. It only reads ml_features
from the DB, trains both models out-of-fold on the SAME splits, and reports:

  1. AUC parity   -> "do we lose accuracy by going glass-box?" (one slide)
  2. Agreement    -> "how often do the two models reach the same decision?"

Why out-of-fold: with ~100 synthetic users, a single train/test split is pure
noise and in-sample agreement is trivially ~100%. We pool out-of-fold predictions
across a StratifiedKFold so every borrower is scored by a model that never saw it.
The folds are identical for both models, so the comparison is apples-to-apples.

Run:  .venv/bin/python -m scripts.benchmark_ebm_vs_catboost
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import pearsonr, spearmanr

from core.seeds import CATBOOST_RANDOM_SEED
from models_ai.catboost_model import train_catboost
from models_ai.constants import FEATURE_COLUMNS, LABEL_COLUMN, fill_missing_features
from models_ai.validation import (
    APPROVE_PD_THRESHOLD,
    REVIEW_PD_THRESHOLD,
    _gini,
    _ks_statistic,
)

N_SPLITS = 5
DB_PATH = Path(__file__).parent.parent / "alt_credit.db"
ARTIFACT_PATH = Path(__file__).parent.parent / "models_ai" / "artifacts" / "ebm_vs_catboost.json"


def _decision(pd_value: float) -> str:
    if pd_value <= APPROVE_PD_THRESHOLD:
        return "APPROVE"
    if pd_value <= REVIEW_PD_THRESHOLD:
        return "REVIEW"
    return "REJECT"


def _metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y, prob))
    return {"auc": round(auc, 4), "gini": round(_gini(auc), 4), "ks": round(_ks_statistic(y, prob), 4)}


def _load() -> tuple[pd.DataFrame, pd.Series]:
    """Read ml_features straight from SQLite and pivot to one row per user."""
    with sqlite3.connect(DB_PATH) as conn:
        long_df = pd.read_sql_query(
            "SELECT user_id, feature_name, feature_value, created_at FROM ml_features", conn
        )
    if long_df.empty:
        raise SystemExit(f"No ml_features in {DB_PATH}. Seed the project DB first.")
    latest = long_df.sort_values("created_at").groupby(["user_id", "feature_name"], as_index=False).last()
    wide = latest.pivot(index="user_id", columns="feature_name", values="feature_value").reset_index()
    wide.columns.name = None
    if LABEL_COLUMN not in wide.columns:
        raise SystemExit(f"No {LABEL_COLUMN} column found in ml_features.")
    # Only keep borrowers that actually have a ground-truth label (don't fillna to 0).
    wide = wide[wide[LABEL_COLUMN].notna()].reset_index(drop=True)
    y = wide[LABEL_COLUMN].astype(int)
    X = fill_missing_features(wide.copy())  # -> DataFrame[FEATURE_COLUMNS]
    return X, y


def run() -> dict:
    X, y = _load()
    n = len(y)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CATBOOST_RANDOM_SEED)

    oof_cb = np.full(n, np.nan)
    oof_ebm = np.full(n, np.nan)
    fold_auc_cb: list[float] = []
    fold_auc_ebm: list[float] = []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        cb = train_catboost(X_tr.reset_index(drop=True), label_series=y_tr.reset_index(drop=True))
        p_cb = cb.predict_proba(X_te)[:, 1]

        ebm = ExplainableBoostingClassifier(random_state=CATBOOST_RANDOM_SEED)
        ebm.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
        p_ebm = ebm.predict_proba(X_te)[:, 1]

        oof_cb[te], oof_ebm[te] = p_cb, p_ebm
        if y_te.nunique() > 1:  # AUC undefined on a single-class fold
            fold_auc_cb.append(roc_auc_score(y_te, p_cb))
            fold_auc_ebm.append(roc_auc_score(y_te, p_ebm))

    y_arr = y.to_numpy()
    cb_m, ebm_m = _metrics(y_arr, oof_cb), _metrics(y_arr, oof_ebm)

    dec_cb = [_decision(p) for p in oof_cb]
    dec_ebm = [_decision(p) for p in oof_ebm]
    three_band = float(np.mean([a == b for a, b in zip(dec_cb, dec_ebm)]))
    binary = float(np.mean([(a == "REJECT") == (b == "REJECT") for a, b in zip(dec_cb, dec_ebm)]))

    bands = ["APPROVE", "REVIEW", "REJECT"]
    crosstab = pd.crosstab(
        pd.Series(dec_cb, name="CatBoost"), pd.Series(dec_ebm, name="EBM")
    ).reindex(index=bands, columns=bands, fill_value=0)

    result = {
        "n_users": int(n),
        "default_rate": round(float(y.mean()), 4),
        "n_folds": N_SPLITS,
        "catboost": {
            **cb_m,
            "cv_auc_mean": round(float(np.mean(fold_auc_cb)), 4),
            "cv_auc_std": round(float(np.std(fold_auc_cb)), 4),
        },
        "ebm": {
            **ebm_m,
            "cv_auc_mean": round(float(np.mean(fold_auc_ebm)), 4),
            "cv_auc_std": round(float(np.std(fold_auc_ebm)), 4),
        },
        "agreement": {
            "three_band_rate": round(three_band, 4),
            "approve_vs_reject_rate": round(binary, 4),
            "pd_pearson": round(float(pearsonr(oof_cb, oof_ebm)[0]), 4),
            "pd_spearman": round(float(spearmanr(oof_cb, oof_ebm)[0]), 4),
            "crosstab": crosstab.to_dict(),
        },
    }

    _print(result, crosstab)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved -> {ARTIFACT_PATH}")
    return result


def _print(r: dict, crosstab: pd.DataFrame) -> None:
    cb, ebm, ag = r["catboost"], r["ebm"], r["agreement"]
    print("\n" + "=" * 60)
    print(f"  EBM vs CatBoost  |  {r['n_users']} users  |  default rate {r['default_rate']:.0%}")
    print(f"  Out-of-fold, {r['n_folds']}-fold stratified CV (identical splits)")
    print("=" * 60)
    print(f"\n  {'metric':<22}{'CatBoost':>12}{'EBM (glass-box)':>18}")
    print(f"  {'-'*52}")
    print(f"  {'OOF AUC':<22}{cb['auc']:>12}{ebm['auc']:>18}")
    print(f"  {'OOF Gini':<22}{cb['gini']:>12}{ebm['gini']:>18}")
    print(f"  {'OOF KS':<22}{cb['ks']:>12}{ebm['ks']:>18}")
    print(f"  {'CV AUC (mean)':<22}{cb['cv_auc_mean']:>12}{ebm['cv_auc_mean']:>18}")
    print(f"  {'CV AUC (std)':<22}{cb['cv_auc_std']:>12}{ebm['cv_auc_std']:>18}")
    print(f"\n  Agreement (out-of-fold):")
    print(f"    APPROVE/REVIEW/REJECT match : {ag['three_band_rate']:.0%}")
    print(f"    lend / no-lend match        : {ag['approve_vs_reject_rate']:.0%}")
    print(f"    PD correlation (Spearman)   : {ag['pd_spearman']:.3f}")
    print(f"    PD correlation (Pearson)    : {ag['pd_pearson']:.3f}")
    print(f"\n  Decision cross-tab (rows=CatBoost, cols=EBM):")
    print("    " + crosstab.to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    run()
