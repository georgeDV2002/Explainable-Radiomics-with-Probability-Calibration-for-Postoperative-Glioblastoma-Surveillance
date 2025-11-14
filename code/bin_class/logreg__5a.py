#!/usr/bin/env python3
"""
Elastic Net Logistic (with Optuna) for radiomics classification.

- Uses the same label definition as your exploration:
    label y = 1 if "Time to First Progression (Days)" is non-NA, else 0
- Train set:  dataset__2_traincv.xlsx (hardcoded)
- Test set:   dataset__2_test.xlsx     (hardcoded)
- Group-aware CV by subject_id (or Patient_ID fallback) to avoid leakage
- Objective: maximize OOF ROC AUC across CV folds
- Threshold: chosen on OOF predictions to minimize FP+FN
- Saves best artifact to: radiomics_logreg_enet.joblib

Functions to call:
    train_logistic()  -> runs Optuna, saves artifact, prints CV metrics & threshold
    test_logistic()   -> loads artifact, evaluates on test set, prints metrics

Requirements: scikit-learn, optuna, pandas, numpy, joblib
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# ==== sklearn ====
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
import optuna
from sklearn.model_selection import StratifiedGroupKFold as SGKFold  # sklearn >= 1.3
import argparse

# =================== HARD-CODED SETTINGS ===================
parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, required=True, help="Top-k features to use")
args = parser.parse_args()

TOPK_FEATURES = args.topk
OUTPUT_DIR = f"{TOPK_FEATURES}_features/"
TRAIN_XLSX = OUTPUT_DIR + "traincv_set__4_topk.xlsx"
TEST_XLSX  = OUTPUT_DIR + "test_set__4_topk.xlsx"
ARTIFACT   = OUTPUT_DIR + "radiomics_logreg_enet.joblib"

RANDOM_STATE = 42
N_SPLITS = 5  # CV folds within train-cv
MAX_ITER = 10000

# Hyperparameter search space
C_MIN, C_MAX = 1e-3, 100.0
L1R_MIN, L1R_MAX = 0.0, 1.0
CORR_MIN, CORR_MAX = 0.75, 0.95
N_TRIALS = 500

# Label & column config
PROG_COL = "Time to First Progression (Days)"
SUBJECT_COL_CANDIDATES = ["subject_id", "Patient_ID", "Subject_ID"]
EXCLUDE_COLS = {"subject_id", "Patient_ID", "Subject_ID", "timepoint", PROG_COL, "delta_days", "days_from_dx"}

TINY = 1e-9
DO_TRAIN = 1

# =================== HELPERS ===================

def get_subject_col(df: pd.DataFrame) -> str:
    for c in SUBJECT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a subject column. Tried: {SUBJECT_COL_CANDIDATES}")


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    from pandas.api.types import is_numeric_dtype
    # Label
    if PROG_COL not in df.columns:
        raise ValueError(f"Label column '{PROG_COL}' not found in dataframe")
    y = df[PROG_COL].notna().astype(int).values

    # Features = numeric columns not in EXCLUDE_COLS
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and is_numeric_dtype(df[c])]
    X = df[feat_cols].copy()
    X.columns = X.columns.astype(str)
    return X, y, feat_cols


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from any pair whose |corr| >= threshold (fit on input only).
    Works on numpy arrays or DataFrames. Does not need names for inference.
    """
    def __init__(self, threshold: float = 0.80, method: str = "spearman"):
        self.threshold = threshold
        self.method = method
        self.keep_features_: list[int] | list[str] | None = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X).copy()
        med = pd.Series(np.nanmedian(Xdf.values, axis=0), index=Xdf.columns)
        Xdf = Xdf.fillna(med)
        valid = Xdf.notna().sum(axis=0) >= 2
        Xdf = Xdf.loc[:, valid]
        corr = Xdf.corr(method=self.method).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = set()
        for col in upper.columns:
            if col in drop:
                continue
            high = upper.index[upper[col] >= self.threshold].tolist()
            drop.update(high)
        self.keep_features_ = [c for c in Xdf.columns if c not in drop]
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X)
        if self.keep_features_ is None:
            return Xdf.values
        return Xdf[self.keep_features_].values


def build_pipeline(C: float, l1_ratio: float, corr_threshold: float) -> Pipeline:
    pipe = Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("vt",   VarianceThreshold(threshold=1e-8)),
        ("corr", CorrelationFilter(threshold=corr_threshold, method="spearman")),
        ("sc",   StandardScaler()),
        ("clf",  LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, C=C,
            max_iter=MAX_ITER, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    return pipe


def oof_cv_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, pipe: Pipeline, n_splits: int, seed: int):
    """Return OOF probabilities, fold AUC list, and fitted models (optional)."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    aucs = []

    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = skf.split(X, y)
    else:
        skf = SGKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = skf.split(X, y, groups)

    for tr, va in split_iter:
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        oof[va] = p

        # AUC only if both classes exist in val
        if np.unique(yva).size == 2:
            aucs.append(roc_auc_score(yva, p))

    # Global OOF AUC (only if both classes present overall, which they are)
    oof_auc = roc_auc_score(y, oof) if np.unique(y).size == 2 else np.nan
    return oof, oof_auc, aucs


def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict]:
    """Choose threshold that minimizes FP + FN on the given set.
    Returns (best_thr, metrics_dict).
    """
    # Grid over unique probabilities (capped) or percentiles for stability
    vals = np.unique(proba)
    if vals.size > 1000:
        # use percentiles for speed
        thr_grid = np.quantile(proba, np.linspace(0.01, 0.99, 199))
        thr_grid = np.unique(thr_grid)
    else:
        thr_grid = vals

    best_thr, best_err = 0.5, float("inf")
    best_stats = {}
    n = len(y_true)

    for t in thr_grid:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        err = fp + fn
        if err < best_err:
            best_err = err
            best_thr = float(t)
            acc = (tp + tn) / max(1, n)
            best_stats = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": err, "accuracy": acc}

    return best_thr, best_stats


# =================== TRAIN & TEST ===================

def train_logistic():
    np.random.seed(RANDOM_STATE)

    # Load train-cv
    df = pd.read_excel(TRAIN_XLSX)
    subj_col = get_subject_col(df)
    df[subj_col] = df[subj_col].astype(str)
    X, y, feat_cols = get_xy(df)
    groups = df[subj_col].values

    def objective(trial: optuna.trial.Trial):
        C = trial.suggest_float("C", C_MIN, C_MAX, log=True)
        l1r = trial.suggest_float("l1_ratio", L1R_MIN, L1R_MAX)
        corr = trial.suggest_float("corr_threshold", CORR_MIN, CORR_MAX)
        pipe = build_pipeline(C=C, l1_ratio=l1r, corr_threshold=corr)
        oof, oof_auc, fold_aucs = oof_cv_predict(X, y, groups, pipe, N_SPLITS, RANDOM_STATE)
        # Store threshold and FP+FN on OOF for monitoring
        thr, stats = find_best_threshold(y, oof)
        trial.set_user_attr("oof_auc", oof_auc)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("errors_fp_fn", stats.get("errors", None))
        return oof_auc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # You can adjust n_trials as needed
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)
    best = study.best_trial
    best_params = {
        "C": best.params["C"],
        "l1_ratio": best.params["l1_ratio"],
        "corr_threshold": best.params["corr_threshold"],
    }

    print("\n[optuna] Best AUC:", best.value)
    print("[optuna] Best params:", best_params)
    print("[optuna] OOF threshold (min FP+FN):", best.user_attrs.get("threshold"))
    print("[optuna] OOF FP+FN errors:", best.user_attrs.get("errors_fp_fn"))

    # Recompute OOF with best params for a clean record & select threshold there
    best_pipe = build_pipeline(**best_params)
    oof, oof_auc, fold_aucs = oof_cv_predict(X, y, groups, best_pipe, N_SPLITS, RANDOM_STATE)
    thr, stats = find_best_threshold(y, oof)

    print("\n[final OOF] AUC:", round(oof_auc, 4), "| mean fold AUC:", round(float(np.mean(fold_aucs)), 4))
    print("[final OOF] threshold:", thr, "| stats:", stats)

    # Fit best pipeline on ALL train-cv data
    best_pipe.fit(X, y)

    # Save artifact
    artifact = {
        "pipeline": best_pipe,
        "threshold": thr,
        "params": best_params,
        "feature_cols": feat_cols,
        "subject_col": subj_col,
        "train_oof_auc": float(oof_auc),
        "train_oof_threshold_stats": stats,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"[saved] {ARTIFACT}")

def test_logistic():
    # Load artifact
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    thr = float(art["threshold"])
    feat_cols = list(art["feature_cols"])

    # Load test set
    df = pd.read_excel(TEST_XLSX)
    if PROG_COL not in df.columns:
        raise ValueError(f"Label column '{PROG_COL}' not found in test file")
    y = df[PROG_COL].notna().astype(int).values

    # Build features with the SAME columns used in training
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feat_cols].copy()

    # Predict
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    # ROC AUC
    auc_val = roc_auc_score(y, proba) if np.unique(y).size == 2 else None
    print(f"[test] ROC AUC: {auc_val:.4f}" if auc_val is not None else "[test] ROC AUC: N/A")

    # Confusion matrix
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print("\n[test] Confusion Matrix:")
    print(cm)

    # Classification report
    report = classification_report(
        y,
        pred,
        labels=[0, 1],
        target_names=["No progression (0)", "Progression (1)"],
        zero_division=0,
        digits=4
    )
    print("\n[test] classification report:\n" + report)

# Example usage (toggle as you wish)
if __name__ == "__main__":
    if DO_TRAIN:
        train_logistic()
    test_logistic()

