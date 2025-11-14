#!/usr/bin/env python3
"""
SVM (with Optuna) for radiomics classification.

- Label y = 1 if "Time to First Progression (Days)" is non-NA, else 0
- Train set:  traincv_set__4_topk.xlsx  (hardcoded)
- Test set:   test_set__4_topk.xlsx     (hardcoded)
- Group-aware CV by subject_id (or Patient_ID fallback) to avoid leakage
- Objective: maximize OOF ROC AUC across CV folds
- Threshold: chosen on *calibrated* OOF probabilities to minimize FP+FN
- Saves artifact to: radiomics_svm.joblib

Outputs:
  - calibration_oof_vs_test__svm.png
  - brier_bars__svm.png
  - roc_plot_test__svm.png
  - feature_index_name_map__svm.csv
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import optuna

from pathlib import Path

# ==== sklearn ====
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold as SGKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss, roc_curve
)
from sklearn.linear_model import LogisticRegression   # external Platt
from sklearn.calibration import calibration_curve
from sklearn.svm import SVC
import argparse

# =================== HARD-CODED SETTINGS ===================
parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, required=True, help="Top-k features to use")
args = parser.parse_args()

TOPK_FEATURES = args.topk
OUTPUT_DIR = f"{TOPK_FEATURES}_features/"
TRAIN_XLSX = OUTPUT_DIR + "traincv_set__4_topk.xlsx"
TEST_XLSX  = OUTPUT_DIR + "test_set__4_topk.xlsx"
ARTIFACT   = OUTPUT_DIR + "radiomics_svm.joblib"
CALIBRATE  = True

RANDOM_STATE = 42
N_SPLITS     = 5   # CV folds within train-cv

# Optuna search space
N_TRIALS              = 300
C_MIN, C_MAX          = 1e-2, 1e3
GAMMA_MIN, GAMMA_MAX  = 1e-5, 1e0   # RBF gamma
CORR_MIN, CORR_MAX    = 0.75, 0.95  # correlation pruning

# Label & column config
PROG_COL = "Time to First Progression (Days)"
SUBJECT_COL_CANDIDATES = ["subject_id", "Patient_ID", "Subject_ID"]
EXCLUDE_COLS = {"subject_id", "Patient_ID", "Subject_ID",
                "timepoint", PROG_COL, "delta_days", "days_from_dx"}

DO_TRAIN = 1  # set to 1 to run Optuna + train, else only test + mapping

# =================== HELPERS ===================
def get_subject_col(df: pd.DataFrame) -> str:
    for c in SUBJECT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a subject column. Tried: {SUBJECT_COL_CANDIDATES}")

def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    from pandas.api.types import is_numeric_dtype
    if PROG_COL not in df.columns:
        raise ValueError(f"Label column '{PROG_COL}' not found in dataframe")
    y = df[PROG_COL].notna().astype(int).values
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS and is_numeric_dtype(df[c])]
    X = df[feat_cols].copy()
    X.columns = X.columns.astype(str)
    return X, y, feat_cols

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from any pair whose |corr| >= threshold (fit on input only)."""
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

def build_pipeline(C: float, gamma: float, corr_threshold: float = 0.80) -> Pipeline:
    """
    Preprocessing + SVM (RBF). We scale before SVM (critical).
    We keep SVC(probability=False) and use external Platt on OOF scores.
    """
    svm = SVC(
        kernel="rbf",
        C=float(C),
        gamma=float(gamma),
        class_weight="balanced",
        probability=False,         # external Platt on decision_function
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("vt",   VarianceThreshold(threshold=1e-12)),
        ("corr", CorrelationFilter(threshold=corr_threshold, method="spearman")),
        ("sc",   StandardScaler(with_mean=True, with_std=True)),
        ("clf",  svm),
    ])
    return pipe

def oof_cv_predict_scores(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
    pipe: Pipeline, n_splits: int, seed: int
):
    """
    Return OOF decision_function scores and fold AUCs
    (AUC is invariant to monotone transforms).
    """
    n = len(y)
    oof_scores = np.full(n, np.nan, dtype=float)
    aucs = []

    skf = SGKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y, groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        pipe.fit(Xtr, ytr)
        # signed distance to hyperplane
        s = pipe.decision_function(Xva).astype(float)
        oof_scores[va] = s

        if np.unique(yva).size == 2:
            aucs.append(roc_auc_score(yva, s))

    oof_auc = roc_auc_score(y, oof_scores) if np.unique(y).size == 2 else np.nan
    return oof_scores, oof_auc, aucs

def fit_platt_on_oof(y_true: np.ndarray, oof_scores: np.ndarray) -> LogisticRegression:
    """
    External Platt calibration on OOF decision scores:
        p = sigmoid(a * score + b)
    """
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(oof_scores.reshape(-1, 1), y_true.astype(int))
    return lr

def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict]:
    """Choose threshold that minimizes FP + FN."""
    vals = np.unique(proba)
    if vals.size > 1000:
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
            best_stats = {
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "errors": err, "accuracy": acc
            }
    return best_thr, best_stats

# =================== TRAIN & TEST ===================
def train_svm():
    np.random.seed(RANDOM_STATE)
    df = pd.read_excel(TRAIN_XLSX)
    subj_col = get_subject_col(df)
    df[subj_col] = df[subj_col].astype(str)
    X, y, feat_cols = get_xy(df)
    groups = df[subj_col].values

    def objective(trial: optuna.trial.Trial):
        C     = trial.suggest_float("C", C_MIN, C_MAX, log=True)
        gamma = trial.suggest_float("gamma", GAMMA_MIN, GAMMA_MAX, log=True)
        corr  = trial.suggest_float("corr_threshold", CORR_MIN, CORR_MAX)

        pipe = build_pipeline(C=C, gamma=gamma, corr_threshold=corr)
        oof_scores, oof_auc, fold_aucs = oof_cv_predict_scores(
            X, y, groups, pipe, N_SPLITS, RANDOM_STATE
        )

        # External Platt just for threshold selection (AUC unaffected)
        if CALIBRATE:
            calibrator = fit_platt_on_oof(y, oof_scores)
            oof_cal = calibrator.predict_proba(oof_scores.reshape(-1, 1))[:, 1]
            thr, stats = find_best_threshold(y, oof_cal)
        else:
            thr, stats = find_best_threshold(y, oof_scores)

        trial.set_user_attr("oof_auc", oof_auc)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("errors_fp_fn", stats.get("errors", None))
        return oof_auc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    best = study.best_trial
    best_params = {
        "C":             float(best.params["C"]),
        "gamma":         float(best.params["gamma"]),
        "corr_threshold":float(best.params["corr_threshold"]),
    }

    print("\n[optuna] Best AUC:", best.value)
    print("[optuna] Best params:", best_params)
    print("[optuna] OOF threshold (min FP+FN):", best.user_attrs.get("threshold"))
    print("[optuna] OOF FP+FN errors:", best.user_attrs.get("errors_fp_fn"))

    # Recompute OOF with best params & pick threshold (and final calibrator)
    best_pipe = build_pipeline(
        C=best_params["C"],
        gamma=best_params["gamma"],
        corr_threshold=best_params["corr_threshold"],
    )
    oof_scores, oof_auc, fold_aucs = oof_cv_predict_scores(
        X, y, groups, best_pipe, N_SPLITS, RANDOM_STATE
    )

    calibrator = None
    if CALIBRATE:
        calibrator = fit_platt_on_oof(y, oof_scores)
        oof_cal = calibrator.predict_proba(oof_scores.reshape(-1, 1))[:, 1]
        thr, stats = find_best_threshold(y, oof_cal)
        # Brier diagnostics on calibrated OOF
        try:
            cal_brier = brier_score_loss(y, oof_cal)
            print(f"[calibration] Platt on OOF: Brier (calibrated)={cal_brier:.4f}")
        except Exception:
            pass
    else:
        thr, stats = find_best_threshold(y, oof_scores)

    print("\n[final OOF] AUC:", round(oof_auc, 4),
          "| mean fold AUC:", round(float(np.mean(fold_aucs)), 4))
    print("[final OOF] threshold (chosen on calibrated OOF if CALIBRATE):",
          thr, "| stats:", stats)
    print(f"[VAL] Final OOF AUC = {oof_auc:.4f}")

    # Fit on ALL train-cv
    best_pipe.fit(X, y)

    # Save artifact
    artifact = {
        "pipeline": best_pipe,
        "threshold": float(thr),
        "params": best_params,
        "feature_cols": feat_cols,
        "subject_col": subj_col,
        "train_oof_auc": float(oof_auc),
        "train_oof_threshold_stats": stats,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "calibrate": bool(CALIBRATE),
        "platt_calibrator": calibrator,
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"[saved] {ARTIFACT}")

def test_svm():
    # Load artifact
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    thr = float(art["threshold"])
    feat_cols = list(art["feature_cols"])
    calibrate = bool(art.get("calibrate", False))
    platt = art.get("platt_calibrator", None)

    # Load test set
    df = pd.read_excel(TEST_XLSX)
    if PROG_COL not in df.columns:
        raise ValueError(f"Label column '{PROG_COL}' not found in test file")
    y = df[PROG_COL].notna().astype(int).values

    # Align features
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feat_cols].copy()

    # Predict: decision_function -> external Platt sigmoid -> proba
    scores_raw = pipe.decision_function(X).astype(float)
    if calibrate and platt is not None:
        proba = platt.predict_proba(scores_raw.reshape(-1, 1))[:, 1]
    else:
        # If no calibrator, map scores to [0,1] via logistic as a monotone proxy
        proba = 1.0 / (1.0 + np.exp(-scores_raw))

    pred = (proba >= thr).astype(int)

    # For calibration plots, also compute in-sample (train-cv) probs
    df_tr = pd.read_excel(TRAIN_XLSX)
    y_tr = df_tr[PROG_COL].notna().astype(int).values
    for c in feat_cols:
        if c not in df_tr.columns:
            df_tr[c] = np.nan
    X_tr = df_tr[feat_cols].copy()
    scores_tr = pipe.decision_function(X_tr).astype(float)
    proba_tr = (platt.predict_proba(scores_tr.reshape(-1, 1))[:, 1]
                if calibrate and platt is not None
                else 1.0 / (1.0 + np.exp(-scores_tr)))

    # ===== Calibration curves: Train-CV vs Test =====
    n_bins = 7
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Left: TRAIN-CV
    ax = axes[0]
    fr_tr, mp_tr = calibration_curve(y_tr, proba_tr, n_bins=n_bins, strategy="quantile")
    ax.plot(mp_tr, fr_tr, "o-", label="SVM (TrainCV)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title("Calibration — Train-CV (IN-SAMPLE)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.grid(True)
    ax.legend(loc="best")

    # Right: TEST
    ax = axes[1]
    fr_te, mp_te = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
    ax.plot(mp_te, fr_te, "o-", label="SVM (Test)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title("Calibration — Test")
    ax.set_xlabel("Mean predicted probability")
    ax.grid(True)
    ax.legend(loc="best")

    plt.suptitle("Calibration curves: Train-CV vs Test (SVM)")
    plt.tight_layout()
    plt.savefig("calibration_oof_vs_test__svm.png", dpi=150)
    plt.close()

    # ===== Brier scores =====
    brier_tr = brier_score_loss(y_tr, proba_tr)
    brier_te = brier_score_loss(y, proba)
    print(f"[brier] TrainCV={brier_tr:.4f} | Test={brier_te:.4f}")

    labels = ["TrainCV", "Test"]
    scores = [brier_tr, brier_te]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, scores)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Brier scores — SVM")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("brier_bars__svm.png", dpi=150)
    plt.close()

    # ===== ROC (Test) =====
    fpr, tpr, _ = roc_curve(y, proba)
    auc_test = roc_auc_score(y, proba)
    print(f"[test] ROC AUC: {auc_test:.4f}")
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"SVM (AUC={auc_test:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Test set (SVM)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_plot_test__svm.png", dpi=300)
    plt.close()

    # Confusion & report
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print("\n[test] Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    report = classification_report(
        y, pred, labels=[0, 1],
        target_names=["No progression (0)", "Progression (1)"],
        zero_division=0, digits=4
    )
    print("\n[test] classification report:\n" + report)
    print("\n[test] first 10 probabilities:", np.round(proba[:10], 4))

def export_feature_mapping_numbers(csv_path=OUTPUT_DIR + "feature_index_name_map__svm.csv"):
    """
    Build mapping AFTER training (no retrain):
    feature_id (0..P-1) -> radiomic feature name that survived VT + CorrFilter.
    """
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    feat_cols = art["feature_cols"]

    imp  = pipe.named_steps["imp"]
    vt   = pipe.named_steps["vt"]
    corr = pipe.named_steps["corr"]

    vt_mask = vt.get_support(indices=False)             # bool mask over feat_cols
    names_after_vt = np.array(feat_cols)[vt_mask]       # (K,)

    kept = corr.keep_features_
    if kept is None or len(kept) == 0:
        kept_idx = np.array([], dtype=int)
        kept_names = np.array([], dtype=str)
    elif isinstance(kept[0], (int, np.integer)):
        kept_idx = np.array(kept, dtype=int)
        kept_names = names_after_vt[kept_idx]
    else:
        kept_names = np.array([k for k in kept if k in names_after_vt])
        kept_idx = np.array(
            [np.where(names_after_vt == k)[0][0] for k in kept_names],
            dtype=int
        )

    orig_index_map = {name: i for i, name in enumerate(feat_cols)}
    orig_index = [orig_index_map[n] for n in kept_names]

    df_map = pd.DataFrame({
        "feature_id": np.arange(len(kept_names), dtype=int),
        "feature_name": kept_names,
        "vt_index": kept_idx,
        "orig_index": orig_index,
        "orig_col": [feat_cols[i] for i in orig_index],
    })
    df_map.to_csv(csv_path, index=False)
    print(f"[mapping] saved {csv_path} with {len(df_map)} rows.")

# =================== main ===================
if __name__ == "__main__":
    if DO_TRAIN:
        train_svm()
    test_svm()
    export_feature_mapping_numbers()

