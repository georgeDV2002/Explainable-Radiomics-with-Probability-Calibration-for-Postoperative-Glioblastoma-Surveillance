#!/usr/bin/env python3
"""
LightGBM (with Optuna) for radiomics classification.

- Label y = 1 if "Time to First Progression (Days)" is non-NA, else 0
- Train set:  traincv_set__4_topk.xlsx  (hardcoded)
- Test set:   test_set__4_topk.xlsx     (hardcoded)
- Group-aware CV by subject_id (or Patient_ID fallback) to avoid leakage
- Objective: maximize OOF ROC AUC across CV folds
- Threshold: chosen on OOF predictions to minimize FP+FN
- Saves best artifact to: radiomics_lgbm.joblib

Functions:
    train_lgbm()  -> runs Optuna, saves artifact, prints CV metrics & threshold
    test_lgbm()   -> loads artifact, evaluates on test set (ROC AUC, CM, report)

Requirements: lightgbm, scikit-learn, optuna, pandas, numpy, joblib
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import optuna

from pathlib import Path

# ==== sklearn ====
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold as SGKFold  # sklearn >= 1.3
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression  # <-- Platt (sigmoid) calibrator
from sklearn.calibration import calibration_curve
 
from lightgbm import LGBMClassifier
import argparse

# =================== HARD-CODED SETTINGS ===================
parser = argparse.ArgumentParser()
parser.add_argument("--topk", type=int, required=True, help="Top-k features to use")
args = parser.parse_args()

TOPK_FEATURES = args.topk
OUTPUT_DIR = f"{TOPK_FEATURES}_features/"
TRAIN_XLSX = OUTPUT_DIR + "traincv_set__4_topk.xlsx"
TEST_XLSX  = OUTPUT_DIR +"test_set__4_topk.xlsx"
ARTIFACT   = OUTPUT_DIR +"radiomics_lgbm.joblib"
CALIBRATE = True

RANDOM_STATE = 42
N_SPLITS = 5  # CV folds within train-cv

# Hyperparameter search space
N_TRIALS = 400
LEARNING_RATE_MIN, LEARNING_RATE_MAX = 0.01, 0.3
N_ESTIMATORS_MIN, N_ESTIMATORS_MAX   = 100, 1200
NUM_LEAVES_MIN, NUM_LEAVES_MAX       = 8, 128
MIN_CHILD_SAMPLES_MIN, MIN_CHILD_SAMPLES_MAX = 5, 80
SUBSAMPLE_MIN, SUBSAMPLE_MAX         = 0.6, 1.0
COLSAMPLE_MIN, COLSAMPLE_MAX         = 0.6, 1.0
REG_ALPHA_MIN, REG_ALPHA_MAX         = 0.0, 2.0
REG_LAMBDA_MIN, REG_LAMBDA_MAX       = 0.0, 4.0
CORR_MIN, CORR_MAX                   = 0.75, 0.95

# Label & column config
PROG_COL = "Time to First Progression (Days)"
SUBJECT_COL_CANDIDATES = ["subject_id", "Patient_ID", "Subject_ID"]
EXCLUDE_COLS = {"subject_id", "Patient_ID", "Subject_ID", "timepoint", PROG_COL, "delta_days", "days_from_dx"}

DO_TRAIN = 0

# =================== HELPERS ===================
def plot_avg_waterfalls_by_correct_class(top_k=10, max_display=10, out_path=OUTPUT_DIR +"shap_avg_waterfalls__5.png"):
    """
    Compute SHAP on both Train-CV and Test in the pipeline's final feature space,
    average SHAP contributions for correctly classified class-0 and class-1 samples,
    and draw two waterfall plots (left: class 0, right: class 1) for the top-K features.
    """

    # ---- Load trained artifact & pieces ----
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    thr  = float(art["threshold"])
    feat_cols = art["feature_cols"]
    calibrate = bool(art.get("calibrate", False))
    platt = art.get("platt_calibrator", None)

    imp  = pipe.named_steps["imp"]
    vt   = pipe.named_steps["vt"]
    corr = pipe.named_steps["corr"]
    lgb  = pipe.named_steps["clf"]

    # ---------- Helper: prepare X,y,proba in final feature space ----------
    def prep(df_path):
        df = pd.read_excel(df_path)
        y = df[PROG_COL].notna().astype(int).values
        for c in feat_cols:
            if c not in df.columns:
                df[c] = np.nan
        X0 = df[feat_cols].copy()

        # probs from full pipeline (raw + calibrated)
        p_raw = pipe.predict_proba(X0)[:, 1]
        p = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1] if (calibrate and platt is not None) else p_raw
        yhat = (p >= thr).astype(int)

        # push through fitted preprocessors to the model input space
        Xp = imp.transform(X0)
        Xp = vt.transform(Xp)
        Xf = corr.transform(pd.DataFrame(Xp))  # ndarray

        # numbered feature labels 0..P-1
        P = Xf.shape[1]
        cols = [str(i) for i in range(P)]
        Xf_df = pd.DataFrame(Xf, columns=cols)
        return Xf_df, y, yhat, p

    Xtr, ytr, yhat_tr, p_tr = prep(TRAIN_XLSX)
    Xte, yte, yhat_te, p_te = prep(TEST_XLSX)

    # ---------- SHAP on both sets ----------
    explainer = shap.TreeExplainer(lgb)
    exp_tr = explainer(Xtr)
    exp_te = explainer(Xte)

    # Merge (stack) traincv + test
    vals   = np.vstack([exp_tr.values, exp_te.values])           # (N,P)
    base   = np.concatenate([np.atleast_1d(exp_tr.base_values),
                             np.atleast_1d(exp_te.base_values)]) # (N,)
    data   = np.vstack([exp_tr.data, exp_te.data])               # (N,P)
    y_all  = np.concatenate([ytr, yte])
    yh_all = np.concatenate([yhat_tr, yhat_te])
    fnames = exp_tr.feature_names  # numbered strings

    # Masks for correctly classified by class
    mask_c0 = (y_all == 0) & (yh_all == 0)
    mask_c1 = (y_all == 1) & (yh_all == 1)

    def avg_explanation(mask):
        """Build an aggregated SHAP Explanation by averaging over the subset."""
        if mask.sum() == 0:
            return None  # nothing to plot
        mvals = vals[mask].mean(axis=0)
        mbase = float(base[mask].mean())
        mdata = data[mask].mean(axis=0)

        # Pick top-K by mean |SHAP|
        top = np.argsort(np.abs(mvals))[::-1][:top_k]
        mvals, mdata = mvals[top], mdata[top]
        names = [fnames[i] for i in top]

        return shap.Explanation(values=mvals,
                                base_values=mbase,
                                data=mdata,
                                feature_names=names)

    agg_c0 = avg_explanation(mask_c0)
    agg_c1 = avg_explanation(mask_c1)

    plt.figure(figsize=(12, 5))

    # Left: correctly classified class 0
    plt.subplot(1, 2, 1)
    shap.plots.waterfall(agg_c0, show=False, max_display=max_display)
    ax = plt.gca()
    ax.tick_params(axis='y', pad=8)
    for txt in ax.texts:
        txt.set_fontsize(9)                 # <-- shrink left labels
    plt.title("Avg SHAP waterfall — Correct class 0")
    
    # Right: correctly classified class 1
    plt.subplot(1, 2, 2)
    shap.plots.waterfall(agg_c1, show=False, max_display=max_display)
    ax = plt.gca()
    ax.tick_params(axis='y', pad=8)
    for txt in ax.texts:
        txt.set_fontsize(9)                 # <-- also shrink right labels
    plt.title("Avg SHAP waterfall — Correct class 1")
    
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    fig.subplots_adjust(left=0.20, wspace=0.35)    # <-- add wspace
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")  # <-- use out_path
    plt.close()

    print(f"[shap] saved {out_path}")

def fit_platt_on_oof(y_true: np.ndarray, oof_proba: np.ndarray) -> LogisticRegression:
    """
    Fit Platt (sigmoid) calibration on OOF predictions: p_cal = sigmoid(a * p_raw + b).
    Uses scikit-learn LogisticRegression with one feature (p_raw).
    """
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(oof_proba.reshape(-1, 1), y_true.astype(int))
    return lr

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

def build_pipeline(params: dict, corr_threshold: float) -> Pipeline:
    """Light preprocessing + optional correlation pruning + LightGBM."""
    clf = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    pipe = Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("vt",   VarianceThreshold(threshold=1e-12)),
        ("corr", CorrelationFilter(threshold=corr_threshold, method="spearman")),
        ("clf",  clf),
    ])
    return pipe

def oof_cv_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, pipe: Pipeline, n_splits: int, seed: int):
    """Return OOF probabilities, fold AUC list."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    aucs = []

    skf = SGKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y, groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xva)[:, 1]
        oof[va] = p

        if np.unique(yva).size == 2:
            aucs.append(roc_auc_score(yva, p))

    oof_auc = roc_auc_score(y, oof) if np.unique(y).size == 2 else np.nan
    return oof, oof_auc, aucs

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
            best_stats = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "errors": err, "accuracy": acc}

    return best_thr, best_stats

# =================== TRAIN & TEST ===================

def train_lgbm():
    np.random.seed(RANDOM_STATE)
    df = pd.read_excel(TRAIN_XLSX)
    subj_col = get_subject_col(df)
    df[subj_col] = df[subj_col].astype(str)
    X, y, feat_cols = get_xy(df)
    groups = df[subj_col].values

    def objective(trial: optuna.trial.Trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", LEARNING_RATE_MIN, LEARNING_RATE_MAX, log=False),
            "n_estimators":  trial.suggest_int("n_estimators", N_ESTIMATORS_MIN, N_ESTIMATORS_MAX),
            "num_leaves":    trial.suggest_int("num_leaves", NUM_LEAVES_MIN, NUM_LEAVES_MAX),
            "min_child_samples": trial.suggest_int("min_child_samples", MIN_CHILD_SAMPLES_MIN, MIN_CHILD_SAMPLES_MAX),
            "subsample":     trial.suggest_float("subsample", SUBSAMPLE_MIN, SUBSAMPLE_MAX),
            "colsample_bytree": trial.suggest_float("colsample_bytree", COLSAMPLE_MIN, COLSAMPLE_MAX),
            "reg_alpha":     trial.suggest_float("reg_alpha", REG_ALPHA_MIN, REG_ALPHA_MAX),
            "reg_lambda":    trial.suggest_float("reg_lambda", REG_LAMBDA_MIN, REG_LAMBDA_MAX),
        }
        corr = trial.suggest_float("corr_threshold", CORR_MIN, CORR_MAX)

        pipe = build_pipeline(params=params, corr_threshold=corr)
        oof, oof_auc, fold_aucs = oof_cv_predict(X, y, groups, pipe, N_SPLITS, RANDOM_STATE)
        thr, stats = find_best_threshold(y, oof)

        trial.set_user_attr("oof_auc", oof_auc)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("errors_fp_fn", stats.get("errors", None))
        return oof_auc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    best = study.best_trial
    best_params = {
        "learning_rate":  best.params["learning_rate"],
        "n_estimators":   best.params["n_estimators"],
        "num_leaves":     best.params["num_leaves"],
        "min_child_samples": best.params["min_child_samples"],
        "subsample":      best.params["subsample"],
        "colsample_bytree": best.params["colsample_bytree"],
        "reg_alpha":      best.params["reg_alpha"],
        "reg_lambda":     best.params["reg_lambda"],
        "corr_threshold": best.params["corr_threshold"],
    }

    print("\n[optuna] Best AUC:", best.value)
    print("[optuna] Best params:", best_params)
    print("[optuna] OOF threshold (min FP+FN):", best.user_attrs.get("threshold"))
    print("[optuna] OOF FP+FN errors:", best.user_attrs.get("errors_fp_fn"))

    # Recompute OOF with best params & pick threshold
    best_pipe = build_pipeline(
        params={k: v for k, v in best_params.items() if k != "corr_threshold"},
        corr_threshold=best_params["corr_threshold"],
    )

    oof_raw, oof_auc, fold_aucs = oof_cv_predict(X, y, groups, best_pipe, N_SPLITS, RANDOM_STATE)

    # ---- NEW: Platt calibration on OOF (no leakage) ----
    calibrator = None
    if CALIBRATE:
        calibrator = fit_platt_on_oof(y, oof_raw)
        oof_cal = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
        # Choose threshold on CALIBRATED OOF
        thr, stats = find_best_threshold(y, oof_cal)
        # Diagnostics
        try:
            raw_brier = brier_score_loss(y, oof_raw)
            cal_brier = brier_score_loss(y, oof_cal)
            print(f"[calibration] Platt: Brier raw={raw_brier:.4f} -> cal={cal_brier:.4f}")
        except Exception:
            pass
    else:
        thr, stats = find_best_threshold(y, oof_raw)

    print("\n[final OOF] AUC (ranking, unaffected by calibration):", round(oof_auc, 4),
          "| mean fold AUC:", round(float(np.mean(fold_aucs)), 4))
    print("[final OOF] threshold (chosen on calibrated OOF if CALIBRATE):", thr, "| stats:", stats)
    print(f"[VAL] Final OOF AUC = {oof_auc:.4f}")

    # Fit on ALL train-cv
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
        "calibrate": bool(CALIBRATE),
        "platt_calibrator": calibrator,
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"[saved] {ARTIFACT}")

def test_lgbm():
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

    # Build features with the SAME columns used in training
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df[feat_cols].copy()

    # Predict
    proba_raw = pipe.predict_proba(X)[:, 1]
    if calibrate and platt is not None:
        proba = platt.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
    else:
        proba = proba_raw

    pred = (proba >= thr).astype(int)

    # Load TRAIN-CV and align feature columns
    df_tr = pd.read_excel(TRAIN_XLSX)
    if PROG_COL not in df_tr.columns:
        raise ValueError(f"Label column '{PROG_COL}' not found in train file")
    y_tr = df_tr[PROG_COL].notna().astype(int).values
    for c in feat_cols:
        if c not in df_tr.columns:
            df_tr[c] = np.nan
    X_tr = df_tr[feat_cols].copy()

    # Raw + calibrated probs on TRAIN-CV (IN-SAMPLE; optimistic)
    proba_raw_tr = pipe.predict_proba(X_tr)[:, 1]
    proba_tr = (platt.predict_proba(proba_raw_tr.reshape(-1, 1))[:, 1]
                if calibrate and platt is not None else proba_raw_tr)

    # ===== Calibration curves: Train-CV vs Test =====
    n_bins = 7  # fewer bins for small sets → stabler curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Left: TRAIN-CV
    ax = axes[0]
    fr_raw_tr, mp_raw_tr = calibration_curve(y_tr, proba_raw_tr, n_bins=n_bins, strategy="quantile")
    ax.plot(mp_raw_tr, fr_raw_tr, "o-", label="Raw (TrainCV)")
    if calibrate and platt is not None:
        fr_cal_tr, mp_cal_tr = calibration_curve(y_tr, proba_tr, n_bins=n_bins, strategy="quantile")
        ax.plot(mp_cal_tr, fr_cal_tr, "o-", label="Platt (TrainCV)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title("Calibration — Train-CV (IN-SAMPLE)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.grid(True)
    ax.legend(loc="best")

    # Right: TEST
    ax = axes[1]
    fr_raw_te, mp_raw_te = calibration_curve(y, proba_raw, n_bins=n_bins, strategy="quantile")
    ax.plot(mp_raw_te, fr_raw_te, "o-", label="Raw (Test)")
    if calibrate and platt is not None:
        fr_cal_te, mp_cal_te = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mp_cal_te, fr_cal_te, "o-", label="Platt (Test)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title("Calibration — Test")
    ax.set_xlabel("Mean predicted probability")
    ax.grid(True)
    ax.legend(loc="best")

    plt.suptitle("Calibration curves: Train-CV vs Test")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "calibration_oof_vs_test__5.png", dpi=150)
    plt.close()

    # ===== Brier scores (print + bar chart) =====
    brier_raw_tr = brier_score_loss(y_tr, proba_raw_tr)
    brier_cal_tr = brier_score_loss(y_tr, proba_tr)
    brier_raw_te = brier_score_loss(y, proba_raw)
    brier_cal_te = brier_score_loss(y, proba)

    print(f"[brier] TrainCV raw={brier_raw_tr:.4f} | TrainCV cal={brier_cal_tr:.4f} | "
          f"Test raw={brier_raw_te:.4f} | Test cal={brier_cal_te:.4f}")

    labels = ["TrainCV Raw", "TrainCV Cal", "Test Raw", "Test Cal"]
    scores = [brier_raw_tr, brier_cal_tr, brier_raw_te, brier_cal_te]
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    bars = ax.bar(labels, scores)
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Brier scores — Raw vs Platt, Train-CV & Test")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # annotate bars
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "brier_bars__5.png", dpi=150)
    plt.close()

    # ROC AUC
    auc_val = roc_auc_score(y, proba) if np.unique(y).size == 2 else None
    print(f"[test] ROC AUC: {auc_val:.4f}" if auc_val is not None else "[test] ROC AUC: N/A")
    # Brier score for probability quality
    try:
        print(f"[test] Brier score: {brier_score_loss(y, proba):.4f}")
    except Exception:
        pass

    # ===== ROC curves (Test) =====
    # Raw
    fpr_raw, tpr_raw, _ = roc_curve(y, proba_raw)
    auc_raw = roc_auc_score(y, proba_raw)

    # Calibrated (if available)
    has_cal = calibrate and (platt is not None)
    if has_cal:
        fpr_cal, tpr_cal, _ = roc_curve(y, proba)
        auc_cal = roc_auc_score(y, proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_raw, tpr_raw, label=f"AUC = {auc_raw:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Test set")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR +"roc_plot_test__5.png", dpi=350)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print("\n[test] Confusion Matrix (rows=true, cols=pred):")
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
    # Peek at calibrated probabilities
    print("\n[test] first 10 probabilities:", np.round(proba[:10], 4))

def export_feature_mapping_numbers(csv_path=OUTPUT_DIR + "feature_index_name_map.csv"):
    """
    Build mapping AFTER training (no retrain):
    feature_id (0..P-1)  -> radiomic feature name that survived VT + CorrFilter.

    Saves a CSV with: feature_id, feature_name, vt_index, orig_index, orig_col
    """
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    feat_cols = art["feature_cols"]  # original column names (pre-VT/corr)

    imp  = pipe.named_steps["imp"]
    vt   = pipe.named_steps["vt"]
    corr = pipe.named_steps["corr"]

    # Names that survived VarianceThreshold, in the same order as VT output
    vt_mask = vt.get_support(indices=False)                  # bool mask over feat_cols
    names_after_vt = np.array(feat_cols)[vt_mask]            # shape (K,)

    # Indices (into names_after_vt) kept by CorrelationFilter, in the final model order
    kept_idx = np.array(corr.keep_features_, dtype=int)      # shape (P,)
    kept_names = names_after_vt[kept_idx]                    # final radiomic names

    # Map each kept name back to its original column index
    orig_index_map = {name: i for i, name in enumerate(feat_cols)}
    orig_index = [orig_index_map[n] for n in kept_names]

    df_map = pd.DataFrame({
        "feature_id": np.arange(len(kept_names), dtype=int),  # the numbers you’ll show in plots
        "feature_name": kept_names,
        "vt_index": kept_idx,                                 # index in VT output
        "orig_index": orig_index,                             # index in original dataframe
        "orig_col": [feat_cols[i] for i in orig_index],       # same as feature_name; kept for clarity
    })
    df_map.to_csv(csv_path, index=False)
    print(f"[mapping] saved {csv_path} with {len(df_map)} rows.")

def run_treeshap_numbered(top_k=30):
    """
    TreeSHAP with numeric feature labels (0..P-1) so plots match your mapping table.
    Waterfall is drawn for the MOST REPRESENTATIVE POSITIVE case:
    the true-1 test sample with the highest predicted probability (calibrated if available).

    Saves:
      - shap_beeswarm_numbers__test.png
      - shap_bar_numbers__test.png
      - shap_dependence_top1_numbers__test.png
      - shap_waterfall_repr_pos_numbers__test__idx{IDX}.png
    """

    # ---- Load trained artifact and pipeline pieces ----
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    feat_cols = art["feature_cols"]
    calibrate = bool(art.get("calibrate", False))
    platt = art.get("platt_calibrator", None)

    imp  = pipe.named_steps["imp"]
    vt   = pipe.named_steps["vt"]
    corr = pipe.named_steps["corr"]
    lgb  = pipe.named_steps["clf"]

    # ---- Prepare TEST features with same columns as training ----
    df_te = pd.read_excel(TEST_XLSX)
    if PROG_COL not in df_te.columns:
        raise ValueError(f"Label column '{PROG_COL}' not found in test file")
    y_test = df_te[PROG_COL].notna().astype(int).values

    for c in feat_cols:
        if c not in df_te.columns:
            df_te[c] = np.nan
    X_te_raw = df_te[feat_cols].copy()

    # ---- Probabilities on TEST (via full pipeline to avoid mismatch) ----
    proba_raw = pipe.predict_proba(X_te_raw)[:, 1]
    proba = (platt.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
             if calibrate and (platt is not None) else proba_raw)

    # ---- Apply fitted preprocessing for SHAP (no refit!) ----
    Xp = imp.transform(X_te_raw)                 # ndarray
    Xp = vt.transform(Xp)                        # ndarray
    X_final = corr.transform(pd.DataFrame(Xp))   # ndarray fed to LGBM in pipeline

    # Numbered labels 0..P-1 for final feature space
    P = X_final.shape[1]
    num_labels = [str(i) for i in range(P)]
    X_final_df = pd.DataFrame(X_final, columns=num_labels)

    # ---- TreeSHAP on LightGBM using numbered labels ----
    explainer = shap.TreeExplainer(lgb)
    exp = explainer(X_final_df)  # SHAP Explanation object

    # Global plots (top-K features)
    shap.plots.beeswarm(exp, show=False, max_display=top_k)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR +"shap_beeswarm_numbers__test.png", dpi=150); plt.close()

    shap.plots.bar(exp, show=False, max_display=top_k)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR +"shap_bar_numbers__test.png", dpi=150); plt.close()

    # Top-1 feature by mean |SHAP|
    top_idx = int(np.argsort(np.abs(exp.values).mean(0))[::-1][0])
    shap.plots.scatter(exp[:, top_idx], color=exp, show=False)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR +"shap_dependence_top1_numbers__test.png", dpi=150); plt.close()

    # ---- Waterfall for MOST REPRESENTATIVE POSITIVE (true-1 with highest prob) ----
    pos_idxs = np.where(y_test == 1)[0]
    if pos_idxs.size > 0:
        # among true positives (by label), pick the one with the highest predicted prob
        best_pos_local = pos_idxs[np.argmax(proba[pos_idxs])]
        shap.plots.waterfall(exp[int(best_pos_local)], show=False)
        out_name = OUTPUT_DIR +f"shap_waterfall_repr_pos_numbers__test__idx{int(best_pos_local)}.png"
        plt.tight_layout(); plt.savefig(out_name, dpi=150); plt.close()
        print(f"[treeshap] Waterfall saved for representative positive at test idx={int(best_pos_local)} "
              f"(prob={proba[int(best_pos_local)]:.4f}).")
    else:
        # Fallback: no positives in test → use overall highest prob
        best_any = int(np.argmax(proba))
        shap.plots.waterfall(exp[best_any], show=False)
        out_name = OUTPUT_DIR +f"shap_waterfall_repr_pos_numbers__test__idx{best_any}_fallback.png"
        plt.tight_layout(); plt.savefig(out_name, dpi=150); plt.close()
        print(f"[treeshap] No positives in test; fallback waterfall at idx={best_any} "
              f"(prob={proba[best_any]:.4f}).")

# Example usage
if __name__ == "__main__":
    if DO_TRAIN:
        train_lgbm()
    test_lgbm()
    export_feature_mapping_numbers()
    run_treeshap_numbered(top_k=15)
    plot_avg_waterfalls_by_correct_class(top_k=20, max_display=11, out_path=OUTPUT_DIR +"shap_avg_waterfalls__5.png")

