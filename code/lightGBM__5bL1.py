#!/usr/bin/env python3
"""
LightGBM (with Optuna) for radiomics classification — MGMT label.

- Label y = MGMT (0/1); rows with NaN in MGMT are dropped.
- Train set:  derivatives/traincv_set__5b_topk.xlsx  (hardcoded)
- Test set:   derivatives/test_set__5b_topk.xlsx     (hardcoded)
- Group-aware CV by subject_id (or Patient_ID/Subject_ID/patient_id fallback) to avoid leakage
- Objective: maximize OOF ROC AUC across CV folds
- Threshold: chosen on OOF predictions to minimize FP+FN
- Saves best artifact to: mgmt_lgbm__5b.joblib

Functions:
    train_lgbm()  -> runs Optuna, saves artifact, prints CV metrics & threshold
    test_lgbm()   -> loads artifact, evaluates on test set (ROC AUC, CM, report)

Requirements: lightgbm, scikit-learn, optuna, pandas, numpy, joblib, shap, matplotlib
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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold as SGKFold  # sklearn >= 1.3
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression  # Platt (sigmoid) calibrator
from sklearn.calibration import calibration_curve

from lightgbm import LGBMClassifier

# =================== HARD-CODED SETTINGS ===================
#TRAIN_XLSX = "derivatives/traincv_set__4_topk.xlsx"
#TEST_XLSX  = "derivatives/test_set__4_topk.xlsx"

TRAIN_XLSX = "derivatives/traincv_set__4.xlsx"
TEST_XLSX = "derivatives/test_set__4.xlsx"

ARTIFACT   = "mgmt_lgbm__5b.joblib"
CALIBRATE  = True

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
C_SEL_MIN, C_SEL_MAX = 1e-3, 10.0
KEEP_FRAC_MIN, KEEP_FRAC_MAX = 0.2, 0.6
KEEP_FRAC_DEFAULT = 0.4


# Label & column config
LABEL_COL = "MGMT"  # 0/1; NaN dropped
SUBJECT_COL_CANDIDATES = ["subject_id", "Patient_ID", "Subject_ID", "patient_id"]
EXTRA_EXCLUDE = {"dataset", "sample_id", "roi", "timepoint"}  # drop if present

# Toggle training when running as script
DO_TRAIN = 1

# =================== HELPERS ===================
class ModelFractionSelector(BaseEstimator, TransformerMixin):
    """
    Fit a base estimator (e.g., L1-logreg) on incoming features and keep the
    top keep_frac by |coef|. Works for binary/multiclass (uses mean |coef| across classes).
    """
    def __init__(self, base_estimator=None, keep_frac=0.5, random_state=42):
        self.base_estimator = base_estimator
        self.keep_frac = float(keep_frac)
        self.random_state = random_state
        self.est_ = None
        self.keep_idx_ = None
        self.mask_ = None

    def fit(self, X, y=None):
        if not (0 < self.keep_frac <= 1):
            raise ValueError("keep_frac must be in (0,1].")
        est = self.base_estimator or LogisticRegression(
            penalty="l1", solver="saga", C=0.5, max_iter=10000,
            class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )
        self.est_ = clone(est).fit(X, y)
        coef = getattr(self.est_, "coef_", None)
        if coef is None:
            raise RuntimeError("Base estimator does not expose coef_.")
        coef_abs = np.mean(np.abs(coef), axis=0)
        n = coef_abs.shape[0]
        k = max(1, int(round(self.keep_frac * n)))
        rank = np.argsort(-coef_abs)
        self.keep_idx_ = np.sort(rank[:k])
        self.mask_ = np.zeros(n, dtype=bool)
        self.mask_[self.keep_idx_] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_idx_] if self.keep_idx_ is not None else np.asarray(X)

    def get_support(self, indices=False):
        if self.mask_ is None: return None
        return self.keep_idx_ if indices else self.mask_

def fit_platt_on_oof(y_true: np.ndarray, oof_proba: np.ndarray) -> LogisticRegression:
    """Platt (sigmoid) calibration on OOF predictions: p_cal = sigmoid(a * p_raw + b)."""
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
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataframe")
    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    mask = y.isin([0, 1])
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int).values

    exclude = set(SUBJECT_COL_CANDIDATES) | EXTRA_EXCLUDE | {LABEL_COL}
    feat_cols = [c for c in df.columns if c not in exclude and is_numeric_dtype(df[c])]
    X = df[feat_cols].copy()
    X.columns = X.columns.astype(str)
    return X, y, feat_cols

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from any pair whose |corr| >= threshold (fit on input only)."""
    def __init__(self, threshold: float = 0.80, method: str = "spearman"):
        self.threshold = threshold
        self.method = method
        self.keep_features_: list[str] | None = None

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
        if not self.keep_features_:
            return Xdf.values
        return Xdf[self.keep_features_].values

def build_pipeline(params: dict, corr_threshold: float, C_sel: float, keep_frac: float) -> Pipeline:
    clf = LGBMClassifier(
        objective="binary",
        keep_training_booster=False,
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
        # NEW: % keep selector (after Corr; trees don’t need scaling)
        ("sel",  ModelFractionSelector(
            base_estimator=LogisticRegression(
                penalty="l1", solver="saga", C=C_sel, max_iter=10000,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            keep_frac=keep_frac, random_state=RANDOM_STATE
        )),
        ("clf",  clf),
    ])
    return pipe

def oof_cv_predict(X, y, groups, pipe, n_splits, seed):
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    aucs = []

    skf = SGKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y, groups):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        fold_pipe = clone(pipe)
        fold_pipe.fit(Xtr, ytr)
        p = fold_pipe.predict_proba(Xva)[:, 1]
        oof[va] = p

        if np.unique(yva).size == 2:
            aucs.append(roc_auc_score(yva, p))

    oof_auc = roc_auc_score(y, oof) if np.unique(y).size == 2 else np.nan
    return oof, oof_auc, aucs

def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict]:
    """Choose threshold that minimizes FP + FN."""
    vals = np.unique(proba)
    thr_grid = np.quantile(proba, np.linspace(0.01, 0.99, 199)) if vals.size > 1000 else vals
    thr_grid = np.unique(thr_grid)

    best_thr, best_err, best_stats = 0.5, float("inf"), {}
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

    X, y, feat_cols = get_xy(df)          # drops NaN MGMT
    groups = df.loc[X.index, subj_col].values  # align groups to X rows

    def objective(trial: optuna.trial.Trial):
        params = {
            "learning_rate":       trial.suggest_float("learning_rate", LEARNING_RATE_MIN, LEARNING_RATE_MAX),
            "n_estimators":        trial.suggest_int("n_estimators", N_ESTIMATORS_MIN, N_ESTIMATORS_MAX),
            "num_leaves":          trial.suggest_int("num_leaves", NUM_LEAVES_MIN, NUM_LEAVES_MAX),
            "min_child_samples":   trial.suggest_int("min_child_samples", MIN_CHILD_SAMPLES_MIN, MIN_CHILD_SAMPLES_MAX),
            "subsample":           trial.suggest_float("subsample", SUBSAMPLE_MIN, SUBSAMPLE_MAX),
            "colsample_bytree":    trial.suggest_float("colsample_bytree", COLSAMPLE_MIN, COLSAMPLE_MAX),
            "reg_alpha":           trial.suggest_float("reg_alpha", REG_ALPHA_MIN, REG_ALPHA_MAX),
            "reg_lambda":          trial.suggest_float("reg_lambda", REG_LAMBDA_MIN, REG_LAMBDA_MAX),
        }
        corr     = trial.suggest_float("corr_threshold", CORR_MIN, CORR_MAX)
        C_sel    = trial.suggest_float("C_sel", C_SEL_MIN, C_SEL_MAX, log=True)
        keep_fr  = trial.suggest_float("keep_frac", KEEP_FRAC_MIN, KEEP_FRAC_MAX)

        pipe = build_pipeline(params=params, corr_threshold=corr, C_sel=C_sel, keep_frac=keep_fr)
        oof, oof_auc, _ = oof_cv_predict(X, y, groups, pipe, N_SPLITS, RANDOM_STATE)
        thr, stats = find_best_threshold(y, oof)

        trial.set_user_attr("oof_auc", oof_auc)
        trial.set_user_attr("threshold", thr)
        trial.set_user_attr("errors_fp_fn", stats.get("errors", None))
        return oof_auc

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=-1, show_progress_bar=True)

    # === AFTER OPTUNA: collect best params, print, and rebuild pipeline ===
    best = study.best_trial
    best_params = {
        "learning_rate":       best.params["learning_rate"],
        "n_estimators":        best.params["n_estimators"],
        "num_leaves":          best.params["num_leaves"],
        "min_child_samples":   best.params["min_child_samples"],
        "subsample":           best.params["subsample"],
        "colsample_bytree":    best.params["colsample_bytree"],
        "reg_alpha":           best.params["reg_alpha"],
        "reg_lambda":          best.params["reg_lambda"],
        "corr_threshold":      best.params["corr_threshold"],
        "C_sel":               best.params["C_sel"],
        "keep_frac":           best.params.get("keep_frac", KEEP_FRAC_DEFAULT),
    }

    print("\n[optuna] Best AUC:", best.value)
    print("[optuna] Best params:", best_params)
    print("[optuna] OOF threshold (min FP+FN):", best.user_attrs.get("threshold"))
    print("[optuna] OOF FP+FN errors:", best.user_attrs.get("errors_fp_fn"))

    # Recompute OOF with best params
    best_pipe = build_pipeline(
        params={k: v for k, v in best_params.items() if k not in ("corr_threshold", "C_sel", "keep_frac")},
        corr_threshold=best_params["corr_threshold"],
        C_sel=best_params["C_sel"],
        keep_frac=best_params["keep_frac"],
    )

    oof_raw, oof_auc, fold_aucs = oof_cv_predict(X, y, groups, best_pipe, N_SPLITS, RANDOM_STATE)

    # Platt calibration on OOF
    calibrator = None
    if CALIBRATE:
        calibrator = fit_platt_on_oof(y, oof_raw)
        oof_cal = calibrator.predict_proba(oof_raw.reshape(-1, 1))[:, 1]
        thr, stats = find_best_threshold(y, oof_cal)
        try:
            raw_brier = brier_score_loss(y, oof_raw)
            cal_brier = brier_score_loss(y, oof_cal)
            print(f"[calibration] Platt: Brier raw={raw_brier:.4f} -> cal={cal_brier:.4f}")
        except Exception:
            pass
    else:
        thr, stats = find_best_threshold(y, oof_raw)

    print("\n[final OOF] AUC:", round(oof_auc, 4),
          "| mean fold AUC:", round(float(np.mean(fold_aucs)), 4))
    print("[final OOF] threshold (chosen on calibrated OOF if CALIBRATE):", thr, "| stats:", stats)

    # Fit on ALL train-cv
    best_pipe.fit(X, y)
    
    try:
        n0 = X.shape[1]
        vt_mask = best_pipe.named_steps["vt"].get_support(indices=False); n_vt = int(vt_mask.sum())
        kept_corr = best_pipe.named_steps["corr"].keep_features_ or []
        n_corr = len(kept_corr)
        sel_mask = best_pipe.named_steps["sel"].get_support(indices=False); n_sel = int(sel_mask.sum())
        print(f"[features] start={n0} | after VT={n_vt} | after Corr={n_corr} | after Selector={n_sel}")
    except Exception as e:
        print(f"[features] could not compute stage counts: {e}")
    
    # Save artifact
    artifact = {
        "pipeline": best_pipe,
        "threshold": thr,
        "params": best_params,
        "feature_cols": feat_cols,
        "subject_col": get_subject_col(df),
        "train_oof_auc": float(oof_auc),
        "train_oof_threshold_stats": stats,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "calibrate": bool(CALIBRATE),
        "platt_calibrator": calibrator,
        "label_col": LABEL_COL,
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
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in test file")
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(-1).astype(int).values  # -1 for unlabeled if any
    mask_eval = np.isin(y, [0, 1])
    y = y[mask_eval]
    # Build features with the SAME columns used in training
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df.loc[mask_eval, feat_cols].copy()

    # Predict
    proba_raw = pipe.predict_proba(X)[:, 1]
    proba = platt.predict_proba(proba_raw.reshape(-1, 1))[:, 1] if (calibrate and platt is not None) else proba_raw
    pred = (proba >= thr).astype(int)

    # Load TRAIN-CV and align feature columns (for calibration plots)
    df_tr = pd.read_excel(TRAIN_XLSX)
    if LABEL_COL not in df_tr.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in train file")
    y_tr = pd.to_numeric(df_tr[LABEL_COL], errors="coerce")
    mask_tr = y_tr.isin([0, 1]).values
    y_tr = y_tr.loc[mask_tr].astype(int).values
    for c in feat_cols:
        if c not in df_tr.columns:
            df_tr[c] = np.nan
    X_tr = df_tr.loc[mask_tr, feat_cols].copy()

    # Raw + calibrated probs on TRAIN-CV (IN-SAMPLE; optimistic)
    proba_raw_tr = pipe.predict_proba(X_tr)[:, 1]
    proba_tr = (platt.predict_proba(proba_raw_tr.reshape(-1, 1))[:, 1]
                if calibrate and platt is not None else proba_raw_tr)

    # ===== Calibration curves: Train-CV vs Test =====
    n_bins = 7
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Train-CV
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
    ax.grid(True); ax.legend(loc="best")

    # Test
    ax = axes[1]
    fr_raw_te, mp_raw_te = calibration_curve(y, proba_raw, n_bins=n_bins, strategy="quantile")
    ax.plot(mp_raw_te, fr_raw_te, "o-", label="Raw (Test)")
    if calibrate and platt is not None:
        fr_cal_te, mp_cal_te = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mp_cal_te, fr_cal_te, "o-", label="Platt (Test)")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_title("Calibration — Test")
    ax.set_xlabel("Mean predicted probability")
    ax.grid(True); ax.legend(loc="best")

    plt.suptitle("Calibration curves: Train-CV vs Test (MGMT)")
    plt.tight_layout()
    plt.savefig("calibration_oof_vs_test__5b.png", dpi=150)
    plt.close()

    # ===== Brier scores =====
    brier_raw_tr = brier_score_loss(y_tr, proba_raw_tr)
    brier_cal_tr = brier_score_loss(y_tr, proba_tr)
    brier_raw_te = brier_score_loss(y, proba_raw)
    brier_cal_te = brier_score_loss(y, proba)

    print(f"[brier] TrainCV raw={brier_raw_tr:.4f} | TrainCV cal={brier_cal_tr:.4f} | "
          f"Test raw={brier_raw_te:.4f} | Test cal={brier_cal_te:.4f}")

    # ROC AUC
    auc_val = roc_auc_score(y, proba) if np.unique(y).size == 2 else None
    print(f"[test] ROC AUC: {auc_val:.4f}" if auc_val is not None else "[test] ROC AUC: N/A")
    print(f"[test] Brier score: {brier_score_loss(y, proba):.4f}")

    # ===== ROC curves (Test) =====
    fpr_raw, tpr_raw, _ = roc_curve(y, proba_raw)
    auc_raw = roc_auc_score(y, proba_raw)
    has_cal = calibrate and (platt is not None)
    if has_cal:
        fpr_cal, tpr_cal, _ = roc_curve(y, proba)
        auc_cal = roc_auc_score(y, proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_raw, tpr_raw, label=f"Raw (AUC={auc_raw:.3f})")
    if has_cal:
        plt.plot(fpr_cal, tpr_cal, label=f"Platt (AUC={auc_cal:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — Test set (MGMT)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_plot_test__5b.png", dpi=350)
    plt.close()

    # Confusion matrix
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print("\n[test] Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Classification report
    report = classification_report(
        y, pred, labels=[0, 1],
        target_names=["Unmethylated (0)", "Methylated (1)"],
        zero_division=0, digits=4
    )
    print("\n[test] classification report:\n" + report)
    print("\n[test] first 10 probabilities:", np.round(proba[:10], 4))

def export_feature_mapping_numbers(csv_path="feature_index_name_map__5b.csv"):
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]; feat_cols = art["feature_cols"]
    vt   = pipe.named_steps["vt"]; corr = pipe.named_steps["corr"]; sel = pipe.named_steps.get("sel")

    vt_mask = vt.get_support(indices=False)
    names_after_vt = np.array(feat_cols)[vt_mask]

    # Corr names
    kept = corr.keep_features_
    if kept is None or len(kept) == 0:
        raise RuntimeError("No features survived CorrFilter.")
    if isinstance(kept[0], (int, np.integer)):
        names_after_corr = names_after_vt[np.array(kept, dtype=int)]
    else:
        names_after_corr = np.array([k for k in kept if k in names_after_vt])

    # Selector mask -> final names
    if sel is not None:
        sel_mask = sel.get_support(indices=False)
        if sel_mask is None or sel_mask.shape[0] != len(names_after_corr):
            raise RuntimeError("Selector mask missing or mismatched.")
        kept_names = names_after_corr[sel_mask]
    else:
        kept_names = names_after_corr

    orig_index_map = {name: i for i, name in enumerate(feat_cols)}
    orig_index = [orig_index_map[n] for n in kept_names]

    df_map = pd.DataFrame({
        "feature_id": np.arange(len(kept_names), dtype=int),  # 0..P-1 in final model
        "feature_name": kept_names,
        "orig_index": orig_index,
        "orig_col": [feat_cols[i] for i in orig_index],
    })
    df_map.to_csv(csv_path, index=False)
    print(f"[mapping] saved {csv_path} with {len(df_map)} rows.")

def run_treeshap_numbered(top_k=30):
    """
    TreeSHAP with numeric feature labels (0..P-1) so plots match the mapping table.
    Waterfall is drawn for the MOST REPRESENTATIVE POSITIVE case (true label = 1).
    Saves:
      - shap_beeswarm_numbers__test__5b.png
      - shap_bar_numbers__test__5b.png
      - shap_dependence_top1_numbers__test__5b.png
      - shap_waterfall_repr_pos_numbers__test__idx{IDX}__5b.png
    """
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    feat_cols = art["feature_cols"]
    calibrate = bool(art.get("calibrate", False))
    platt = art.get("platt_calibrator", None)

    corr = pipe.named_steps["corr"]
    lgb  = pipe.named_steps["clf"]

    # Prepare TEST like in test_lgbm
    df_te = pd.read_excel(TEST_XLSX)
    y_test = pd.to_numeric(df_te[LABEL_COL], errors="coerce").fillna(-1).astype(int).values
    mask_eval = np.isin(y_test, [0, 1])
    y_test = y_test[mask_eval]
    for c in feat_cols:
        if c not in df_te.columns:
            df_te[c] = np.nan
    X_te_raw = df_te.loc[mask_eval, feat_cols].copy()

    proba_raw = pipe.predict_proba(X_te_raw)[:, 1]
    proba = (platt.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
             if calibrate and (platt is not None) else proba_raw)

    imp  = pipe.named_steps["imp"]
    vt   = pipe.named_steps["vt"]
    corr = pipe.named_steps["corr"]
    sel  = pipe.named_steps.get("sel")

    Xp = imp.transform(X_te_raw)
    Xp = vt.transform(Xp)
    Xc = corr.transform(pd.DataFrame(Xp))

    if sel is not None:
        X_final = sel.transform(Xc)
    else:
        X_final = Xc

    P = X_final.shape[1]
    X_final_df = pd.DataFrame(X_final, columns=[str(i) for i in range(P)])

    explainer = shap.TreeExplainer(lgb)
    exp = explainer(X_final_df)

    shap.plots.beeswarm(exp, show=False, max_display=top_k)
    plt.tight_layout(); plt.savefig("shap_beeswarm_numbers__test__5b.png", dpi=150); plt.close()

    shap.plots.bar(exp, show=False, max_display=top_k)
    plt.tight_layout(); plt.savefig("shap_bar_numbers__test__5b.png", dpi=150); plt.close()

    top_idx = int(np.argsort(np.abs(exp.values).mean(0))[::-1][0])
    shap.plots.scatter(exp[:, top_idx], color=exp, show=False)
    plt.tight_layout(); plt.savefig("shap_dependence_top1_numbers__test__5b.png", dpi=150); plt.close()

    pos_idxs = np.where(y_test == 1)[0]
    if pos_idxs.size > 0:
        best_pos_local = pos_idxs[np.argmax(proba[pos_idxs])]
        shap.plots.waterfall(exp[int(best_pos_local)], show=False)
        out_name = f"shap_waterfall_repr_pos_numbers__test__idx{int(best_pos_local)}__5b.png"
        plt.tight_layout(); plt.savefig(out_name, dpi=150); plt.close()
        print(f"[treeshap] Waterfall saved for representative positive at test idx={int(best_pos_local)} "
              f"(prob={proba[int(best_pos_local)]:.4f}).")
    else:
        best_any = int(np.argmax(proba))
        shap.plots.waterfall(exp[best_any], show=False)
        out_name = f"shap_waterfall_repr_pos_numbers__test__idx{best_any}_fallback__5b.png"
        plt.tight_layout(); plt.savefig(out_name, dpi=150); plt.close()
        print(f"[treeshap] No positives in test; fallback waterfall at idx={best_any} "
              f"(prob={proba[best_any]:.4f}).")

def plot_avg_waterfalls_by_correct_class(top_k=10, max_display=10, out_path="shap_avg_waterfalls__5b.png"):
    """
    Average SHAP waterfalls over correctly classified 0s and 1s (traincv+test).
    """
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    thr  = float(art["threshold"])
    feat_cols = art["feature_cols"]
    calibrate = bool(art.get("calibrate", False))
    platt = art.get("platt_calibrator", None)
    lgb  = pipe.named_steps["clf"]

    def prep(df_path):
        df = pd.read_excel(df_path)
        y = pd.to_numeric(df[LABEL_COL], errors="coerce")
        mask = y.isin([0, 1]).values
        y = y.loc[mask].astype(int).values
        for c in feat_cols:
            if c not in df.columns:
                df[c] = np.nan
        X0 = df.loc[mask, feat_cols].copy()
    
        # Predictions (raw + calibrated)
        p_raw = pipe.predict_proba(X0)[:, 1]
        p = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1] if (calibrate and platt is not None) else p_raw
        yhat = (p >= thr).astype(int)
    
        # --- Preprocess to final model space (VT -> Corr -> Selector) ---
        imp  = pipe.named_steps["imp"]
        vt   = pipe.named_steps["vt"]
        corr = pipe.named_steps["corr"]
        sel  = pipe.named_steps.get("sel")  # may be None
    
        Xp = imp.transform(X0)
        Xp = vt.transform(Xp)
        Xc = corr.transform(pd.DataFrame(Xp))
    
        # Apply %keep selector if present
        if sel is not None:
            Xf = sel.transform(Xc)
        else:
            Xf = Xc
    
        # Use numeric feature labels 0..P-1 to match SHAP-numbered plots
        cols = [str(i) for i in range(Xf.shape[1])]
        Xf_df = pd.DataFrame(Xf, columns=cols)
        return Xf_df, y, yhat, p

    Xtr, ytr, yhat_tr, p_tr = prep(TRAIN_XLSX)
    Xte, yte, yhat_te, p_te = prep(TEST_XLSX)

    explainer = shap.TreeExplainer(lgb)
    exp_tr = explainer(Xtr); exp_te = explainer(Xte)

    vals   = np.vstack([exp_tr.values, exp_te.values])
    base   = np.concatenate([np.atleast_1d(exp_tr.base_values), np.atleast_1d(exp_te.base_values)])
    data   = np.vstack([exp_tr.data, exp_te.data])
    y_all  = np.concatenate([ytr, yte])
    yh_all = np.concatenate([yhat_tr, yhat_te])
    fnames = exp_tr.feature_names

    mask_c0 = (y_all == 0) & (yh_all == 0)
    mask_c1 = (y_all == 1) & (yh_all == 1)

    def avg_explanation(mask):
        if mask.sum() == 0:
            return None
        mvals = vals[mask].mean(axis=0)
        mbase = float(base[mask].mean())
        mdata = data[mask].mean(axis=0)
        top = np.argsort(np.abs(mvals))[::-1][:top_k]
        mvals, mdata = mvals[top], mdata[top]
        names = [fnames[i] for i in top]
        return shap.Explanation(values=mvals, base_values=mbase, data=mdata, feature_names=names)

    agg_c0 = avg_explanation(mask_c0)
    agg_c1 = avg_explanation(mask_c1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); shap.plots.waterfall(agg_c0, show=False, max_display=max_display); plt.title("Avg SHAP — correct 0")
    plt.subplot(1, 2, 2); shap.plots.waterfall(agg_c1, show=False, max_display=max_display); plt.title("Avg SHAP — correct 1")
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[shap] saved {out_path}")

# Example usage
if __name__ == "__main__":
    if DO_TRAIN:
        train_lgbm()
    test_lgbm()
    export_feature_mapping_numbers()
    run_treeshap_numbered(top_k=15)
    plot_avg_waterfalls_by_correct_class(top_k=20, max_display=11, out_path="shap_avg_waterfalls__5b.png")

