#!/usr/bin/env python3
"""
Confusion-matrix plot for the MGMT LightGBM pipeline (5b).

Unpickling helpers:
- CorrelationFilter: same as training (keeps fitted 'keep_features_').
- ModelFractionSelector: robust stub that tries to use stored kept columns/indices.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# ---------- Custom classes needed for unpickling ----------

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
        if not getattr(self, "keep_features_", None):
            return Xdf.values
        return Xdf[self.keep_features_].values

class ModelFractionSelector(BaseEstimator, TransformerMixin):
    """
    Robust passthrough that respects the feature subset stored during training.
    It tries to find any attribute that looks like kept columns or indices.
    """
    def __init__(self, fraction: float = 1.0, random_state: int | None = None,
                 by: str = "features", preserve_order: bool = True):
        self.fraction = fraction
        self.random_state = random_state
        self.by = by
        self.preserve_order = preserve_order

    def fit(self, X, y=None):
        return self

    def _guess_kept(self, X):
        # Try common attribute names first
        candidates = [
            "kept_columns_", "kept_cols_", "columns_", "feature_columns_",
            "kept_features_", "features_", "support_", "mask_",
            "kept_idx_", "indices_", "feature_indices_", "idx_"
        ]
        for name in candidates:
            if hasattr(self, name):
                val = getattr(self, name)
                if val is None:
                    continue
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                    arr = np.array(val, dtype=object)
                    if arr.dtype.kind in {"U", "S", "O"}:  # strings / objects
                        colnames = list(X.columns) if hasattr(X, "columns") else None
                        if colnames is not None:
                            keep = [c for c in arr.tolist() if c in colnames]
                            if keep:
                                return ("names", keep)
                    try:
                        arr_int = np.asarray(val, dtype=int)
                        if arr_int.ndim == 1 and len(arr_int) > 0:
                            if hasattr(X, "shape") and np.all((arr_int >= 0) & (arr_int < X.shape[1])):
                                return ("idx", arr_int.tolist())
                    except Exception:
                        pass

        for k, v in getattr(self, "__dict__", {}).items():
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
                arr = np.array(v, dtype=object)
                # names
                if arr.dtype.kind in {"U", "S", "O"} and hasattr(X, "columns"):
                    keep = [c for c in arr.tolist() if c in list(X.columns)]
                    if keep:
                        return ("names", keep)
                # indices
                try:
                    arr_int = np.asarray(v, dtype=int)
                    if arr_int.ndim == 1 and len(arr_int) > 0:
                        if hasattr(X, "shape") and np.all((arr_int >= 0) & (arr_int < X.shape[1])):
                            return ("idx", arr_int.tolist())
                except Exception:
                    pass

        return (None, None)

    def transform(self, X):
        # Preserve DataFrame for name-based selection
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        kind, keep = self._guess_kept(Xdf)
        if kind == "names":
            return Xdf.loc[:, keep].values
        if kind == "idx":
            return Xdf.iloc[:, keep].values
        raise RuntimeError(
            "ModelFractionSelector: could not detect kept feature mask/indices on the unpickled object. "
            "Open the artifact in Python and inspect the selector inside the pipeline to learn which "
            "attribute holds the kept features, then add it to the candidates list.\n\n"
            "Example debug snippet:\n"
            "  import joblib; art = joblib.load('mgmt_lgbm__5b.joblib')\n"
            "  sel = art['pipeline'].named_steps.get('modfrac') or art['pipeline'].named_steps.get('selector')\n"
            "  print(sel.__dict__.keys()); print(sel.__dict__)\n"
        )

# ---------------------------------------------------------

ARTIFACT = "mgmt_lgbm__5b.joblib"
TEST_XLSX = "derivatives/test_set__4.xlsx"
LABEL_COL = "MGMT"
CLASS_LABELS = ["Unmethylated (0)", "Methylated (1)"]
FIGNAME = "confusion_matrix__6.png"

def main():
    art = joblib.load(ARTIFACT)
    pipe = art["pipeline"]
    thr = float(art["threshold"])
    feat_cols = list(art["feature_cols"])
    calibrate = bool(art.get("calibrate", False))
    platt = art.get("platt_calibrator", None)

    df = pd.read_excel(TEST_XLSX)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in test file")

    y = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(-1).astype(int).values
    mask_eval = np.isin(y, [0, 1])
    y = y[mask_eval]

    # Ensure all training features exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    X = df.loc[mask_eval, feat_cols].copy()

    # Predict with same calibration + threshold as artifact
    proba_raw = pipe.predict_proba(X)[:, 1]
    proba = (platt.predict_proba(proba_raw.reshape(-1, 1))[:, 1]
             if (calibrate and platt is not None) else proba_raw)
    yhat = (proba >= thr).astype(int)

    cm = confusion_matrix(y, yhat, labels=[0, 1])

    # --- plot ---
    fontsize = 10
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm, cmap="Blues")

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = (cm / row_sums) * 100.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0f}%)",
                    ha="center", va="center", color="black", fontsize=fontsize)

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels([f"Pred {CLASS_LABELS[0]}", f"Pred {CLASS_LABELS[1]}"],
                       fontsize=fontsize-2, rotation=15)
    ax.set_yticklabels([f"True {CLASS_LABELS[0]}", f"True {CLASS_LABELS[1]}"],
                       fontsize=fontsize-2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGNAME, dpi=900, bbox_inches="tight")
    plt.close()
    print(f"[saved] {FIGNAME}")

if __name__ == "__main__":
    main()

