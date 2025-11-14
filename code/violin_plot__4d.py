#!/usr/bin/env python3
"""
Redundancy-focused BEFORE vs AFTER plot for radiomics features.

Reads:
  - dataset__2.xlsx          (all/original features)
  - feature_list__4.csv      (subset retained after correlation filtering)

Metrics per family (and overall):
  1) 95th percentile of |Pearson r| among feature pairs, lower is better
  2) Proportion of pairs with |r| > threshold (default 0.8), lower is better

Outputs:
  ./redundancy_tail_before_after.png
  ./redundancy_tail_before_after.csv   (table of metrics)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- edit if needed ----------
excel_path = Path("dataset__2.xlsx")
csv_path   = Path("bin_class/feature_list__4.csv")
outdir     = Path("./")
THRESHOLD  = 0.80
# ------------------------------------

outdir.mkdir(exist_ok=True, parents=True)

# ---------- helpers ----------
def load_selected_features(path_csv: Path) -> list[str]:
    df = pd.read_csv(path_csv)
    candidates = {"feature","features","name","names","feature_name","feature_names"}
    col = next((c for c in df.columns if str(c).strip().lower() in candidates), df.columns[0])
    return df[col].dropna().astype(str).tolist()

def prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric columns and drop constants (avoid NaNs in corr)."""
    num = df.select_dtypes(include=[np.number]).copy()
    const = num.nunique(dropna=True) <= 1
    if const.any():
        num = num.loc[:, ~const]
    return num

def family(name: str) -> str:
    n = str(name).lower()
    if "glcm"  in n: return "GLCM"
    if "glrlm" in n: return "GLRLM"
    if "glszm" in n: return "GLSZM"
    if "gldm"  in n: return "GLDM"
    if "ngtdm" in n: return "NGTDM"
    if "firstorder" in n or "first_order" in n: return "First order"
    return "First order"

def corr_tail_metrics(df_num: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Compute tail metrics per family and 'All features' bucket."""
    cols = df_num.columns.tolist()
    fam_map = {c: family(c) for c in cols}
    fams = ["All features", "First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]

    rows = []
    for fam in fams:
        fam_cols = cols if fam == "All features" else [c for c in cols if fam_map[c] == fam]
        if len(fam_cols) >= 2:
            C = df_num[fam_cols].corr(method="pearson").abs().to_numpy()
            iu = np.triu_indices(len(fam_cols), k=1)
            vals = C[iu]
            vals = vals[np.isfinite(vals)]
            p95  = float(np.nanpercentile(vals, 95)) if vals.size else np.nan
            prop = float(np.mean(vals > thr)) if vals.size else np.nan
            pairs = int(vals.size)
        else:
            p95, prop, pairs = np.nan, np.nan, 0
        rows.append({"family": fam, "n_features": len(fam_cols), "pairs": pairs,
                     "p95": p95, f"prop>|r|>{thr}": prop})
    return pd.DataFrame(rows)

# ---------- load data ----------
df = pd.read_excel(excel_path)

# BEFORE: all numeric
df_before = prepare_numeric(df)
# AFTER: intersect with list
keep = load_selected_features(csv_path)
keep = [c for c in keep if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
if not keep:
    raise RuntimeError("No matching numeric features from filtered list.")
df_after = prepare_numeric(df[keep])

# ---------- metrics ----------
metrics_before = corr_tail_metrics(df_before, THRESHOLD)
metrics_after  = corr_tail_metrics(df_after,  THRESHOLD)

# keep fixed family order
fam_order = ["All features", "First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]
metrics_before = metrics_before.set_index("family").loc[fam_order].reset_index()
metrics_after  = metrics_after.set_index("family").loc[fam_order].reset_index()

# ---------- plot ----------
x = np.arange(len(fam_order))
w = 0.42
dark = "#2a6fdb"   # BEFORE color
light = "#9bbcf4"  # AFTER color

fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), dpi=300, sharex=True)

# Top: 95th percentile bars
axes[0].bar(x - w/2, metrics_before["p95"],  width=w, color=dark,  label=f"BEFORE (n={df_before.shape[1]})")
axes[0].bar(x + w/2, metrics_after["p95"],   width=w, color=light, label=f"AFTER (n={df_after.shape[1]})")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("95th percentile |r|")
axes[0].set_title("High-correlation tail per family (lower is better)")
axes[0].legend(loc="upper right")
axes[0].grid(axis="y", alpha=0.25)

# Bottom: proportion of highly correlated pairs
prop_col = [c for c in metrics_before.columns if c.startswith("prop>|r|>")][0]
axes[1].bar(x - w/2, metrics_before[prop_col], width=w, color=dark,  label="BEFORE")
axes[1].bar(x + w/2, metrics_after[prop_col],  width=w, color=light, label="AFTER")
axes[1].set_yscale("log")
axes[1].set_ylim(1e-4, 1.0)
axes[1].set_ylabel(prop_col)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{fam}\n(n={nf})"
                         for fam, nf in zip(metrics_before["family"], metrics_before["n_features"])])
axes[1].grid(axis="y", alpha=0.25)

fig.tight_layout()
out_path = outdir / "redundancy_tail_before_after__4c.png"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Saved figure: {out_path}")

