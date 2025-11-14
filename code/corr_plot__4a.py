#!/usr/bin/env python3
"""
Option 1: Clustered full correlation heatmaps (BEFORE vs AFTER filtering)
- Input: dataset__2.xlsx (radiomics table)
         bin_class/feature_list__4.csv (filtered features)
- Output:
    outputs/corr_all_clustered__before__4a.png
    outputs/corr_all_clustered__after__4a.png
    outputs/corr_all_clustered__order__before.csv
    outputs/corr_all_clustered__order__after.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====== EDIT PATHS ======
excel_path = Path("dataset__2.xlsx")
csv_path   = Path("bin_class/feature_list__4.csv")
outdir     = Path("./outputs__4a")
# ========================

outdir.mkdir(parents=True, exist_ok=True)

def load_selected_features(path_csv: Path) -> list[str]:
    df = pd.read_csv(path_csv)
    candidates = {"feature","features","name","names","feature_name","feature_names"}
    col = next((c for c in df.columns if str(c).strip().lower() in candidates), df.columns[0])
    return df[col].dropna().astype(str).tolist()

# Load data
df  = pd.read_excel(excel_path)
num = df.select_dtypes(include=[np.number]).copy()

# BEFORE: all numeric features
corr_before = num.corr(method="pearson").abs()
print(f"[INFO] BEFORE: using ALL numeric features → {corr_before.shape[0]}")

# AFTER: filtered by CSV (intersect with numeric)
keep = load_selected_features(csv_path)
keep = [c for c in keep if c in num.columns and pd.api.types.is_numeric_dtype(df[c])]
if not keep:
    raise RuntimeError("[ERROR] No filtered features matched numeric columns.")
corr_after = df[keep].corr(method="pearson").abs()
print(f"[INFO] AFTER: using filtered list → {corr_after.shape[0]}")

# Shared color scale
VMIN, VMAX = 0.0, 1.0

def make_clustermap(corr: pd.DataFrame, title: str, out_png: Path, out_order_csv: Path):
    sns.set_context("notebook")
    g = sns.clustermap(
        corr,
        method="average", metric="euclidean",
        cmap="viridis", vmin=VMIN, vmax=VMAX,
        row_cluster=True, col_cluster=True,
        xticklabels=False, yticklabels=False,
        linewidths=0,
        dendrogram_ratio=0.15, cbar_pos=(0.02, 0.8, 0.03, 0.18)
    )
    g.fig.suptitle(title, y=1.02)
    g.savefig(out_png, dpi=800, bbox_inches="tight")
    plt.close(g.fig)

    # Save reordered feature list (row order)
    order_idx = g.dendrogram_row.reordered_ind
    ordered_features = corr.index.to_numpy()[order_idx]
    pd.Series(ordered_features, name="feature").to_csv(out_order_csv, index=False)

# Make both heatmaps
make_clustermap(corr_before,
                "All features — clustered |r| (BEFORE filtering)",
                outdir / "corr_all_clustered__before__4a.png",
                outdir / "corr_all_clustered__order__before.csv")

make_clustermap(corr_after,
                "Filtered features — clustered |r| (AFTER filtering)",
                outdir / "corr_all_clustered__after__4a.png",
                outdir / "corr_all_clustered__order__after.csv")

print("[OK] Saved BEFORE and AFTER clustered heatmaps + order CSVs in", outdir.resolve())

