#!/usr/bin/env python3
"""
Before vs After correlation distributions per radiomics family (stacked violins).

- BEFORE: all numeric features in dataset__2.xlsx
- AFTER : only features listed in bin_class/feature_list__4.csv (subset of BEFORE)

For each family (First order, GLCM, GLRLM, GLSZM, GLDM, NGTDM), compute the distribution
of pairwise absolute Pearson correlations (upper triangle, no diagonal) and plot violins.

Adds:
- mean line (short horizontal bar),
- 5, 95th percentile whiskers (vertical line),
- prints/saves summary stats (mean/median/p5/p95) per family for BEFORE & AFTER.

Output:
  ./corr_violins__4c.png
  ./corr_stats__4c.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== EDIT PATHS IF NEEDED ==========
excel_path = Path("dataset__2.xlsx")              # full radiomics table
csv_path   = Path("bin_class/feature_list__4.csv")# kept features after filtering
outdir     = Path("./")
# ==========================================

outdir.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_selected_features(path_csv: Path) -> list[str]:
    df = pd.read_csv(path_csv)
    candidates = {"feature","features","name","names","feature_name","feature_names"}
    col = next((c for c in df.columns if str(c).strip().lower() in candidates), df.columns[0])
    return df[col].dropna().astype(str).tolist()

def prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric cols and drop constant ones (avoid NaNs in corr)."""
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
    return "First order"  # fallback

def pairwise_abs_r(vals_df: pd.DataFrame) -> np.ndarray:
    """Upper-triangle absolute Pearson r (exclude diagonal)."""
    p = vals_df.shape[1]
    if p < 2:
        return np.array([])
    C = vals_df.corr(method="pearson").abs().to_numpy()
    iu = np.triu_indices(p, k=1)
    vals = C[iu]
    return vals[np.isfinite(vals)]

def per_family_distributions(df_num: pd.DataFrame, families_order: list[str]):
    """Return distributions and labels; also a tidy stats table."""
    cols = list(df_num.columns)
    fam_map = {c: family(c) for c in cols}

    dists, labels, stats_rows = [], [], []
    for fam in families_order:
        fam_cols = [c for c in cols if fam_map[c] == fam]
        if len(fam_cols) >= 2:
            arr = pairwise_abs_r(df_num[fam_cols])
        else:
            arr = np.array([])

        labels.append(f"{fam}\n(n={len(fam_cols)})")
        dists.append(arr)

        if arr.size:
            p5, q1, med, mean, q3, p95 = np.nanpercentile(arr, [5, 25, 50, 50, 75, 95])
            # mean computed separately for precision
            mean = float(np.nanmean(arr))
        else:
            p5=q1=med=mean=q3=p95=np.nan

        stats_rows.append({
            "family": fam,
            "n_features": len(fam_cols),
            "pairs": int(arr.size),
            "p05": p5, "q25": q1, "median": med, "mean": mean, "q75": q3, "p95": p95
        })
    stats_df = pd.DataFrame(stats_rows)
    return dists, labels, stats_df

def violin_strip(ax, data, labels, title):
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    # Make the first violin (All features) a darker blue
    for pc in parts['bodies'][0:1]:
        pc.set_facecolor("royalblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.9)

    half_width = 0.30
    for i, arr in enumerate(data, start=1):
        if arr.size:
            p5, q1, med, q3, p95 = np.nanpercentile(arr, [5, 25, 50, 75, 95])
            mean = float(np.nanmean(arr))
            ax.vlines(i, p5, p95, lw=1.5, color="k")
            ax.vlines(i, q1, q3, lw=3, color="k")
            ax.scatter(i, med, s=18, color="k", zorder=3)
            ax.hlines(mean, i - half_width, i + half_width, lw=2, color="k")
        else:
            ax.text(i, 0.5, "—", ha="center", va="center", fontsize=10)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("|Pearson r| (pairwise)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

# ---------- load data ----------
df = pd.read_excel(excel_path)

# BEFORE
df_before = prepare_numeric(df)
print(f"[BEFORE] numeric features: {df_before.shape[1]}")

# AFTER (intersect with kept list)
keep = load_selected_features(csv_path)
keep = [c for c in keep if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
if not keep:
    raise RuntimeError("[ERROR] No matching numeric features from filtered list.")
df_after = prepare_numeric(df[keep])
print(f"[AFTER ] numeric features: {df_after.shape[1]}")

# families (fixed order)
fam_order = ["All features", "First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]

# Build an "All features" block + per-family blocks
def build_blocks(df_num):
    all_vals = pairwise_abs_r(df_num)
    all_label = f"All features\n(n={df_num.shape[1]})"
    fam_list = ["First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]
    fam_data, fam_labels, stats = per_family_distributions(df_num, fam_list)
    data = [all_vals] + fam_data
    labels = [all_label] + fam_labels

    # Prepend "All features" row in stats
    all_row = {
        "family": "All features",
        "n_features": df_num.shape[1],
        "pairs": int(all_vals.size),
        "p05": np.nanpercentile(all_vals, 5) if all_vals.size else np.nan,
        "q25": np.nanpercentile(all_vals, 25) if all_vals.size else np.nan,
        "median": np.nanpercentile(all_vals, 50) if all_vals.size else np.nan,
        "mean": float(np.nanmean(all_vals)) if all_vals.size else np.nan,
        "q75": np.nanpercentile(all_vals, 75) if all_vals.size else np.nan,
        "p95": np.nanpercentile(all_vals, 95) if all_vals.size else np.nan,
    }
    stats = pd.concat([pd.DataFrame([all_row]), stats], ignore_index=True)
    return data, labels, stats

data_b, labels, stats_b = build_blocks(df_before)
data_a, _,      stats_a = build_blocks(df_after)

# Merge stats (BEFORE vs AFTER) for console/CSV
stats_b["set"] = "BEFORE"
stats_a["set"] = "AFTER"
stats_out = pd.concat([stats_b, stats_a], ignore_index=True)
stats_out.to_csv(outdir / "corr_stats__4c.csv", index=False)

# Pretty console print
print("\n=== Summary (mean, median, p95) of |r| per family ===")
for fam in stats_out["family"].unique():
    sb = stats_out[(stats_out["family"]==fam) & (stats_out["set"]=="BEFORE")]
    sa = stats_out[(stats_out["family"]==fam) & (stats_out["set"]=="AFTER")]
    if not sb.empty and not sa.empty:
        print(f"{fam:12s} :: BEFORE mean={sb['mean'].values[0]:.3f}, median={sb['median'].values[0]:.3f}, p95={sb['p95'].values[0]:.3f}  |  "
              f"AFTER mean={sa['mean'].values[0]:.3f}, median={sa['median'].values[0]:.3f}, p95={sa['p95'].values[0]:.3f}")

# ---------- plot ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=500, sharey=True)

violin_strip(axes[0], data_b, labels, f"BEFORE filtering — {df_before.shape[1]} features")
violin_strip(axes[1], data_a, labels, f"AFTER filtering — {df_after.shape[1]} features")

fig.tight_layout()
png = outdir / "corr_violins__4c.png"
fig.savefig(png, bbox_inches="tight")
plt.close(fig)

print(f"\n[OK] Saved figure: {png}")
print(f"[OK] Saved stats CSV: {outdir/'corr_stats__4c.csv'}")

