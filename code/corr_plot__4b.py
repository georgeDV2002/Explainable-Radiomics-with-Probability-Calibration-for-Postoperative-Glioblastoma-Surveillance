#!/usr/bin/env python3
"""
Option 2 (MEAN): Group-level block correlation heatmaps (BEFORE vs AFTER)
- Uses MEAN(|r|) for between-group blocks.
- Diagonal is displayed as 1.0 (by definition). We still compute the
  within-group OFF-DIAGONAL mean for logging only.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== EDIT PATHS ======
excel_path = Path("dataset__2.xlsx")
csv_path   = Path("bin_class/feature_list__4.csv")
outdir     = Path("./outputs__4b")
# ========================

outdir.mkdir(parents=True, exist_ok=True)

def load_selected_features(path_csv: Path) -> list[str]:
    df = pd.read_csv(path_csv)
    candidates = {"feature","features","name","names","feature_name","feature_names"}
    col = next((c for c in df.columns if str(c).strip().lower() in candidates), df.columns[0])
    return df[col].dropna().astype(str).tolist()

# Load numeric
df  = pd.read_excel(excel_path)
num = df.select_dtypes(include=[np.number]).copy()

# BEFORE (all numeric)
corr_before = num.corr(method="pearson").abs()
print(f"[INFO] BEFORE: ALL numeric features → {corr_before.shape[0]}")

# AFTER (filtered)
keep = load_selected_features(csv_path)
keep = [c for c in keep if c in num.columns and pd.api.types.is_numeric_dtype(df[c])]
if not keep:
    raise RuntimeError("[ERROR] No filtered features matched numeric columns.")
corr_after = df[keep].corr(method="pearson").abs()
print(f"[INFO] AFTER: filtered list → {corr_after.shape[0]}")

# Family mapping
def family(name: str) -> str:
    n = name.lower()
    if "glcm"  in n: return "GLCM"
    if "glrlm" in n: return "GLRLM"
    if "glszm" in n: return "GLSZM"
    if "gldm"  in n: return "GLDM"
    if "ngtdm" in n: return "NGTDM"
    if "firstorder" in n or "first_order" in n: return "FirstOrder"
    return "FirstOrder"

def group_block_mean(corr: pd.DataFrame):
    """Return (M, groups, within_offdiag_means) where:
       - M[i,j] = MEAN |r| between group i and j (diag forced to 1.0)
       - within_offdiag_means[i] = mean of within-group OFF-DIAGONAL |r| (for logging)"""
    fam = {c: family(c) for c in corr.columns}
    groups = sorted(set(fam.values()))
    G = len(groups)
    M = np.full((G, G), np.nan, dtype=float)
    within_offdiag_means = {}

    for i, gi in enumerate(groups):
        cols_i = [c for c,f in fam.items() if f == gi]
        for j, gj in enumerate(groups):
            cols_j = [c for c,f in fam.items() if f == gj]
            block = corr.loc[cols_i, cols_j].to_numpy()

            if i == j:
                # mean of OFF-DIAGONAL within-group correlations (for reference)
                n = block.shape[0]
                if n >= 2:
                    off = block[~np.eye(n, dtype=bool)]
                    within_offdiag_means[gi] = float(np.nanmean(off)) if off.size else np.nan
                else:
                    within_offdiag_means[gi] = np.nan
                # display 1.0 on the diagonal
                M[i, j] = 1.0
            else:
                M[i, j] = float(np.nanmean(block)) if block.size else np.nan

    return M, groups, within_offdiag_means

M_before, groups_b, within_b = group_block_mean(corr_before)
M_after,  groups_a, within_a = group_block_mean(corr_after)

# Align groups across before/after (union)
groups = sorted(set(groups_b).union(groups_a))
def realign(M, groups_src, groups_tgt):
    idx = {g:i for i,g in enumerate(groups_src)}
    out = np.full((len(groups_tgt), len(groups_tgt)), np.nan, dtype=float)
    for i,g1 in enumerate(groups_tgt):
        for j,g2 in enumerate(groups_tgt):
            if g1 in idx and g2 in idx:
                out[i,j] = M[idx[g1], idx[g2]]
    return out

M_before = realign(M_before, groups_b, groups)
M_after  = realign(M_after,  groups_a, groups)

VMIN, VMAX = 0.0, 1.0

def plot_block(M, groups, title, out_png):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(M, vmin=VMIN, vmax=VMAX, cmap="viridis")
    ax.set_xticks(range(len(groups))); ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right"); ax.set_yticklabels(groups)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Mean |Pearson r|")
    ax.set_title(title); fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

# Save individual BEFORE/AFTER blocks
plot_block(M_before, groups, "Group-level MEAN |r| — BEFORE", outdir / "corr_groups_block__before__4b.png")
plot_block(M_after,  groups, "Group-level MEAN |r| — AFTER",  outdir / "corr_groups_block__after__4b.png")

# Combined side-by-side (shared colorbar)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
ims = []
for ax, M, title in zip(axes, [M_before, M_after], ["BEFORE", "AFTER"]):
    im = ax.imshow(M, vmin=VMIN, vmax=VMAX, cmap="viridis")
    ims.append(im)
    ax.set_xticks(range(len(groups))); ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_yticklabels(groups if ax is axes[0] else [])
    ax.set_title(f"Group-level MEAN |r| — {title}")

cbar = fig.colorbar(ims[1], ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
cbar.set_label("Mean |Pearson r|")
fig.tight_layout()
fig.savefig(outdir / "corr_groups_block__before_after__4b.png", bbox_inches="tight")
plt.close(fig)

# Console summary for within-group off-diagonal means
print("\nWithin-group OFF-DIAGONAL mean |r| (not plotted):")
print("BEFORE:", within_b)
print("AFTER :", within_a)

print("[OK] Saved group-level BEFORE/AFTER (individual and combined) in", outdir.resolve())

