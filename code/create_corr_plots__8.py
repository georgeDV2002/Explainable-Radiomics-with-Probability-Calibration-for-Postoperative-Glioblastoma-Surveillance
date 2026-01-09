#!/usr/bin/env python3
"""
Correlation reports for radiomics2 MGMT pipeline (all-in-one).

Generates:
  1) Clustered full correlation heatmaps (BEFORE vs AFTER)  [seaborn if available]
  2) Group-level block (family) heatmaps (BEFORE vs AFTER + combined)
  3) Violin distributions of pairwise |r| per family (BEFORE vs AFTER)
  4) Redundancy tail metrics: p95 |r| and proportion(|r|>thr) per family (BEFORE vs AFTER)

Defaults match your radiomics2 layout:
  BEFORE: derivatives/traincv_set__4.xlsx
  AFTER : features from derivatives/feature_list__4.csv
          (fallback to mgmt_lgbm__5b.joblib -> 'feature_cols' if CSV is missing)

Outputs are saved under: derivatives/plots_corr__8/
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# ====== CONFIG (edit if needed) ======
BEFORE_XLSX   = Path("derivatives/traincv_set__4.xlsx")
AFTER_LIST_CSV= Path("derivatives/feature_list__4.csv")   # Top-K list
ARTIFACT_PATH = Path("mgmt_lgbm__5b.joblib")              # fallback for feature list
LABEL_COL     = "MGMT"
META_DROP     = {"subject_id","Patient_ID","Subject_ID","patient_id","dataset","sample_id","roi","timepoint", LABEL_COL}
OUTDIR        = Path("plots_corr__8")
THRESHOLD     = 0.80   # for redundancy tail
# =====================================

OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- optional seaborn ----------
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False
    warnings.warn("[note] seaborn not found — clustered heatmaps will be skipped.")

# ---------- helpers ----------
def prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric columns not in META_DROP and drop constants."""
    cols = [c for c in df.columns if c not in META_DROP]
    num = df.loc[:, cols].select_dtypes(include=[np.number]).copy()
    const = num.nunique(dropna=True) <= 1
    if const.any():
        num = num.loc[:, ~const]
    return num

def load_after_feature_list() -> list[str]:
    """Load AFTER feature list from CSV; fallback to artifact if CSV missing."""
    if AFTER_LIST_CSV.exists():
        df = pd.read_csv(AFTER_LIST_CSV)
        candidates = {"feature","features","name","names","feature_name","feature_names"}
        col = next((c for c in df.columns if str(c).strip().lower() in candidates), df.columns[0])
        feats = df[col].dropna().astype(str).tolist()
        if len(feats):
            print(f"[after list] Loaded {len(feats)} features from {AFTER_LIST_CSV}")
            return feats
    # Fallback to artifact
    try:
        import joblib
        art = joblib.load(ARTIFACT_PATH)
        feats = list(art.get("feature_cols", []))
        if feats:
            print(f"[after list] Using {len(feats)} feature_cols from {ARTIFACT_PATH}")
            return feats
    except Exception as e:
        warnings.warn(f"[after list] Fallback to artifact failed: {e}")
    raise RuntimeError("Could not determine AFTER feature list. Provide feature_list__4.csv or artifact.")

def family(name: str) -> str:
    n = str(name).lower()
    if "glcm"  in n: return "GLCM"
    if "glrlm" in n: return "GLRLM"
    if "glszm" in n: return "GLSZM"
    if "gldm"  in n: return "GLDM"
    if "ngtdm" in n: return "NGTDM"
    if "firstorder" in n or "first_order" in n: return "First order"
    return "First order"

def corr_abs(df_num: pd.DataFrame) -> pd.DataFrame:
    return df_num.corr(method="pearson").abs()

def pairwise_abs_r(vals_df: pd.DataFrame) -> np.ndarray:
    p = vals_df.shape[1]
    if p < 2:
        return np.array([])
    C = corr_abs(vals_df).to_numpy()
    iu = np.triu_indices(p, k=1)
    vals = C[iu]
    return vals[np.isfinite(vals)]

def per_family_distributions(df_num: pd.DataFrame, families_order: list[str]):
    cols = list(df_num.columns)
    fam_map = {c: family(c) for c in cols}
    dists, labels, rows = [], [], []
    for fam in families_order:
        fam_cols = [c for c in cols if fam_map[c] == fam] if fam != "All features" else cols
        if len(fam_cols) >= 2:
            arr = pairwise_abs_r(df_num[fam_cols])
            p5, q1, med, q3, p95 = np.nanpercentile(arr, [5, 25, 50, 75, 95]) if arr.size else (np.nan,)*5
            mean = float(np.nanmean(arr)) if arr.size else np.nan
            pairs = int(arr.size)
        else:
            arr = np.array([])
            p5=q1=med=q3=p95=mean=np.nan; pairs=0
        label = f"{fam}\n(n={len(fam_cols)})"
        dists.append(arr); labels.append(label)
        rows.append({"family": fam, "n_features": len(fam_cols), "pairs": pairs,
                     "p05": p5, "q25": q1, "median": med, "mean": mean, "q75": q3, "p95": p95})
    return dists, labels, pd.DataFrame(rows)

def group_block_mean(corr: pd.DataFrame):
    """Return (M, groups, within_offdiag) where:
       M[i,j] = mean |r| between group i and j (diag forced to 1.0)."""
    fam = {c: family(c) for c in corr.columns}
    groups = sorted(set(fam.values()))
    G = len(groups)
    M = np.full((G, G), np.nan)
    within_off = {}
    for i, gi in enumerate(groups):
        ci = [c for c,f in fam.items() if f == gi]
        for j, gj in enumerate(groups):
            cj = [c for c,f in fam.items() if f == gj]
            block = corr.loc[ci, cj].to_numpy()
            if i == j:
                n = block.shape[0]
                if n >= 2:
                    off = block[~np.eye(n, dtype=bool)]
                    within_off[gi] = float(np.nanmean(off)) if off.size else np.nan
                else:
                    within_off[gi] = np.nan
                M[i, j] = 1.0
            else:
                M[i, j] = float(np.nanmean(block)) if block.size else np.nan
    return M, groups, within_off

def plot_violin(ax, data, labels, title):
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    # highlight first violin (All features) if present
    if len(parts["bodies"]) >= 1:
        parts["bodies"][0].set_facecolor("royalblue")
        parts["bodies"][0].set_edgecolor("black")
        parts["bodies"][0].set_alpha(0.9)
    half = 0.30
    for i, arr in enumerate(data, start=1):
        if arr.size:
            p5, q1, med, q3, p95 = np.nanpercentile(arr, [5, 25, 50, 75, 95])
            mean = float(np.nanmean(arr))
            ax.vlines(i, p5, p95, lw=1.5, color="k")
            ax.vlines(i, q1, q3, lw=3, color="k")
            ax.scatter(i, med, s=18, color="k", zorder=3)
            ax.hlines(mean, i - half, i + half, lw=2, color="k")
        else:
            ax.text(i, 0.5, "—", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("|Pearson r| (pairwise)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

# ---------- load BEFORE ----------
df_before_full = pd.read_excel(BEFORE_XLSX)
df_before = prepare_numeric(df_before_full)
print(f"[BEFORE] numeric features: {df_before.shape[1]}")

# ---------- load AFTER ----------
after_list = load_after_feature_list()
keep = [c for c in after_list if c in df_before_full.columns and pd.api.types.is_numeric_dtype(df_before_full[c])]
if not keep:
    raise RuntimeError("[AFTER] No matching numeric features from list/artifact.")
df_after = prepare_numeric(df_before_full[keep])
print(f"[AFTER ] numeric features: {df_after.shape[1]}")

# ---------- 1) Clustered heatmaps ----------
if HAS_SEABORN:
    VMIN, VMAX = 0.0, 1.0
    def make_clustermap(df_num, title, out_png, out_order_csv):
        corr = corr_abs(df_num)
        sns.set_context("notebook")
        g = sns.clustermap(
            corr, method="average", metric="euclidean",
            cmap="viridis", vmin=VMIN, vmax=VMAX,
            row_cluster=True, col_cluster=True,
            xticklabels=False, yticklabels=False,
            linewidths=0, dendrogram_ratio=0.15, cbar_pos=(0.02, 0.8, 0.03, 0.18)
        )
        g.fig.suptitle(title, y=1.02)
        g.savefig(out_png, dpi=800, bbox_inches="tight")
        plt.close(g.fig)
        # save dendrogram order
        order_idx = g.dendrogram_row.reordered_ind
        ordered_features = corr.index.to_numpy()[order_idx]
        pd.Series(ordered_features, name="feature").to_csv(out_order_csv, index=False)

    make_clustermap(
        df_before, "All features — clustered |r| (BEFORE filtering)",
        OUTDIR / "corr_all_clustered__before__5b.png",
        OUTDIR / "corr_all_clustered__order__before__5b.csv"
    )
    make_clustermap(
        df_after, "Filtered features — clustered |r| (AFTER filtering)",
        OUTDIR / "corr_all_clustered__after__5b.png",
        OUTDIR / "corr_all_clustered__order__after__5b.csv"
    )
    print("[OK] Clustered heatmaps saved.")
else:
    print("[skip] seaborn not available — clustered heatmaps not generated.")

# ---------- 2) Group block heatmaps ----------
C_before = corr_abs(df_before)
C_after  = corr_abs(df_after)
M_b, groups_b, within_b = group_block_mean(C_before)
M_a, groups_a, within_a = group_block_mean(C_after)
groups = sorted(set(groups_b).union(groups_a))
def realign(M, g_src, g_tgt):
    idx = {g:i for i,g in enumerate(g_src)}
    out = np.full((len(g_tgt), len(g_tgt)), np.nan)
    for i,g1 in enumerate(g_tgt):
        for j,g2 in enumerate(g_tgt):
            if g1 in idx and g2 in idx:
                out[i,j] = M[idx[g1], idx[g2]]
    return out
M_b = realign(M_b, groups_b, groups)
M_a = realign(M_a, groups_a, groups)

VMIN, VMAX = 0.0, 1.0
def plot_block(M, groups, title, out_png):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(M, vmin=VMIN, vmax=VMAX, cmap="viridis")
    ax.set_xticks(range(len(groups))); ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right"); ax.set_yticklabels(groups)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("Mean |Pearson r|")
    ax.set_title(title); fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

plot_block(M_b, groups, "Group-level MEAN |r| — BEFORE", OUTDIR / "corr_groups_block__before__5b.png")
plot_block(M_a, groups, "Group-level MEAN |r| — AFTER",  OUTDIR / "corr_groups_block__after__5b.png")

# combined side-by-side
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
for ax, M, title in zip(axes, [M_b, M_a], ["BEFORE", "AFTER"]):
    im = ax.imshow(M, vmin=VMIN, vmax=VMAX, cmap="viridis")
    ax.set_xticks(range(len(groups))); ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_yticklabels(groups if ax is axes[0] else [])
    ax.set_title(f"Group-level MEAN |r| — {title}")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
cbar.set_label("Mean |Pearson r|")
fig.tight_layout()
fig.savefig(OUTDIR / "corr_groups_block__before_after__5b.png", bbox_inches="tight")
plt.close(fig)

print("\nWithin-group OFF-DIAGONAL mean |r| (not plotted):")
print("BEFORE:", within_b)
print("AFTER :", within_a)

# ---------- 3) Violins per family ----------
fam_order = ["All features", "First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]

def build_blocks(df_num):
    # All + families
    all_vals = pairwise_abs_r(df_num)
    all_label = f"All features\n(n={df_num.shape[1]})"
    fams = ["First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]
    d_fam, l_fam, stats = per_family_distributions(df_num, fams)
    data = [all_vals] + d_fam
    labels = [all_label] + [f"{f}\n(n={int(stats.loc[stats['family']==f,'n_features'].values[0])})" for f in fams]
    # Prepend all row in stats
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

data_b, labels_b, stats_b = build_blocks(df_before)
data_a, labels_a, stats_a = build_blocks(df_after)

# save stats
stats_b["set"] = "BEFORE"
stats_a["set"] = "AFTER"
stats_out = pd.concat([stats_b, stats_a], ignore_index=True)
stats_out.to_csv(OUTDIR / "corr_stats__violins__5b.csv", index=False)

# plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=500, sharey=True)
plot_violin(axes[0], data_b, labels_b, f"BEFORE filtering — {df_before.shape[1]} features")
plot_violin(axes[1], data_a, labels_a, f"AFTER filtering — {df_after.shape[1]} features")
fig.tight_layout()
fig.savefig(OUTDIR / "corr_violins__5b.png", bbox_inches="tight")
plt.close(fig)
print("[OK] Violins saved.")

# ---------- 4) Redundancy tail metrics ----------
def corr_tail_metrics(df_num: pd.DataFrame, thr: float) -> pd.DataFrame:
    cols = df_num.columns.tolist()
    fam_map = {c: family(c) for c in cols}
    fams = ["All features", "First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]
    rows = []
    for fam in fams:
        fam_cols = cols if fam == "All features" else [c for c in cols if fam_map[c] == fam]
        if len(fam_cols) >= 2:
            C = corr_abs(df_num[fam_cols]).to_numpy()
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

mb = corr_tail_metrics(df_before, THRESHOLD)
ma = corr_tail_metrics(df_after,  THRESHOLD)

fam_order = ["All features", "First order", "GLCM", "GLRLM", "GLSZM", "GLDM", "NGTDM"]
mb = mb.set_index("family").loc[fam_order].reset_index()
ma = ma.set_index("family").loc[fam_order].reset_index()

x = np.arange(len(fam_order)); w = 0.42
dark = "#2a6fdb"; light = "#9bbcf4"

fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), dpi=300, sharex=True)
# p95 bars
axes[0].bar(x - w/2, mb["p95"], width=w, color=dark,  label=f"BEFORE (n={df_before.shape[1]})")
axes[0].bar(x + w/2, ma["p95"], width=w, color=light, label=f"AFTER (n={df_after.shape[1]})")
axes[0].set_ylim(0, 1); axes[0].set_ylabel("95th percentile |r|")
axes[0].set_title("High-correlation tail per family (lower is better)")
axes[0].legend(loc="upper right"); axes[0].grid(axis="y", alpha=0.25)

# proportion bars (log scale)
prop_col = [c for c in mb.columns if c.startswith("prop>|r|>")][0]
axes[1].bar(x - w/2, mb[prop_col], width=w, color=dark,  label="BEFORE")
axes[1].bar(x + w/2, ma[prop_col], width=w, color=light, label="AFTER")
axes[1].set_yscale("log"); axes[1].set_ylim(1e-4, 1.0)
axes[1].set_ylabel(prop_col)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{fam}\n(n={nf})" for fam, nf in zip(mb["family"], mb["n_features"])])
axes[1].grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUTDIR / "redundancy_tail_before_after__5b.png", bbox_inches="tight")
plt.close(fig)
pd.concat([mb.assign(set="BEFORE"), ma.assign(set="AFTER")], ignore_index=True)\
  .to_csv(OUTDIR / "redundancy_tail_before_after__5b.csv", index=False)

print(f"[DONE] All outputs saved in {OUTDIR.resolve()}")

