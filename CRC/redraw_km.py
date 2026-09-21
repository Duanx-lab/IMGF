# -*- coding: utf-8 -*-
"""用正确时间（月份，不除30.44）重画 CRC KM 曲线"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import os

crc_dir = r"D:\integrAO\vs\CRC"

# 加载生存数据
survival = pd.read_csv(os.path.join(crc_dir, "crc_survival.csv"))
survival = survival.rename(columns={
    "coad.clin....time..": "futime",
    "coad.clin....status..": "fustat",
    "coad.clin....cms..": "CMS"
})

# 加载聚类标签
labels_df = pd.read_csv(os.path.join(crc_dir, "crc_k4_labels.csv"))

n = len(survival)
survival["futime"] = pd.to_numeric(survival["futime"], errors="coerce")  # already in months
survival["fustat"] = pd.to_numeric(survival["fustat"], errors="coerce")

# 按 orig_index 对齐聚类标签
cluster_map = dict(zip(labels_df["orig_index"], labels_df["cluster"]))
survival["cluster"] = [cluster_map.get(i, -1) for i in range(n)]
survival = survival.dropna(subset=["futime", "fustat"])
survival = survival[survival["cluster"] >= 0]

print(f"有效样本: {len(survival)}, 时间范围: {survival['futime'].min():.1f} - {survival['futime'].max():.1f} months")
print(f"簇分布:\n{survival['cluster'].value_counts().sort_index()}")

n_clusters = 4
cluster_kmfs = {}
colors = ["#1f77b4", "#d62728", "#e377c2", "#17becf"]

for c in range(n_clusters):
    mask = survival["cluster"] == c
    if mask.sum() < 5:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=survival.loc[mask, "futime"],
        event_observed=survival.loc[mask, "fustat"],
        label=f"Cluster {c+1} (n={mask.sum()})"
    )
    cluster_kmfs[c] = kmf

valid_clusters = list(cluster_kmfs.keys())
mask_valid = survival["cluster"].isin(valid_clusters)
mlr = multivariate_logrank_test(
    survival.loc[mask_valid, "futime"],
    survival.loc[mask_valid, "cluster"],
    survival.loc[mask_valid, "fustat"]
)
mlr_pval = mlr.p_value

# CMS reference
cms_pval = None
cms_groups = survival["CMS"].dropna()
if len(cms_groups.unique()) >= 2:
    try:
        cms_mlr = multivariate_logrank_test(
            survival.loc[cms_groups.index, "futime"],
            survival.loc[cms_groups.index, "CMS"],
            survival.loc[cms_groups.index, "fustat"]
        )
        cms_pval = cms_mlr.p_value
    except:
        pass

# 画 KM 曲线
fig, ax = plt.subplots(figsize=(10, 7))

for c in range(n_clusters):
    if c not in cluster_kmfs:
        continue
    cluster_kmfs[c].plot_survival_function(
        ax=ax, color=colors[c], linewidth=2.5, ci_show=False)

ax.set_xlabel("Time (months)", fontsize=13)
ax.set_ylabel("Survival Probability", fontsize=13)
ax.set_title("IMGF Clustering Survival Analysis (K=4, overlap=0.8)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower left")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.2)

def _fmt_p(p):
    if p is None or (p <= 0 or np.isnan(p)) or p < 1e-15:
        return r"$p < 10^{-15}$"
    exp = int(np.floor(np.log10(p)))
    coef = p / (10 ** exp)
    return fr"$p = {coef:.2f} \times 10^{{{exp}}}$"

sig_star = ""
if mlr_pval < 0.001:
    sig_star = " ***"
elif mlr_pval < 0.01:
    sig_star = " **"
elif mlr_pval < 0.05:
    sig_star = " *"

p_text = "Multivariate log-rank\n" + _fmt_p(mlr_pval) + sig_star
ax.text(0.98, 0.98, p_text, transform=ax.transAxes,
        fontsize=14, fontweight="bold", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92))

if cms_pval is not None:
    cms_text = "CMS reference\n" + _fmt_p(cms_pval)
    ax.text(0.02, 0.98, cms_text, transform=ax.transAxes,
            fontsize=9, ha="left", va="top",
            color="#666666", style="italic")

plt.tight_layout()
out_path = os.path.join(crc_dir, "CRC_KM_K4_fixed.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"已保存: {out_path}")
print(f"log-rank p = {mlr_pval:.6f}")
