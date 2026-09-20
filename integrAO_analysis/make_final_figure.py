# -*- coding: utf-8 -*-
"""生成最终 2x4 组合大图（delta=0.9/p.DMP=0.2 + epochs=40）。"""
import os, sys
import pandas as pd

os.environ["OMICS_DATA_DIR"] = r"d:\integrAO\vs\integrAO_analysis\_pdmp_scan\delta0.9_p0.1"
os.environ["OUT_TAG"] = "delta0.9p0.1_ndm8"

sys.path.insert(0, r"d:\integrAO\vs\integrAO_analysis")
import reproduce_figures as rf

BASE = r"d:\integrAO\vs\integrAO_analysis"

expr, protein, methyl, truelabel = rf.load_omics_data()
n_clusters = truelabel["cluster.id"].nunique()

df = pd.read_csv(os.path.join(BASE, "nmi_results_delta0.9p0.1_ndm8.csv"))
all_results = {}
for key_prefix in ["B_all_missing", "A_mRNA_complete", "A_Protein_complete", "A_Methyl_complete"]:
    all_results[key_prefix] = {
        "ratio": df["ratio"].tolist(),
        "IMGF": df[f"{key_prefix}_IMGF"].tolist(),
        "IntegrAO": df[f"{key_prefix}_IntegrAO"].tolist(),
        "KMeans(on intact feature)": df[f"{key_prefix}_KMeans"].tolist(),
        "NEMO": df[f"{key_prefix}_NEMO"].tolist(),
        "MSNE": df[f"{key_prefix}_MSNE"].tolist(),
    }

rf.plot_final_figure(all_results, expr, protein, methyl, truelabel, n_clusters)
print("完成")
