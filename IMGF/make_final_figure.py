# -*- coding: utf-8 -*-
"""生成最终 2x4 组合大图（上排 NMI vs overlap ratio，下排 UMAP）。"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reproduce_figures as rf

expr, protein, methyl, truelabel = rf.load_omics_data()
n_clusters = truelabel["cluster.id"].nunique()

tag = os.environ.get("OUT_TAG", "")
csv_name = f"nmi_results_{tag}.csv" if tag else "nmi_results.csv"
df = pd.read_csv(os.path.join(HERE, csv_name))
all_results = {}
for key_prefix in ["B_all_missing", "A_mRNA_complete", "A_Protein_complete", "A_Methyl_complete"]:
    all_results[key_prefix] = {
        "ratio": df["ratio"].tolist(),
        "IMGF": df[f"{key_prefix}_IMGF"].tolist(),
        "IntegrAO": df[f"{key_prefix}_IntegrAO"].tolist(),
        "NEMO": df[f"{key_prefix}_NEMO"].tolist(),
        "MSNE": df[f"{key_prefix}_MSNE"].tolist(),
    }

rf.plot_final_figure(all_results, expr, protein, methyl, truelabel, n_clusters)
print("完成")
