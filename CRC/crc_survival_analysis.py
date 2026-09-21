"""
CRC 三组学 missing 数据分析：IntegrAO 聚类 + 生存分析。

数据: mRNA (GE) + DNA 甲基化 (ME) + miRNA (MI), 297 样本
场景: overlap ratio = 0.8，三组学均有部分缺失
方法: IntegrAO 融合 → Diffusion Map → KMeans → Kaplan-Meier 生存分析
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import sys, os, warnings
warnings.filterwarnings("ignore")

BASE_DIR = r"d:\integrAO\vs"
INTEGRAO_DIR = os.path.join(BASE_DIR, "IntegrAO")
sys.path.insert(0, INTEGRAO_DIR)
sys.path.insert(0, BASE_DIR)

from integrao.main import dist2, integrao_fuse
from integrao.util import data_indexing
import snf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


# ============================================================
# 1. 加载 CRC 数据
# ============================================================
def load_crc_data():
    """加载 CRC 三组学数据 + 生存数据（从 CSV 文件）"""
    crc_dir = os.path.join(BASE_DIR, "CRC")

    ge = pd.read_csv(os.path.join(crc_dir, "crc_mRNA.csv"))       # mRNA: 297 x 20531
    me = pd.read_csv(os.path.join(crc_dir, "crc_methyl.csv"))     # 甲基化: 297 x 2080
    mi = pd.read_csv(os.path.join(crc_dir, "crc_miRNA.csv"))      # miRNA: 297 x 705
    survival = pd.read_csv(os.path.join(crc_dir, "crc_survival.csv"))  # CMS, time, status

    print(f"mRNA: {ge.shape}, 甲基化: {me.shape}, miRNA: {mi.shape}")
    print(f"生存数据: {survival.shape}")
    print(f"事件数: {int(survival['coad.clin....status..'].sum())}/{len(survival)}")
    return ge, me, mi, survival


# ============================================================
# 2. 特征筛选：保留每个组学方差最大的 top_n 特征
# ============================================================
def select_top_variable_features(data_list, top_n=2000):
    """对每个组学数据，按方差排序选取 top_n 特征"""
    selected = []
    for df in data_list:
        variances = df.var(axis=0)
        top_idx = variances.sort_values(ascending=False).head(top_n).index
        selected.append(df[top_idx])
        print(f"  特征筛选: {df.shape} -> {selected[-1].shape}")
    return selected


# ============================================================
# 3. 三组学 missing 数据拆分 (overlap ratio = 0.8)
# ============================================================
def split_three_omics_missing(data_list, ratio=0.8, seed=42):
    """
    三组学均部分缺失：ratio x N common + unique/view = Union。
    ratio: 三组学共有的样本比例
    """
    np.random.seed(seed)
    n_total = len(data_list[0])
    n_common = int(n_total * ratio)
    n_unique_each = (n_total - n_common) // 3

    leftover = n_total - n_common - 3 * n_unique_each
    n_common += leftover

    indices = np.arange(n_total)
    np.random.shuffle(indices)
    common_idx = set(indices[:n_common])

    u_start = n_common
    u1 = set(indices[u_start:u_start + n_unique_each])
    u_start += n_unique_each
    u2 = set(indices[u_start:u_start + n_unique_each])
    u_start += n_unique_each
    u3 = set(indices[u_start:u_start + n_unique_each])

    def build_view(df, my_unique_set, prefix):
        my_idx = list(common_idx) + list(my_unique_set)
        sub = df.iloc[my_idx].copy()
        new_names = [f"common_{i}" if i in common_idx else f"{prefix}_{i}"
                     for i in my_idx]
        sub.index = new_names
        return sub

    d0 = build_view(data_list[0], u1, "ge")
    d1 = build_view(data_list[1], u2, "me")
    d2 = build_view(data_list[2], u3, "mi")

    datasets = [d0, d1, d2]

    all_names = sorted(set(d0.index) | set(d1.index) | set(d2.index))
    union_to_orig = {}
    for name in all_names:
        orig_idx = int(name.split("_")[1])
        union_to_orig[name] = orig_idx

    stats = {
        "ratio": ratio, "n_total": n_total,
        "n_common": len(common_idx),
        "n_unique_per_view": n_unique_each,
        "n_union": len(all_names),
        "view_sizes": [len(d) for d in datasets],
    }
    print(f"  重叠比 {ratio}: common={stats['n_common']}, unique/view={n_unique_each}, "
          f"union={stats['n_union']}, 视图大小={stats['view_sizes']}")
    return datasets, union_to_orig, stats


# ============================================================
# 4. IntegrAO 融合 + Diffusion Map + KMeans
# ============================================================
def integrao_cluster(datasets, n_clusters, neighbor_size=20, fusing_iteration=20):
    """IntegrAO 融合 -> DM -> KMeans，返回聚类标签（按 union 样本顺序）"""
    (dicts_common, dicts_commonIndex, dict_sampleToIndexs,
     dicts_unique, original_order, dict_original_order) = data_indexing(datasets)

    S_dfs = []
    for i, view in enumerate(datasets):
        dist_mat = dist2(view.values, view.values)
        S_mat = snf.compute.affinity_matrix(dist_mat, K=neighbor_size, mu=0.5)
        S_df = pd.DataFrame(data=S_mat, index=original_order[i],
                            columns=original_order[i])
        S_dfs.append(S_df)

    fused = integrao_fuse(S_dfs.copy(), dicts_common=dicts_common,
                          dicts_unique=dicts_unique, original_order=original_order,
                          neighbor_size=neighbor_size, fusing_iteration=fusing_iteration,
                          normalization_factor=1.0)

    # 构建 union 相似度矩阵 -> DM
    union_samples = sorted(dict_sampleToIndexs.keys())
    n_union = len(union_samples)
    S_sum = np.zeros((n_union, n_union))
    S_count = np.zeros((n_union, n_union))

    for mat in fused:
        arr = mat.values if isinstance(mat, pd.DataFrame) else mat
        idx_map = (mat.index.tolist() if isinstance(mat, pd.DataFrame)
                   else list(range(arr.shape[0])))
        name_to_vi = {idx_map[k]: k for k in range(len(idx_map))}
        for u in range(n_union):
            su = union_samples[u]
            if su not in name_to_vi:
                continue
            vu = name_to_vi[su]
            for v in range(u, n_union):
                sv = union_samples[v]
                if sv not in name_to_vi:
                    continue
                vv = name_to_vi[sv]
                val = arr[vu, vv]
                S_sum[u, v] += val
                S_sum[v, u] += val
                S_count[u, v] += 1
                S_count[v, u] += 1

    S_count[S_count == 0] = 1
    S_union = S_sum / S_count

    # Diffusion Map
    n_dm = 5
    W = np.maximum(S_union, 0)
    W = (W + W.T) / 2
    D = np.sum(W, axis=1)
    D[D == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    P_sym = D_inv_sqrt @ W @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(P_sym)
    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    embedding = np.zeros((n_union, n_dm))
    for i in range(n_dm):
        lam = eigenvalues[idx[i + 1]]
        if lam > 1e-10:
            embedding[:, i] = lam * (D_inv_sqrt @ eigenvectors[:, i + 1])

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(embedding)

    return labels, embedding, union_samples


# ============================================================
# 5. 生存分析
# ============================================================
def survival_analysis(cluster_labels, union_samples, union_to_orig, survival,
                     n_clusters, save_path):
    """
    Kaplan-Meier 生存曲线 + log-rank 检验。
    参考: CMS 分型
    """
    # CRC 生存列
    survival = survival.copy()
    survival = survival.rename(columns={
        "coad.clin....time..": "futime",
        "coad.clin....status..": "fustat",
        "coad.clin....cms..": "CMS"
    })

    # 为每个样本分配簇标签
    sample_cluster = {}
    for i, name in enumerate(union_samples):
        orig_idx = union_to_orig[name]
        sample_cluster[orig_idx] = cluster_labels[i]

    surv_df = survival.copy()
    surv_df["futime"] = pd.to_numeric(surv_df["futime"], errors="coerce")  # already in months
    surv_df["fustat"] = pd.to_numeric(surv_df["fustat"], errors="coerce")
    surv_df["cluster"] = [sample_cluster.get(i, -1) for i in range(len(surv_df))]
    surv_df = surv_df.dropna(subset=["futime", "fustat"])
    surv_df = surv_df[surv_df["cluster"] >= 0]

    # 多变量 log-rank
    from itertools import combinations
    from lifelines.statistics import multivariate_logrank_test

    cluster_sizes = surv_df["cluster"].value_counts().sort_index()
    cluster_kmfs = {}
    for c in range(n_clusters):
        mask = surv_df["cluster"] == c
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=surv_df.loc[mask, "futime"],
            event_observed=surv_df.loc[mask, "fustat"],
            label=f"Cluster {c+1} (n={mask.sum()})"
        )
        cluster_kmfs[c] = kmf

    valid_clusters = [c for c in range(n_clusters) if c in cluster_kmfs]
    mask_valid = surv_df["cluster"].isin(valid_clusters)
    mlr = multivariate_logrank_test(
        surv_df.loc[mask_valid, "futime"],
        surv_df.loc[mask_valid, "cluster"],
        surv_df.loc[mask_valid, "fustat"]
    )
    mlr_pval = mlr.p_value

    # 两两 log-rank
    pairwise_results = []
    significant_pairs = 0
    for c1, c2 in combinations(valid_clusters, 2):
        mask1 = surv_df["cluster"] == c1
        mask2 = surv_df["cluster"] == c2
        lr = logrank_test(
            surv_df.loc[mask1, "futime"], surv_df.loc[mask2, "futime"],
            surv_df.loc[mask1, "fustat"], surv_df.loc[mask2, "fustat"]
        )
        pairwise_results.append((c1, c2, lr.p_value,
                                 int(mask1.sum()), int(mask2.sum())))
        if lr.p_value < 0.05:
            significant_pairs += 1

    # CMS 参考 (CMS1-4 组间)
    cms_pval = None
    cms_groups = surv_df["CMS"].dropna()
    cms_valid_idx = cms_groups.index
    surv_cms = surv_df.loc[cms_valid_idx]
    cms_types = surv_cms["CMS"].unique()
    if len(cms_types) >= 2:
        from lifelines.statistics import multivariate_logrank_test as mlr_test
        try:
            cms_mlr = mlr_test(
                surv_cms["futime"],
                surv_cms["CMS"],
                surv_cms["fustat"]
            )
            cms_pval = cms_mlr.p_value
        except Exception:
            cms_pval = None

    # --- 绘制 KM 曲线 ---
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#F9C74F", "#577590", "#F3722C", "#6A994E"]

    for c in range(n_clusters):
        if c not in cluster_kmfs:
            continue
        cluster_kmfs[c].plot_survival_function(
            ax=ax, color=colors[c], linewidth=2.5, ci_show=False)

    ax.set_xlabel("Time (months)", fontsize=13)
    ax.set_ylabel("Survival Probability", fontsize=13)
    ax.set_title(f"CRC — IntegrAO Clustering Survival Analysis (K={n_clusters}, overlap=0.8)",
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
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    pdf_path = save_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  KM曲线已保存: {save_path}, {pdf_path}")

    # --- 打印统计结果 ---
    print(f"\n  === Log-rank test (K={n_clusters}) ===")
    for c1, c2, pv, n1, n2 in pairwise_results:
        sig = " *" if pv < 0.05 else ""
        print(f"    Cluster {c1+1}(n={n1}) vs Cluster {c2+1}(n={n2}): "
              f"p={pv:.4f}{sig}")
    print(f"\n  多变量 log-rank p = {mlr_pval:.6f} "
          f"({'显著' if mlr_pval < 0.05 else '不显著'})")
    print(f"  显著配对: {significant_pairs}/{len(pairwise_results)}")

    if cms_pval is not None:
        print(f"\n  === CMS 生存差异（参考）===")
        print(f"    CMS 多变量 log-rank p = {cms_pval:.6f}")

    return mlr_pval


# ============================================================
# 6. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("CRC 三组学 IntegrAO 聚类 + 生存分析")
    print("=" * 60)

    ge, me, mi, survival = load_crc_data()
    data_all = [ge, me, mi]

    print("\n>>> 特征筛选 (top 2000)...")
    data_filtered = select_top_variable_features(data_all, top_n=2000)

    print("\n>>> 创建三组学 missing 数据 (overlap=0.8)...")
    datasets, union_to_orig, stats = split_three_omics_missing(
        data_filtered, ratio=0.8, seed=42)

    crc_dir = os.path.join(BASE_DIR, "CRC")
    n_clusters = 4

    print(f"\n{'='*60}")
    print(f">>> K={n_clusters} 聚类 + 生存分析")
    print("=" * 60)

    labels, embedding, union_samples = integrao_cluster(
        datasets, n_clusters=n_clusters)

    save_path = os.path.join(crc_dir, f"CRC_KM_K{n_clusters}.png")
    p_val = survival_analysis(labels, union_samples, union_to_orig, survival,
                              n_clusters, save_path)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  簇大小: {dict(zip(unique, counts))}")

    print(f"\n===== 完成! =====")
    print(f"输出: {save_path}")


if __name__ == "__main__":
    main()
