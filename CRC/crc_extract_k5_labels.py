"""
提取 IMGF 聚类标签，保存为 CSV 供 R 脚本使用。
用于 CRC 数据。K 值通过命令行参数指定。

用法: python crc_extract_labels.py K
示例: python crc_extract_labels.py 4
"""
import numpy as np
import pandas as pd
import sys, os, warnings
warnings.filterwarnings("ignore")

BASE_DIR = r"d:\integrAO\vs"
INTEGRAO_DIR = os.path.join(BASE_DIR, "IMGF")
sys.path.insert(0, INTEGRAO_DIR)
sys.path.insert(0, BASE_DIR)

from integrao.main import dist2, integrao_fuse
from integrao.util import data_indexing
import snf
from sklearn.cluster import KMeans


def load_crc_data():
    crc_dir = os.path.join(BASE_DIR, "CRC")
    ge = pd.read_csv(os.path.join(crc_dir, "crc_mRNA.csv"))
    me = pd.read_csv(os.path.join(crc_dir, "crc_methyl.csv"))
    mi = pd.read_csv(os.path.join(crc_dir, "crc_miRNA.csv"))
    survival = pd.read_csv(os.path.join(crc_dir, "crc_survival.csv"))
    print(f"mRNA: {ge.shape}, 甲基化: {me.shape}, miRNA: {mi.shape}")
    return ge, me, mi, survival


def select_top_variable_features(data_list, top_n=2000):
    selected = []
    for df in data_list:
        variances = df.var(axis=0)
        top_idx = variances.sort_values(ascending=False).head(top_n).index
        selected.append(df[top_idx])
        print(f"  特征筛选: {df.shape} -> {selected[-1].shape}")
    return selected


def split_three_omics_missing(data_list, ratio=0.8, seed=42):
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
          f"union={stats['n_union']}")
    return datasets, union_to_orig, stats


def imgf_cluster(datasets, n_clusters, neighbor_size=20, fusing_iteration=20):
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

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(embedding)

    return labels, embedding, union_samples


def main():
    # 从命令行获取 K 值
    if len(sys.argv) > 1:
        K = int(sys.argv[1])
    else:
        K = 5  # 默认值
    print(f"K = {K}")

    print("=" * 60)
    print(f"提取 IMGF K={K} 聚类标签 — CRC")
    print("=" * 60)

    ge, me, mi, survival = load_crc_data()
    data_all = [ge, me, mi]

    print("\n>>> 特征筛选 (top 2000)...")
    data_filtered = select_top_variable_features(data_all, top_n=2000)

    print("\n>>> 创建三组学 missing 数据 (overlap=0.8)...")
    datasets, union_to_orig, stats = split_three_omics_missing(
        data_filtered, ratio=0.8, seed=42)

    print(f"\n>>> IMGF 聚类 K={K}...")
    labels, embedding, union_samples = imgf_cluster(
        datasets, n_clusters=K)

    # 构建 原始索引 -> 聚类标签 映射
    orig_to_cluster = {}
    for i, name in enumerate(union_samples):
        orig_idx = union_to_orig[name]
        orig_to_cluster[orig_idx] = int(labels[i])

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  簇大小: {dict(zip(unique.astype(int), counts))}")

    # 为 mRNA 视图中的样本生成标签
    ge_view = datasets[0]
    ge_samples = ge_view.index.tolist()
    ge_orig_indices = [int(s.split("_")[1]) for s in ge_samples]
    ge_clusters = [orig_to_cluster.get(i, -1) for i in ge_orig_indices]

    crc_dir = os.path.join(BASE_DIR, "CRC")

    # 保存：原始索引 + 聚类标签（所有union样本）
    df_all = pd.DataFrame({
        "orig_index": list(orig_to_cluster.keys()),
        "cluster": list(orig_to_cluster.values())
    })
    df_all = df_all.sort_values("orig_index")
    out_csv = os.path.join(crc_dir, f"crc_k{K}_labels.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\n  标签已保存: {out_csv}")
    print(f"  总计 {len(df_all)} 个样本有聚类标签")

    # 同时保存 mRNA 视图子集的标签
    df_ge = pd.DataFrame({
        "orig_index": ge_orig_indices,
        "cluster": ge_clusters
    })
    df_ge = df_ge.sort_values("orig_index")
    df_ge = df_ge[df_ge["cluster"] >= 0]
    out_ge_csv = os.path.join(crc_dir, f"crc_k{K}_labels_mRNA.csv")
    df_ge.to_csv(out_ge_csv, index=False)
    print(f"  mRNA视图标签已保存: {out_ge_csv}")
    print(f"  mRNA视图样本数: {len(df_ge)}")

    # 打印颜色参考
    import matplotlib.pyplot as plt
    colors = plt.cm.tab10(np.linspace(0, 1, K))
    print(f"\n>>> 生存曲线颜色 (tab10, K={K}):")
    for i in range(K):
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
        print(f"  Cluster {i+1}: {hex_color}")

    print("\n===== 完成! =====")


if __name__ == "__main__":
    main()
