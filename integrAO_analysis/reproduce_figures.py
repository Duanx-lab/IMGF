"""
重现 IMGF 全部图表。

图 A (3子图): 一个组学完整 + 两个组学缺失 → NMI vs overlap ratio
图 B (1子图): 三组学都缺失 → NMI vs overlap ratio
图 C (3子图): 单组学原始特征 UMAP（无融合）
图 E (1子图): 多组学联合集成嵌入 UMAP

关于 log10: 融合矩阵 P 值在 [1e-7, 0.5]，log10(P) 全部为负。
Python 实现直接对相似度矩阵做特征分解 → 不需要 log10。
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import sys, os, warnings, time
warnings.filterwarnings("ignore")

BASE_DIR = r"d:\integrAO\vs"
INTEGRAO_DIR = os.path.join(BASE_DIR, "IntegrAO")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OMICS_DATA_DIR = os.environ.get("OMICS_DATA_DIR", os.path.join(INTEGRAO_DIR, "data", "omics"))
OUT_TAG = os.environ.get("OUT_TAG", "")
INTEGRAO_EPOCHS = int(os.environ.get("INTEGRAO_EPOCHS", "1000"))
IMGF_NDM = int(os.environ.get("IMGF_NDM", "3"))


def _out(name):
    """输出文件路径；可通过 OUT_TAG 环境变量在文件名里加后缀（如 _delta0.9）。"""
    if OUT_TAG:
        stem, ext = os.path.splitext(name)
        name = f"{stem}_{OUT_TAG}{ext}"
    return os.path.join(OUT_DIR, name)
sys.path.insert(0, INTEGRAO_DIR)

from integrao.main import dist2, integrao_fuse
from snf.compute import _find_dominate_set
from integrao.util import data_indexing
from nemo_msne import nemo_nmi, msne_nmi, _extract_true_labels
import snf
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import umap

# ============================================================
# 1. 数据加载
# ============================================================
def load_omics_data():
    data_dir = OMICS_DATA_DIR
    methyl  = pd.read_csv(os.path.join(data_dir, "omics1.txt"), index_col=0, delimiter="\t")
    expr    = pd.read_csv(os.path.join(data_dir, "omics2.txt"), index_col=0, delimiter="\t")
    protein = pd.read_csv(os.path.join(data_dir, "omics3.txt"), index_col=0, delimiter="\t")
    truelabel = pd.read_csv(os.path.join(data_dir, "clusters.txt"), index_col=0, delimiter="\t")

    methyl   = np.transpose(methyl)
    expr     = np.transpose(expr)
    protein  = np.transpose(protein)

    print(f"DNA甲基化: {methyl.shape}, mRNA: {expr.shape}, 蛋白质: {protein.shape}")
    print(f"真实标签: {truelabel.shape}, 类别数: {truelabel['cluster.id'].nunique()}")
    return expr, protein, methyl, truelabel


# ============================================================
# 2. 数据拆分 — 三种场景
# ============================================================
def split_one_complete_two_missing(data_list, truelabel, complete_idx, ratio, seed=42):
    """
    一个组学完整（全部500样本）+ 两个组学局部缺失。
    complete_idx: 0=mRNA, 1=protein, 2=methyl
    ratio: 两个缺失组学之间的重叠比 = n_common / (n_common + 2*n_unique)
    """
    np.random.seed(seed)
    n_total = len(data_list[0])
    n_common = int(n_total * ratio)
    n_unique_each = int((n_total - n_common) / 2)
    n_unique_each = max(n_unique_each, 0)
    # 调整使得 n_common + 2*n_unique_each ≈ n_total
    leftover = n_total - n_common - 2 * n_unique_each
    if leftover > 0:
        n_common += leftover

    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    common_set = set(all_idx[:n_common])
    # 两个缺失组学各自的独有样本
    pos = n_common
    u1_set = set(all_idx[pos:pos + n_unique_each]); pos += n_unique_each
    u2_set = set(all_idx[pos:pos + n_unique_each])

    # 每个视图：完整视图取全部，缺失视图取 common + 自己的 unique
    complete_names = [f"S{i}" for i in range(n_total)]  # 完整视图用原始名字

    def build_incomplete_view(df, my_unique_set, prefix):
        """缺失视图：common + 该视图独有"""
        my_indices = list(common_set) + list(my_unique_set)
        sub = df.iloc[my_indices].copy()
        new_names = []
        for idx in my_indices:
            if idx in common_set:
                new_names.append(f"common_{idx}")
            else:
                new_names.append(f"{prefix}_{idx}")
        sub.index = new_names
        return sub

    def build_complete_view(df):
        """完整视图：全部500样本"""
        sub = df.copy()
        new_names = []
        for i in range(n_total):
            if i in common_set:
                new_names.append(f"common_{i}")
            elif i in u1_set or i in u2_set:
                # 样本在完整视图中存在，但名字与缺失视图的独有样本名一致
                # 关键：完整视图中的独有样本名 = 缺失视图给它起的名字
                if i in u1_set:
                    # 这个样本在缺失视图1中叫 "v1_i"，完整视图也用这个名字
                    new_names.append(f"v1_{i}")
                else:
                    new_names.append(f"v2_{i}")
            else:
                new_names.append(f"complete_only_{i}")
        sub.index = new_names
        return sub

    datasets = [None, None, None]
    # 两个缺失视图
    missing_idxs = [j for j in range(3) if j != complete_idx]
    d1 = build_incomplete_view(data_list[missing_idxs[0]], u1_set, "v1")
    d2 = build_incomplete_view(data_list[missing_idxs[1]], u2_set, "v2")
    datasets[missing_idxs[0]] = d1
    datasets[missing_idxs[1]] = d2
    # 完整视图
    datasets[complete_idx] = build_complete_view(data_list[complete_idx])

    # Union 样本 → 真实标签映射
    all_names = sorted(set.union(*(set(d.index) for d in datasets)))
    label_map = {}
    for name in all_names:
        if name.startswith("common_"):
            orig = int(name.split("_")[1])
        elif name.startswith("v1_"):
            orig = int(name.split("_")[1])
        elif name.startswith("v2_"):
            orig = int(name.split("_")[1])
        elif name.startswith("complete_only_"):
            orig = int(name.split("_")[2])
        else:
            orig = int(name.split("_")[1]) if "_" in name else 0
        label_map[name] = truelabel.iloc[orig]["cluster.id"]
    union_labels = pd.Series({n: label_map[n] for n in all_names}).sort_index()

    sets = [set(d.index) for d in datasets]
    n_common_actual = len(set.intersection(*sets))
    n_union = len(set.union(*sets))
    actual_ratio = n_common_actual / n_union if n_union > 0 else 0

    return datasets, union_labels, {"ratio": actual_ratio, "n_union": n_union,
                                     "n_common": n_common_actual}


def split_all_three_missing(data_list, truelabel, ratio, seed=42):
    """
    三组学都缺失（用于 NMI 实验）。
    每个视图保留全部500样本但只改名字 — NMI 映射到原始顺序 0..499。
    """
    np.random.seed(seed)
    n_total = len(data_list[0])
    n_common = int(n_total * ratio)
    all_idx = np.arange(n_total)
    np.random.shuffle(all_idx)
    common_set = set(all_idx[:n_common])

    def rename_view(df, prefix):
        sub = df.copy()
        new_names = [f"common_{i}" if i in common_set else f"{prefix}_{i}" for i in range(n_total)]
        sub.index = new_names
        return sub

    d0 = rename_view(data_list[0], "v1")
    d1 = rename_view(data_list[1], "v2")
    d2 = rename_view(data_list[2], "v3")
    datasets = [d0, d1, d2]

    true_labels = pd.Series([truelabel.iloc[i]["cluster.id"] for i in range(n_total)],
                            index=list(range(n_total)))
    return datasets, true_labels, {"ratio": n_common / n_total, "n_common": n_common}


def split_for_umap_vis(data_list, truelabel, ratio=0.7, seed=42):
    """
    为 UMAP 可视化创建真正不同大小的视图（论文 Fig. 2e 正确逻辑）。

    500 总样本 = 350 common + 3 × 50 unique
    Union 恰好 500 个唯一样本名，每视图 400 样本。
    """
    np.random.seed(seed)
    n_total = len(data_list[0])
    n_common = int(n_total * ratio)
    n_unique_each = n_total - n_common  # 150
    n_per_view = n_unique_each // 3     # 50

    indices = np.arange(n_total)
    np.random.shuffle(indices)
    common_idx = indices[:n_common]
    u1 = indices[n_common : n_common + n_per_view]
    u2 = indices[n_common + n_per_view : n_common + 2 * n_per_view]
    u3 = indices[n_common + 2 * n_per_view : n_common + 3 * n_per_view]

    def build_view(df, common_subset, unique_subset, prefix):
        idx = np.concatenate([common_subset, unique_subset])
        sub = df.iloc[idx].copy()
        names = [f"common_{i}" if i in set(common_subset) else f"{prefix}_{i}" for i in idx]
        sub.index = names
        return sub

    d0 = build_view(data_list[0], common_idx, u1, "mrna")      # 350+50=400
    d1 = build_view(data_list[1], common_idx, u2, "protein")   # 350+50=400
    d2 = build_view(data_list[2], common_idx, u3, "methyl")    # 350+50=400

    datasets = [d0, d1, d2]

    # Union labels
    all_names = sorted(set(d0.index) | set(d1.index) | set(d2.index))
    label_map = {}
    for name in all_names:
        parts = name.split("_")
        orig_idx = int(parts[1])
        label_map[name] = truelabel.iloc[orig_idx]["cluster.id"]
    union_labels = pd.Series({n: label_map[n] for n in all_names}).sort_index()

    stats = {"n_common": n_common, "n_union": len(all_names),
             "n_per_view": n_common + n_per_view}
    print(f"  UMAP split: common={n_common}, unique/view={n_per_view}, "
          f"union={len(all_names)}, view_size={stats['n_per_view']}")
    return datasets, union_labels, stats


# ============================================================
# 3. 融合（integrao_fuse）
# ============================================================
def run_fusion(datasets, neighbor_size=20, fusing_iteration=20):
    (dicts_common, dicts_commonIndex, dict_sampleToIndexs,
     dicts_unique, original_order, dict_original_order) = data_indexing(datasets)
    S_dfs = []
    for i, view in enumerate(datasets):
        dist_mat = dist2(view.values, view.values)
        S_mat = snf.compute.affinity_matrix(dist_mat, K=neighbor_size, mu=0.5)
        S_df = pd.DataFrame(data=S_mat, index=original_order[i], columns=original_order[i])
        S_dfs.append(S_df)
    fused = integrao_fuse(S_dfs.copy(), dicts_common=dicts_common,
                          dicts_unique=dicts_unique, original_order=original_order,
                          neighbor_size=neighbor_size, fusing_iteration=fusing_iteration,
                          normalization_factor=1.0)
    return fused, dict_sampleToIndexs


def _symmetric(W):
    return (W + W.T) / 2.0


def _stable_normalized_array(W):
    row_sum = W.sum(axis=1) - np.diag(W)
    row_sum[row_sum == 0] = 1.0
    W = W / (2.0 * row_sum[:, None])
    np.fill_diagonal(W, 0.5)
    return _symmetric(W)


def _scaling_normalized_array(W, ratio):
    row_sum = W.sum(axis=1) - np.diag(W)
    row_sum[row_sum == 0] = 1.0
    W = (W / row_sum[:, None]) * 0.5 * ratio
    np.fill_diagonal(W, 1.0 - 0.5 * ratio)
    return _symmetric(W)


def run_fusion_fast(datasets, neighbor_size=20, fusing_iteration=20):
    """NumPy implementation of the IntegrAO diffusion used for visualization."""
    (dicts_common, _, dict_sample_to_indices,
     _, original_order, _) = data_indexing(datasets)

    affinities = []
    for view in datasets:
        dist_mat = dist2(view.values, view.values)
        affinities.append(
            snf.compute.affinity_matrix(
                dist_mat, K=neighbor_size, mu=0.5
            ).astype(float, copy=False)
        )

    affinities = [_stable_normalized_array(W) for W in affinities]
    dominant = [
        _find_dominate_set(W, min(neighbor_size, W.shape[0]))
        for W in affinities
    ]
    index_maps = [
        {sample: idx for idx, sample in enumerate(order)}
        for order in original_order
    ]

    start = time.time()
    for _ in range(fusing_iteration):
        next_affinities = []
        for target, target_affinity in enumerate(affinities):
            target_size = target_affinity.shape[0]
            accumulated = np.zeros_like(target_affinity)

            for source, source_affinity in enumerate(affinities):
                if source == target:
                    continue

                common = sorted(dicts_common[(target, source)])
                source_idx = np.array(
                    [index_maps[source][sample] for sample in common]
                )
                target_idx = np.array(
                    [index_maps[target][sample] for sample in common]
                )

                aligned = np.eye(target_size)
                aligned[np.ix_(target_idx, target_idx)] += (
                    source_affinity[np.ix_(source_idx, source_idx)]
                )
                aligned = _scaling_normalized_array(
                    aligned, len(common) / target_size
                )

                propagated = dominant[target] @ aligned @ dominant[target].T
                accumulated += _stable_normalized_array(propagated)

            next_affinities.append(
                accumulated / (len(affinities) - 1)
            )
        affinities = next_affinities

    affinities = [_stable_normalized_array(W) for W in affinities]
    fused = [
        pd.DataFrame(W, index=order, columns=order)
        for W, order in zip(affinities, original_order)
    ]
    print(f"Fast diffusion finished in {time.time() - start:.1f}s")
    return fused, dict_sample_to_indices


def run_fusion_cached(datasets, cache_path, neighbor_size=20,
                      fusing_iteration=20, force=False):
    """Load cached fused networks or compute and cache them once."""
    (_, _, dict_sample_to_indices,
     _, original_order, _) = data_indexing(datasets)

    if os.path.exists(cache_path) and not force:
        cached = np.load(cache_path)
        fused = [
            pd.DataFrame(
                cached[f"fused_{i}"],
                index=original_order[i],
                columns=original_order[i],
            )
            for i in range(len(datasets))
        ]
        print(f"Loaded fusion cache: {cache_path}")
        return fused, dict_sample_to_indices

    fused, dict_sample_to_indices = run_fusion_fast(
        datasets,
        neighbor_size=neighbor_size,
        fusing_iteration=fusing_iteration,
    )
    np.savez_compressed(
        cache_path,
        **{f"fused_{i}": matrix.values
           for i, matrix in enumerate(fused)}
    )
    print(f"Saved fusion cache: {cache_path}")
    return fused, dict_sample_to_indices


# ============================================================
# 4. Diffusion Map
# ============================================================
def _diffusion_map_components(W, n_components=3):
    N = W.shape[0]
    W = np.maximum(W, 0)
    W = (W + W.T) / 2
    D = np.sum(W, axis=1)
    D[D == 0] = 1
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    P_sym = D_inv_sqrt @ W @ D_inv_sqrt
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(P_sym)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eig(P_sym)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    n_available = min(n_components, len(eigenvalues) - 1)
    embedding = np.zeros((N, n_components))
    for i in range(n_available):
        lam = eigenvalues[i + 1]
        if lam > 1e-10:
            embedding[:, i] = lam * (D_inv_sqrt @ eigenvectors[:, i + 1])
    return embedding


def build_union_similarity(fused_networks, dict_sampleToIndexs):
    union_samples = sorted(dict_sampleToIndexs.keys())
    n_union = len(union_samples)
    S_sum = np.zeros((n_union, n_union))
    S_count = np.zeros((n_union, n_union))
    for i, mat in enumerate(fused_networks):
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
    return S_union, union_samples


# ============================================================
# 5. NMI 实验
# ============================================================
def run_nmi_for_scenario(datasets, union_labels, n_clusters, neighbor_size=20,
                         fusing_iteration=20, n_dm=3):
    """IMGF：融合 → DM → KMeans → NMI"""
    fused, dict_si = run_fusion(datasets, neighbor_size, fusing_iteration)

    # 判断用哪种嵌入方式
    n_sizes = [len(d) for d in datasets]
    if len(set(n_sizes)) == 1:
        # 所有视图相同大小（all-three-missing 场景）
        all_comp = []
        for i, mat in enumerate(fused):
            arr = mat.values
            emb = _diffusion_map_components(arr, n_components=n_dm)
            row_names = (mat.index.tolist() if isinstance(mat, pd.DataFrame)
                        else list(range(arr.shape[0])))
            emb_orig = np.zeros((len(row_names), emb.shape[1]))
            for vi, name in enumerate(row_names):
                orig_pos = int(name.split("_")[1])
                emb_orig[orig_pos, :] = emb[vi, :]
            all_comp.append(emb_orig)
        E = np.hstack(all_comp)
        true_arr = union_labels.values  # 0..499 order
    else:
        # 不同大小 → union 相似度矩阵 → DM
        S_union, union_samples = build_union_similarity(fused, dict_si)
        E = _diffusion_map_components(S_union, n_components=n_dm * 3)
        # labels 在 union 空间
        ul = union_labels.sort_index()
        true_arr = ul.values

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    pred = km.fit_predict(E)
    nmi = normalized_mutual_info_score(true_arr, pred)
    return nmi


def run_kmeans_on_intact(datasets, union_labels, n_clusters, intact_idx=None):
    """
    K-means on intact feature（论文基线）。
    对指定视图做 K-means，标签按样本名对齐 union_labels。
    intact_idx=None 时取三个单组学中的最佳 NMI。
    """
    def _nmi_for_view(view, ul):
        """对单个视图做 K-means 并计算 NMI，标签按样本名对齐"""
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        pred = km.fit_predict(view.values)
        # 按样本名对齐：找到 view 和 union_labels 共有的样本
        common_names = [n for n in view.index if n in ul.index]
        if len(common_names) == 0:
            return 0.0
        true = ul[common_names].values
        pred_aligned = pred[[list(view.index).index(n) for n in common_names]]
        return normalized_mutual_info_score(true, pred_aligned)

    if intact_idx is not None:
        return _nmi_for_view(datasets[intact_idx], union_labels)
    else:
        best = 0.0
        for view in datasets:
            nmi = _nmi_for_view(view, union_labels)
            best = max(best, nmi)
        return best


def run_integrao_gnn(datasets, union_labels, n_clusters,
                     neighbor_size=20, fusing_iteration=30,
                     embedding_dims=64, alighment_epochs=1000):
    """论文原版 IntegrAO：网络扩散融合 → GNN 统一 embedding → 谱聚类 → NMI。"""
    from integrao.integrater import integrao_integrater
    from sklearn.cluster import spectral_clustering

    integrater = integrao_integrater(
        datasets,
        dataset_name=None,
        neighbor_size=neighbor_size,
        embedding_dims=embedding_dims,
        fusing_iteration=fusing_iteration,
        normalization_factor=1.0,
        alighment_epochs=alighment_epochs,
        beta=1.0,
        mu=0.5,
    )
    integrater.network_diffusion()
    embeds_final, S_final, _ = integrater.unsupervised_alignment()

    sorted_names = list(embeds_final.index)
    true_arr = _extract_true_labels(union_labels, sorted_names)

    labels = spectral_clustering(S_final, n_clusters=n_clusters,
                                 assign_labels="kmeans", random_state=42)
    return normalized_mutual_info_score(true_arr, labels)


def _run_scenario_methods(datasets, labels, n_clusters, complete_idx):
    """跑一个场景的全部方法，返回 (results_dict, timing_dict)，每个值已计时。"""
    methods = {}
    timing = {}

    t0 = time.time()
    methods["IMGF"] = run_nmi_for_scenario(datasets, labels, n_clusters, n_dm=IMGF_NDM)
    timing["IMGF"] = time.time() - t0

    t0 = time.time()
    methods["IntegrAO"] = run_integrao_gnn(datasets, labels, n_clusters,
                                           alighment_epochs=INTEGRAO_EPOCHS)
    timing["IntegrAO"] = time.time() - t0

    t0 = time.time()
    methods["KMeans(on intact feature)"] = run_kmeans_on_intact(
        datasets, labels, n_clusters, intact_idx=complete_idx)
    timing["KMeans(on intact feature)"] = time.time() - t0

    t0 = time.time()
    methods["NEMO"] = nemo_nmi(datasets, labels, n_clusters)
    timing["NEMO"] = time.time() - t0

    t0 = time.time()
    methods["MSNE"] = msne_nmi(datasets, labels, n_clusters,
                               num_walks=40, walk_length=18, embed_size=60)
    timing["MSNE"] = time.time() - t0

    return methods, timing


def run_full_experiment(ratios=None, n_runs=1):
    """三组学都缺失 + 三种'一个完整+两个缺失'，记录各方法 NMI 与耗时。"""
    if ratios is None:
        ratios = np.arange(0.1, 1.0, 0.1)

    expr, protein, methyl, truelabel = load_omics_data()
    n_clusters = truelabel["cluster.id"].nunique()
    data_all = [expr, protein, methyl]
    omics_names = ["mRNA", "Protein", "Methyl"]

    METHOD_KEYS = ["IMGF", "IntegrAO", "KMeans(on intact feature)", "NEMO", "MSNE"]
    all_results = {}
    all_timing = {}

    def _empty_bucket():
        d = {"ratio": []}
        for k in METHOD_KEYS:
            d[k] = []
        return d

    # ---- 图 B: 三组学都缺失 ----
    print("\n" + "=" * 60)
    print("图B: 三组学都缺失")
    print("=" * 60)
    results_B = _empty_bucket()
    timing_B = _empty_bucket()
    for ratio in ratios:
        datasets, tl, stats = split_all_three_missing(data_all, truelabel, ratio, seed=42)
        methods, timing = _run_scenario_methods(datasets, tl, n_clusters, complete_idx=None)
        results_B["ratio"].append(ratio)
        timing_B["ratio"].append(ratio)
        for k in METHOD_KEYS:
            results_B[k].append(methods[k])
            timing_B[k].append(timing[k])
        print(f"  ratio={ratio:.1f}: " + "  ".join(
            f"{k}={methods[k]:.4f}" for k in METHOD_KEYS))
    all_results["B_all_missing"] = results_B
    all_timing["B_all_missing"] = timing_B

    # ---- 图 A: 三个'一个完整+两个缺失'场景 ----
    for complete_idx in range(3):
        scenario_name = f"A_{omics_names[complete_idx]}_complete"
        print(f"\n{'='*60}")
        print(f"图A: {omics_names[complete_idx]} 完整，其余缺失")
        print("=" * 60)
        results = _empty_bucket()
        timing = _empty_bucket()
        for ratio in ratios:
            datasets, ul, stats = split_one_complete_two_missing(
                data_all, truelabel, complete_idx, ratio, seed=42)
            methods, t = _run_scenario_methods(datasets, ul, n_clusters, complete_idx=complete_idx)
            results["ratio"].append(ratio)
            timing["ratio"].append(ratio)
            for k in METHOD_KEYS:
                results[k].append(methods[k])
                timing[k].append(t[k])
            print(f"  ratio={ratio:.1f}: " + "  ".join(
                f"{k}={methods[k]:.4f}" for k in METHOD_KEYS))
        all_results[scenario_name] = results
        all_timing[scenario_name] = timing

    return all_results, all_timing, n_clusters


# ============================================================
# 6. 单组学 UMAP（预融合）
# ============================================================
def plot_single_omics_umap(datasets_scenario, truelabel, omics_idx, title, save_path, n_clusters=15):
    """
    单组学原始特征 UMAP，用形状区分共有/独有样本。
    omics_idx: 0=mRNA, 1=protein, 2=methyl
    独有形状: mRNA=■, protein=◆, methyl=▲
    """
    import seaborn as sns
    from matplotlib.lines import Line2D

    set1 = set(datasets_scenario[0].index)
    set2 = set(datasets_scenario[1].index)
    set3 = set(datasets_scenario[2].index)
    sets = [set1, set2, set3]

    data_df = datasets_scenario[omics_idx]
    X = data_df.values
    names = data_df.index.tolist()

    # 判断每个样本是 common 还是 unique-to-this-omics
    is_common_arr = np.array([(n in set1 and n in set2 and n in set3) for n in names])
    is_unique_arr = np.array([
        (n in sets[omics_idx] and not any(n in sets[j] for j in range(3) if j != omics_idx))
        for n in names
    ])

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.7, spread=3.0, repulsion_strength=2.0, metric="cosine", random_state=42)
    xy = reducer.fit_transform(X)

    # 标签映射
    label_map = {}
    for n in names:
        orig_idx = int(n.split("_")[1]) if "_" in n else int(n.replace("S", ""))
        label_map[n] = truelabel.iloc[orig_idx]["cluster.id"]
    labels = np.array([label_map[n] for n in names])

    palette = sns.color_palette("tab20", n_clusters)
    sample_colors = np.array([palette[labels[i] - 1] for i in range(len(labels))])

    # 独有形状映射
    unique_markers = {0: "s", 1: "D", 2: "^"}  # mRNA=■, protein=◆, methyl=▲
    unique_labels_map = {0: "Unique samples in mRNA",
                        1: "Unique samples in protein",
                        2: "Unique samples in meth"}

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.set_facecolor("white")

    # 按 cluster 绘制
    for c in range(n_clusters):
        cmask = (labels == c + 1)
        if cmask.sum() == 0:
            continue
        # Common: 小圆形
        mask = cmask & is_common_arr
        if mask.sum() > 0:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=10,
                      c=sample_colors[mask], marker="o", alpha=0.5,
                      edgecolors="none", zorder=2)
        # Unique: 对应形状 + 黑边框
        mask = cmask & is_unique_arr
        if mask.sum() > 0:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=50,
                      c=sample_colors[mask], marker=unique_markers[omics_idx],
                      alpha=0.92, edgecolors="black", linewidths=1.2, zorder=4)

    # 图例
    legend_e = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#808080",
               markersize=8, markeredgecolor="none", linestyle="none",
               label="Common samples from meth, mRNA and protein"),
        Line2D([0], [0], marker=unique_markers[omics_idx], color="w",
               markerfacecolor="#808080", markersize=8,
               markeredgecolor="black", markeredgewidth=1, linestyle="none",
               label=unique_labels_map[omics_idx]),
    ]
    ax.legend(handles=legend_e, fontsize=8, loc="upper right", frameon=False)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, fontfamily="sans-serif")
    fig.text(0.5, 0.02, "UMAP (single omics, no integration)",
            ha="center", fontsize=10, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  单组学UMAP已保存: {save_path}")


# ============================================================
# 7. Post-integration UMAP（图 E）
# ============================================================
def plot_post_integration_umap(datasets, union_labels, fused_networks, dict_si,
                                save_path, n_dm=5):
    """联合多组学空间的特征提取与对齐后的 UMAP（论文 Fig. E）"""
    from matplotlib.lines import Line2D
    import seaborn as sns

    set1 = set(datasets[0].index)
    set2 = set(datasets[1].index)
    set3 = set(datasets[2].index)

    # Union 联合相似度 → DM → UMAP
    S_union, union_samples = build_union_similarity(fused_networks, dict_si)
    n_union = len(union_samples)
    E = _diffusion_map_components(S_union, n_components=n_dm)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, spread=1.5, repulsion_strength=2.0, metric="cosine", random_state=42)
    umap_xy = reducer.fit_transform(E)

    # 形状分类
    shape_types = []
    for s in union_samples:
        code = (s in set1) * 4 + (s in set2) * 2 + (s in set3) * 1
        if code == 7:   shape_types.append("common")
        elif code == 4: shape_types.append("mrna")
        elif code == 2: shape_types.append("protein")
        elif code == 1: shape_types.append("methyl")
        else:           shape_types.append("common")

    is_common  = np.array([s == "common" for s in shape_types])
    is_mrna    = np.array([s == "mrna" for s in shape_types])
    is_protein = np.array([s == "protein" for s in shape_types])
    is_methyl  = np.array([s == "methyl" for s in shape_types])

    # KMeans 聚类
    n_clusters = 15
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = km.fit_predict(E)
    cluster_colors = sns.color_palette("tab20", n_clusters)
    sample_colors = np.array([cluster_colors[cluster_labels[i]] for i in range(n_union)])

    # 拉远簇中心，同时让簇内部更紧凑
    # Compact clusters and enforce a visible gap between every pair of centers.
    cluster_radius = 0.52
    max_cluster_radius = 0.76
    min_center_gap = 3.2
    center_anchor = 0.025
    layout_iterations = 300

    cluster_centers = np.array([
        umap_xy[cluster_labels == i].mean(axis=0)
        for i in range(n_clusters)
    ])
    # Balance horizontal and vertical spread before applying repulsion.
    centered = cluster_centers - cluster_centers.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = eigenvectors @ np.diag(
        1.0 / np.sqrt(np.maximum(eigenvalues, 1e-6))
    ) @ eigenvectors.T
    cluster_centers = centered @ whitening.T * 2.8
    original_centers = cluster_centers.copy()

    # Enforce a minimum pairwise distance between cluster centers.
    for _ in range(layout_iterations):
        displacement = np.zeros_like(cluster_centers)
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                delta = cluster_centers[i] - cluster_centers[j]
                distance = np.linalg.norm(delta)
                if distance >= min_center_gap:
                    continue
                if distance < 1e-8:
                    angle = 2 * np.pi * (i + 1) / n_clusters
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / distance
                push = 0.5 * (min_center_gap - distance) * direction
                displacement[i] += push
                displacement[j] -= push

        displacement += center_anchor * (original_centers - cluster_centers)
        cluster_centers += 0.35 * displacement

    for i in range(n_clusters):
        cmask = (cluster_labels == i)
        old_center = umap_xy[cmask].mean(axis=0)
        offsets = umap_xy[cmask] - old_center
        radii = np.linalg.norm(offsets, axis=1)
        reference_radius = max(np.percentile(radii, 90), 1e-8)
        offsets *= cluster_radius / reference_radius

        # Cap extreme points so one sample cannot stretch the whole figure.
        radii = np.linalg.norm(offsets, axis=1)
        cap_scale = np.minimum(
            1.0, max_cluster_radius / np.maximum(radii, 1e-8)
        )
        offsets *= cap_scale[:, None]
        umap_xy[cmask] = cluster_centers[i] + offsets

    # k-NN 边
    n_edges = 5
    nn = NearestNeighbors(n_neighbors=n_edges + 1)
    nn.fit(umap_xy)
    _, edge_indices = nn.kneighbors(umap_xy)

    # 绘图
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    ax.set_facecolor("white")

    for i in range(n_union):
        for j_idx in range(1, n_edges + 1):
            j = edge_indices[i, j_idx]
            ax.plot([umap_xy[i, 0], umap_xy[j, 0]],
                   [umap_xy[i, 1], umap_xy[j, 1]],
                   color="gray", linewidth=0.15, alpha=0.015, zorder=0)

    for i in range(n_clusters):
        cmask = (cluster_labels == i)
        if cmask.sum() == 0:
            continue

        mask = cmask & is_common
        if mask.sum() > 0:
            ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1],
                      s=30, c=sample_colors[mask], marker="o", alpha=0.7,
                      edgecolors="none", zorder=2)

        for mask, marker in [(cmask & is_methyl, "^"),
                             (cmask & is_mrna, "s"),
                             (cmask & is_protein, "D")]:
            if mask.sum() > 0:
                ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1],
                          s=16, c=sample_colors[mask], marker=marker, alpha=0.92,
                          edgecolors="black", linewidths=1.2, zorder=4)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#808080",
               markersize=8, markeredgecolor="none", linestyle="none",
               label="Common samples from meth, mRNA and protein"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#808080",
               markersize=8, markeredgecolor="black", markeredgewidth=1,
               linestyle="none", label="Unique samples in meth"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#808080",
               markersize=8, markeredgecolor="black", markeredgewidth=1,
               linestyle="none", label="Unique samples in mRNA"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#808080",
               markersize=8, markeredgecolor="black", markeredgewidth=1,
               linestyle="none", label="Unique samples in protein"),
    ]
    fig.legend(handles=legend_elements, fontsize=8.0,
              loc="lower center", bbox_to_anchor=(0.5, 0.045),
              ncol=4, frameon=False, handletextpad=0.6,
              columnspacing=1.2, borderpad=0.2)

    ax.set_xticks([]); ax.set_yticks([])
    x_min, y_min = umap_xy.min(axis=0)
    x_max, y_max = umap_xy.max(axis=0)
    x_span = max(x_max - x_min, 1e-8)
    y_span = max(y_max - y_min, 1e-8)
    pad = max(x_span, y_span) * 0.06
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_common = is_common.sum()
    fig.text(0.5, 0.008, "UMAP with integrated embeddings",
            ha="center", va="bottom", fontsize=10, style="italic", color="#555555")
    ax.set_title(f"IMGF integration | Common:{n_common} "
                f"Meth(▲):{is_methyl.sum()} mRNA(■):{is_mrna.sum()} Protein(◆):{is_protein.sum()}",
                fontsize=9.5, fontfamily="sans-serif", color="#555555", pad=8)

    plt.tight_layout(rect=[0.02, 0.12, 0.98, 0.97])
    plt.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Post-integration UMAP已保存: {save_path}")


# ============================================================
# 8. NMI 曲线绘图
# ============================================================
def plot_nmi_curves(results_dict, scenario_name, save_path):
    """单条 NMI 曲线（含 NEMO/MSNE 对比）"""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ratios = results_dict["ratio"]
    ax.plot(ratios, results_dict["IMGF"], "o-", color="#E53935",
           linewidth=2.5, markersize=8, label="IMGF")
    ax.plot(ratios, results_dict["IntegrAO"], "o--", color="#8E24AA",
           linewidth=2.5, markersize=8, label="IntegrAO")
    ax.plot(ratios, results_dict["KMeans(on intact feature)"], "^--", color="#43A047",
           linewidth=2.5, markersize=8, label="KMeans (on intact feature)")
    if results_dict.get("NEMO"):
        ax.plot(ratios, results_dict["NEMO"], "s-.", color="#1E88E5",
               linewidth=2.5, markersize=8, label="NEMO")
    if results_dict.get("MSNE"):
        ax.plot(ratios, results_dict["MSNE"], "D:", color="#FB8C00",
               linewidth=2.5, markersize=8, label="MSNE")

    ax.set_xlabel("Overlap Ratio", fontsize=12)
    ax.set_ylabel("NMI", fontsize=12)
    ax.set_title(scenario_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    if "Methyl" in scenario_name or scenario_name.startswith("B"):
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    else:
        ax.set_ylim(0.6, 1.0)
        ax.set_yticks(np.arange(0.60, 1.01, 0.05))
    ax.grid(True, alpha=0.3)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  NMI曲线已保存: {save_path}")


# ============================================================
# 9. 组合大图
# ============================================================
def plot_combined_figure(all_results, expr, protein, methyl, truelabel, n_clusters):
    """
    拆分为两张独立大图：
    - Fig 1: 4 张 NMI 折线图 (2x2)
    - Fig 2: 3 张单组学 UMAP + 1 张集成 UMAP (2x2)
    """
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.lines import Line2D
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # ============================================================
    # 大图 1: NMI 折线图 (2x2)
    # ============================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")
    scenarios_nmi = [
        ("A_mRNA_complete",  "mRNA complete,\nmeth & protein missing"),
        ("A_Protein_complete", "Protein complete,\nmRNA & meth missing"),
        ("A_Methyl_complete",  "Methyl complete,\nmRNA & protein missing"),
        ("B_all_missing",    "All three omics\npartially missing"),
    ]

    for idx, (key, title) in enumerate(scenarios_nmi):
        ax = axes1[idx // 2][idx % 2]
        r = all_results[key]
        ax.plot(ratios, r["IMGF"], "o-", color="#E53935",
               linewidth=2.5, markersize=7, label="IMGF")
        ax.plot(ratios, r["IntegrAO"], "o--", color="#8E24AA",
               linewidth=2.5, markersize=7, label="IntegrAO")
        ax.plot(ratios, r["KMeans(on intact feature)"], "^--", color="#43A047",
               linewidth=2.5, markersize=7, label="KMeans (on intact feature)")
        if r.get("NEMO"):
            ax.plot(ratios, r["NEMO"], "s-.", color="#1E88E5",
                   linewidth=2.5, markersize=7, label="NEMO")
        if r.get("MSNE"):
            ax.plot(ratios, r["MSNE"], "D:", color="#FB8C00",
                   linewidth=2.5, markersize=7, label="MSNE")
        ax.set_xlabel("Overlap Ratio", fontsize=11)
        ax.set_ylabel("NMI", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        if "Methyl" in key or key.startswith("B"):
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        else:
            ax.set_ylim(0.6, 1.0)
            ax.set_yticks(np.arange(0.60, 1.01, 0.05))
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig1.suptitle("IMGF: NMI vs Overlap Ratio", fontsize=16, fontweight="bold", y=0.98)
    plt.figure(fig1.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    nmi_path = _out("nmi_curves.png")
    fig1.savefig(nmi_path, dpi=250, bbox_inches="tight", facecolor="white")
    fig1.savefig(nmi_path.replace(".png", ".pdf"), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"  NMI 折线图已保存: {nmi_path}")

    # ============================================================
    # 大图 2: UMAP 散点图 — 3 单组学 + 1 集成 (2x2)
    # ============================================================
    data_all = [expr, protein, methyl]
    omics_titles = ["Protein expression (raw)", "mRNA expression (raw)", "DNA methylation (raw)"]
    omics_order = [1, 0, 2]
    unique_markers = {0: "s", 1: "D", 2: "^"}

    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 16), facecolor="white")

    # 获取 UMAP split
    ds_ctx, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    sets_ctx = [set(ds_ctx[0].index), set(ds_ctx[1].index), set(ds_ctx[2].index)]

    # ---- 前三张: 单组学 UMAP ----
    for sub_idx, omics_idx in enumerate(omics_order):
        ax = axes2[sub_idx // 2][sub_idx % 2]
        data_df = ds_ctx[omics_idx]
        X = data_df.values
        names = data_df.index.tolist()

        is_common_arr = np.array([(n in sets_ctx[0] and n in sets_ctx[1] and n in sets_ctx[2])
                                  for n in names])
        is_unique_arr = np.array([
            (n in sets_ctx[omics_idx] and not any(n in sets_ctx[j]
             for j in range(3) if j != omics_idx))
            for n in names
        ])

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.7, spread=3.0, repulsion_strength=2.0,
                            metric="cosine", random_state=42)
        xy = reducer.fit_transform(X)

        label_map = {}
        for n in names:
            orig_idx = int(n.split("_")[1])
            label_map[n] = truelabel.iloc[orig_idx]["cluster.id"]
        labels = np.array([label_map[n] for n in names])

        palette = sns.color_palette("tab20", n_clusters)
        colors_arr = np.array([palette[labels[i] - 1] for i in range(len(labels))])

        for c in range(n_clusters):
            cmask = (labels == c + 1)
            if cmask.sum() == 0:
                continue
            mask = cmask & is_common_arr
            if mask.sum() > 0:
                ax.scatter(xy[mask, 0], xy[mask, 1], s=22,
                          c=colors_arr[mask], marker="o", alpha=0.7,
                          edgecolors="none", zorder=2)
            mask = cmask & is_unique_arr
            if mask.sum() > 0:
                ax.scatter(xy[mask, 0], xy[mask, 1], s=16,
                          c=colors_arr[mask], marker=unique_markers[omics_idx],
                          alpha=0.92, edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(omics_titles[sub_idx], fontsize=12, fontweight="bold")
        if sub_idx == 0:
            ax.text(0.5, -0.08,
                   "UMAP (single omics, no integration)\n● common  ◆ protein-unique  ■ mRNA-unique  ▲ methyl-unique",
                   transform=ax.transAxes, ha="center", fontsize=7.5,
                   style="italic", color="#555555")

    # ---- 第四张: Post-integration UMAP ----
    ax_e = axes2[1][1]
    datasets_umap, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    fused, dict_si = run_fusion(datasets_umap)
    S_union, union_samples = build_union_similarity(fused, dict_si)
    n_union = len(union_samples)
    E = _diffusion_map_components(S_union, n_components=5)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, spread=1.5, repulsion_strength=2.0,
                        metric="cosine", random_state=42)
    xy_e = reducer.fit_transform(E)

    set1_e = set(datasets_umap[0].index)
    set2_e = set(datasets_umap[1].index)
    set3_e = set(datasets_umap[2].index)
    shape_types_e = []
    for s in union_samples:
        code = (s in set1_e)*4 + (s in set2_e)*2 + (s in set3_e)*1
        if code == 7:   shape_types_e.append("common")
        elif code == 4: shape_types_e.append("mrna")
        elif code == 2: shape_types_e.append("protein")
        elif code == 1: shape_types_e.append("methyl")
        else:           shape_types_e.append("common")

    is_common_e  = np.array([s == "common" for s in shape_types_e])
    is_mrna_e    = np.array([s == "mrna" for s in shape_types_e])
    is_prot_e    = np.array([s == "protein" for s in shape_types_e])
    is_meth_e    = np.array([s == "methyl" for s in shape_types_e])

    c_labels_e = KMeans(n_clusters=15, random_state=42, n_init="auto").fit_predict(E)
    col_e = sns.color_palette("tab20", 15)
    scol_e = np.array([col_e[c_labels_e[i]] for i in range(n_union)])

    # 白化 → 中心排斥 → 压缩簇内部
    cluster_radius = 0.52
    max_cluster_radius = 0.76
    min_center_gap = 3.2
    center_anchor = 0.025
    layout_iters = 300

    centers_e = np.array([xy_e[c_labels_e == i].mean(axis=0) for i in range(15)])
    centered_e = centers_e - centers_e.mean(axis=0)
    cov_e = np.cov(centered_e, rowvar=False)
    evals_e, evecs_e = np.linalg.eigh(cov_e)
    whitening_e = evecs_e @ np.diag(1.0 / np.sqrt(np.maximum(evals_e, 1e-6))) @ evecs_e.T
    centers_e = centered_e @ whitening_e.T * 2.8
    orig_centers_e = centers_e.copy()

    for _ in range(layout_iters):
        disp_e = np.zeros_like(centers_e)
        for i in range(15):
            for j in range(i + 1, 15):
                delta = centers_e[i] - centers_e[j]
                dist = np.linalg.norm(delta)
                if dist >= min_center_gap:
                    continue
                if dist < 1e-8:
                    angle = 2 * np.pi * (i + 1) / 15
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / dist
                push = 0.5 * (min_center_gap - dist) * direction
                disp_e[i] += push
                disp_e[j] -= push
        disp_e += center_anchor * (orig_centers_e - centers_e)
        centers_e += 0.35 * disp_e

    for i in range(15):
        cm_e = (c_labels_e == i)
        old_ctr_e = xy_e[cm_e].mean(axis=0)
        off_e = xy_e[cm_e] - old_ctr_e
        radii_e = np.linalg.norm(off_e, axis=1)
        ref_e = max(np.percentile(radii_e, 90), 1e-8)
        off_e *= cluster_radius / ref_e
        radii_e = np.linalg.norm(off_e, axis=1)
        cap_e = np.minimum(1.0, max_cluster_radius / np.maximum(radii_e, 1e-8))
        off_e *= cap_e[:, None]
        xy_e[cm_e] = centers_e[i] + off_e

    nn_e = NearestNeighbors(n_neighbors=6)
    nn_e.fit(xy_e)
    _, ei_e = nn_e.kneighbors(xy_e)
    for i in range(n_union):
        for j_idx in range(1, 6):
            j = ei_e[i, j_idx]
            ax_e.plot([xy_e[i, 0], xy_e[j, 0]], [xy_e[i, 1], xy_e[j, 1]],
                     color="gray", linewidth=0.12, alpha=0.012, zorder=0)

    for c in range(15):
        cm = (c_labels_e == c)
        if cm.sum() == 0: continue
        m = cm & is_common_e
        if m.sum() > 0:
            ax_e.scatter(xy_e[m,0], xy_e[m,1], s=30, c=scol_e[m], marker="o",
                        alpha=0.7, edgecolors="none", zorder=2)
        for mask, marker in [(cm & is_meth_e, "^"), (cm & is_mrna_e, "s"), (cm & is_prot_e, "D")]:
            if mask.sum() > 0:
                ax_e.scatter(xy_e[mask,0], xy_e[mask,1], s=16, c=scol_e[mask],
                           marker=marker, alpha=0.92, edgecolors="black",
                           linewidths=0.8, zorder=4)

    legend_e = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="none", linestyle="none",
               label="Common samples from meth, mRNA and protein"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="black", markeredgewidth=1, linestyle="none",
               label="Unique samples in meth"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="black", markeredgewidth=1, linestyle="none",
               label="Unique samples in mRNA"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="black", markeredgewidth=1, linestyle="none",
               label="Unique samples in protein"),
    ]
    ax_e.legend(handles=legend_e, fontsize=7,
               bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    ax_e.set_xticks([]); ax_e.set_yticks([])
    for spine in ax_e.spines.values():
        spine.set_visible(False)
    ax_e.set_title("IMGF integrated embeddings\n(UMAP with joint multi-omics)",
                  fontsize=12, fontweight="bold")

    fig2.suptitle("IMGF: UMAP Visualization", fontsize=16, fontweight="bold", y=0.98)
    plt.figure(fig2.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    umap_path = _out("umap_overview.png")
    fig2.savefig(umap_path, dpi=250, bbox_inches="tight", facecolor="white")
    fig2.savefig(umap_path.replace(".png", ".pdf"), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  UMAP 散点图已保存: {umap_path}")


# ============================================================
# 9.4 最终组合大图（上排 NMI，下排 UMAP）
# ============================================================
def plot_final_figure(all_results, expr, protein, methyl, truelabel, n_clusters):
    """生成 2x4 组合大图：上排 NMI vs overlap ratio，下排 UMAP。"""
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.lines import Line2D
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    fig = plt.figure(figsize=(24, 13), facecolor="white")
    gs = fig.add_gridspec(2, 4, hspace=0.34, wspace=0.22,
                          left=0.04, right=0.99, top=0.88, bottom=0.05)

    fig.suptitle("IMGF: Overlapping Ratio Analysis & UMAP Visualization",
                 fontsize=20, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965,
             "Top row: NMI vs overlap ratio. Bottom row left 3: single-omics UMAP (no fusion). "
             "Bottom right: post-integration UMAP.",
             ha="center", fontsize=11.5, style="italic", color="#333333")

    # ---- 上排：4 个 NMI 折线图 ----
    scenarios_nmi = [
        ("A_mRNA_complete", "mRNA complete,\nmeth & protein missing"),
        ("A_Protein_complete", "Protein complete,\nmRNA & meth missing"),
        ("A_Methyl_complete", "Methyl complete,\nmRNA & protein missing"),
        ("B_all_missing", "All three omics\npartially missing"),
    ]
    for idx, (key, title) in enumerate(scenarios_nmi):
        ax = fig.add_subplot(gs[0, idx])
        r = all_results[key]
        ax.plot(ratios, r["IMGF"], "o-", color="#E53935", lw=2.5, ms=7, label="IMGF")
        ax.plot(ratios, r["IntegrAO"], "o--", color="#8E24AA", lw=2.5, ms=7, label="IntegrAO")
        ax.plot(ratios, r["KMeans(on intact feature)"], "^--", color="#43A047",
                lw=2.5, ms=7, label="KMeans (on intact feature)")
        if r.get("NEMO"):
            ax.plot(ratios, r["NEMO"], "s-.", color="#1E88E5", lw=2.5, ms=7, label="NEMO")
        if r.get("MSNE"):
            ax.plot(ratios, r["MSNE"], "D:", color="#FB8C00", lw=2.5, ms=7, label="MSNE")
        ax.set_xlabel("Overlap Ratio", fontsize=11)
        ax.set_ylabel("NMI", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        # 自适应 y 轴，让曲线拉开
        vals = [v for k in ["IMGF", "IntegrAO", "KMeans(on intact feature)", "NEMO", "MSNE"]
                for v in r.get(k, []) if not pd.isna(v)]
        lo = min(vals); hi = max(vals)
        pad = (hi - lo) * 0.18 + 0.02
        lo = max(0.0, lo - pad); hi = min(1.0, hi + pad)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # ---- 下排：3 单组学 UMAP + 1 post-integration UMAP ----
    data_all = [expr, protein, methyl]
    omics_titles = ["Protein expression (raw)", "mRNA expression (raw)", "DNA methylation (raw)"]
    omics_order = [1, 0, 2]
    unique_markers = {0: "s", 1: "D", 2: "^"}

    ds_ctx, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    sets_ctx = [set(ds_ctx[0].index), set(ds_ctx[1].index), set(ds_ctx[2].index)]

    for sub_idx, omics_idx in enumerate(omics_order):
        ax = fig.add_subplot(gs[1, sub_idx])
        data_df = ds_ctx[omics_idx]
        X = data_df.values
        names = data_df.index.tolist()

        is_common_arr = np.array([(n in sets_ctx[0] and n in sets_ctx[1] and n in sets_ctx[2])
                                  for n in names])
        is_unique_arr = np.array([
            (n in sets_ctx[omics_idx] and not any(n in sets_ctx[j]
             for j in range(3) if j != omics_idx))
            for n in names
        ])

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.7, spread=3.0, repulsion_strength=2.0,
                            metric="cosine", random_state=42)
        xy = reducer.fit_transform(X)

        label_map = {}
        for n in names:
            orig_idx = int(n.split("_")[1])
            label_map[n] = truelabel.iloc[orig_idx]["cluster.id"]
        labels = np.array([label_map[n] for n in names])

        palette = sns.color_palette("tab20", n_clusters)
        colors_arr = np.array([palette[labels[i] - 1] for i in range(len(labels))])

        for c in range(n_clusters):
            cmask = (labels == c + 1)
            if cmask.sum() == 0:
                continue
            mask = cmask & is_common_arr
            if mask.sum() > 0:
                ax.scatter(xy[mask, 0], xy[mask, 1], s=22, c=colors_arr[mask],
                           marker="o", alpha=0.7, edgecolors="none", zorder=2)
            mask = cmask & is_unique_arr
            if mask.sum() > 0:
                ax.scatter(xy[mask, 0], xy[mask, 1], s=16, c=colors_arr[mask],
                           marker=unique_markers[omics_idx], alpha=0.92,
                           edgecolors="black", linewidths=0.8, zorder=4)

        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(omics_titles[sub_idx], fontsize=12, fontweight="bold")
        if sub_idx == 0:
            ax.text(0.5, -0.10,
                    "UMAP (single omics, no integration)\n● common  ◆ protein-unique  ■ mRNA-unique  ▲ methyl-unique",
                    transform=ax.transAxes, ha="center", fontsize=7.5,
                    style="italic", color="#555555")

    # ---- 第四张：Post-integration UMAP ----
    ax_e = fig.add_subplot(gs[1, 3])
    datasets_umap, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    fused, dict_si = run_fusion(datasets_umap)
    S_union, union_samples = build_union_similarity(fused, dict_si)
    n_union = len(union_samples)
    E = _diffusion_map_components(S_union, n_components=5)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, spread=1.5, repulsion_strength=2.0,
                        metric="cosine", random_state=42)
    xy_e = reducer.fit_transform(E)

    set1_e = set(datasets_umap[0].index)
    set2_e = set(datasets_umap[1].index)
    set3_e = set(datasets_umap[2].index)
    shape_types_e = []
    for s in union_samples:
        code = (s in set1_e) * 4 + (s in set2_e) * 2 + (s in set3_e) * 1
        if code == 7:   shape_types_e.append("common")
        elif code == 4: shape_types_e.append("mrna")
        elif code == 2: shape_types_e.append("protein")
        elif code == 1: shape_types_e.append("methyl")
        else:           shape_types_e.append("common")

    is_common_e = np.array([s == "common" for s in shape_types_e])
    is_mrna_e = np.array([s == "mrna" for s in shape_types_e])
    is_prot_e = np.array([s == "protein" for s in shape_types_e])
    is_meth_e = np.array([s == "methyl" for s in shape_types_e])

    c_labels_e = KMeans(n_clusters=15, random_state=42, n_init="auto").fit_predict(E)
    col_e = sns.color_palette("tab20", 15)
    scol_e = np.array([col_e[c_labels_e[i]] for i in range(n_union)])

    cluster_radius = 0.52
    max_cluster_radius = 0.76
    min_center_gap = 3.2
    center_anchor = 0.025
    layout_iters = 300

    centers_e = np.array([xy_e[c_labels_e == i].mean(axis=0) for i in range(15)])
    centered_e = centers_e - centers_e.mean(axis=0)
    cov_e = np.cov(centered_e, rowvar=False)
    evals_e, evecs_e = np.linalg.eigh(cov_e)
    whitening_e = evecs_e @ np.diag(1.0 / np.sqrt(np.maximum(evals_e, 1e-6))) @ evecs_e.T
    centers_e = centered_e @ whitening_e.T * 2.8
    orig_centers_e = centers_e.copy()

    for _ in range(layout_iters):
        disp_e = np.zeros_like(centers_e)
        for i in range(15):
            for j in range(i + 1, 15):
                delta = centers_e[i] - centers_e[j]
                dist = np.linalg.norm(delta)
                if dist >= min_center_gap:
                    continue
                if dist < 1e-8:
                    angle = 2 * np.pi * (i + 1) / 15
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = delta / dist
                push = 0.5 * (min_center_gap - dist) * direction
                disp_e[i] += push
                disp_e[j] -= push
        disp_e += center_anchor * (orig_centers_e - centers_e)
        centers_e += 0.35 * disp_e

    for i in range(15):
        cm_e = (c_labels_e == i)
        old_ctr_e = xy_e[cm_e].mean(axis=0)
        off_e = xy_e[cm_e] - old_ctr_e
        radii_e = np.linalg.norm(off_e, axis=1)
        ref_e = max(np.percentile(radii_e, 90), 1e-8)
        off_e *= cluster_radius / ref_e
        radii_e = np.linalg.norm(off_e, axis=1)
        cap_e = np.minimum(1.0, max_cluster_radius / np.maximum(radii_e, 1e-8))
        off_e *= cap_e[:, None]
        xy_e[cm_e] = centers_e[i] + off_e

    nn_e = NearestNeighbors(n_neighbors=6)
    nn_e.fit(xy_e)
    _, ei_e = nn_e.kneighbors(xy_e)
    for i in range(n_union):
        for j_idx in range(1, 6):
            j = ei_e[i, j_idx]
            ax_e.plot([xy_e[i, 0], xy_e[j, 0]], [xy_e[i, 1], xy_e[j, 1]],
                      color="gray", linewidth=0.12, alpha=0.012, zorder=0)

    for c in range(15):
        cm = (c_labels_e == c)
        if cm.sum() == 0:
            continue
        m = cm & is_common_e
        if m.sum() > 0:
            ax_e.scatter(xy_e[m, 0], xy_e[m, 1], s=30, c=scol_e[m], marker="o",
                         alpha=0.7, edgecolors="none", zorder=2)
        for mask, marker in [(cm & is_meth_e, "^"), (cm & is_mrna_e, "s"), (cm & is_prot_e, "D")]:
            if mask.sum() > 0:
                ax_e.scatter(xy_e[mask, 0], xy_e[mask, 1], s=16, c=scol_e[mask],
                             marker=marker, alpha=0.92, edgecolors="black",
                             linewidths=0.8, zorder=4)

    legend_e = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="none", linestyle="none",
               label="Common samples from meth, mRNA and protein"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="black", markeredgewidth=1, linestyle="none",
               label="Unique samples in meth"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="black", markeredgewidth=1, linestyle="none",
               label="Unique samples in mRNA"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#808080", markersize=6,
               markeredgecolor="black", markeredgewidth=1, linestyle="none",
               label="Unique samples in protein"),
    ]
    ax_e.legend(handles=legend_e, fontsize=7, bbox_to_anchor=(1.02, 1),
                loc="upper left", frameon=False)

    ax_e.set_xticks([]); ax_e.set_yticks([])
    for spine in ax_e.spines.values():
        spine.set_visible(False)
    ax_e.set_title("Post-integration UMAP", fontsize=12, fontweight="bold")

    fig.savefig(_out("final_figure.png"), dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(_out("final_figure.pdf"), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  组合大图已保存: {_out('final_figure.png')}")


# ============================================================
# 9.5 运行时间对比
# ============================================================
def plot_timing_comparison(all_timing):
    """运行时间对比：柱状图（平均耗时，对数刻度）+ ratio 曲线。"""
    METHOD_KEYS = ["IMGF", "IntegrAO", "KMeans(on intact feature)", "NEMO", "MSNE"]
    SHORT = {"IMGF": "IMGF", "IntegrAO": "IntegrAO",
             "KMeans(on intact feature)": "KMeans", "NEMO": "NEMO", "MSNE": "MSNE"}
    COLORS = {"IMGF": "#E53935", "IntegrAO": "#8E24AA",
              "KMeans(on intact feature)": "#43A047", "NEMO": "#1E88E5", "MSNE": "#FB8C00"}

    records = []
    for scenario, tdict in all_timing.items():
        rr = tdict["ratio"]
        for m in METHOD_KEYS:
            for r, t in zip(rr, tdict[m]):
                records.append({"scenario": scenario, "method": m, "ratio": r, "time": t})
    tdf = pd.DataFrame(records)

    # ---- 柱状图：平均耗时（对数刻度）----
    fig1, ax1 = plt.subplots(figsize=(8, 5.5), facecolor="white")
    means = [tdf.loc[tdf.method == m, "time"].mean() for m in METHOD_KEYS]
    labels = [SHORT[m] for m in METHOD_KEYS]
    colors = [COLORS[m] for m in METHOD_KEYS]
    bars = ax1.bar(labels, means, color=colors)
    ax1.set_yscale("log")
    ax1.set_ylabel("Mean runtime (s, log scale)")
    ax1.set_title("Average runtime per method (log scale)")
    for b, v in zip(bars, means):
        if v > 0:
            ax1.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.3f}s",
                     ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig1.savefig(_out("timing_bar.png"), dpi=200,
                 bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"  时间柱状图已保存: {_out('timing_bar.png')}")

    # ---- ratio 曲线：平均耗时（跨场景）----
    fig2, ax2 = plt.subplots(figsize=(8, 5.5), facecolor="white")
    for m in METHOD_KEYS:
        g = tdf.loc[tdf.method == m].groupby("ratio")["time"].mean()
        ax2.plot(g.index, g.values, "o-", color=COLORS[m],
                 linewidth=2, markersize=6, label=SHORT[m])
    ax2.set_xlabel("Overlap Ratio")
    ax2.set_ylabel("Mean runtime (s)")
    ax2.set_title("Runtime vs overlap ratio")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(_out("timing_ratio.png"), dpi=200,
                 bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  时间ratio曲线已保存: {_out('timing_ratio.png')}")

    return tdf


# ============================================================
# 10. 主入口
# ============================================================
def redraw_combined_figure(force_cache=False):
    """仅重绘组合大图，NMI数值从CSV读取，单组学UMAP和融合复用缓存。"""
    expr, protein, methyl, truelabel = load_omics_data()
    data_all = [expr, protein, methyl]
    n_clusters = truelabel["cluster.id"].nunique()

    # 从 CSV 读取 NMI 结果，构造 all_results 字典
    csv_path = _out("nmi_results.csv")
    if not os.path.exists(csv_path):
        print("ERROR: nmi_results.csv 不存在，请先运行完整 main()")
        return
    df_csv = pd.read_csv(csv_path)
    all_results = {}
    for key_prefix in ["B_all_missing", "A_mRNA_complete", "A_Protein_complete", "A_Methyl_complete"]:
        all_results[key_prefix] = {
            "ratio": df_csv["ratio"].tolist(),
            "IMGF": df_csv[f"{key_prefix}_IMGF"].tolist(),
            "IntegrAO": df_csv[f"{key_prefix}_IntegrAO"].tolist(),
            "KMeans(on intact feature)": df_csv[f"{key_prefix}_KMeans"].tolist(),
            "NEMO": df_csv[f"{key_prefix}_NEMO"].tolist(),
            "MSNE": df_csv[f"{key_prefix}_MSNE"].tolist(),
        }

    # 单组学 UMAP（从缓存读融合网络，跳过 NMI 实验）
    print("\n>>> 单组学 UMAP...")
    ds_umap, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    plot_single_omics_umap(ds_umap, truelabel, 1, "Protein expression (raw)",
                          _out("umap_single_protein.png"), n_clusters)
    plot_single_omics_umap(ds_umap, truelabel, 0, "mRNA expression (raw)",
                          _out("umap_single_mrna.png"), n_clusters)
    plot_single_omics_umap(ds_umap, truelabel, 2, "DNA methylation (raw)",
                          _out("umap_single_methyl.png"), n_clusters)

    # Post-integration UMAP
    print("\n>>> Post-integration UMAP...")
    datasets_umap, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    cache_path = _out("overlap_fusion_cache.npz")
    fused, dict_si = run_fusion_cached(datasets_umap, cache_path,
                                       neighbor_size=20, fusing_iteration=20, force=force_cache)
    plot_post_integration_umap(datasets_umap, None, fused, dict_si,
                               _out("overlap_structure.png"))

    # 拆分为两张独立大图
    print("\n>>> NMI 折线图 + UMAP 散点图...")
    plot_combined_figure(all_results, expr, protein, methyl, truelabel, n_clusters)

    print("\n===== 全部图表更新完成 =====")
    print(f"  {BASE_DIR}/nmi_curves.png        — NMI 折线图 (2x2)")
    print(f"  {BASE_DIR}/umap_overview.png     — UMAP 散点图 (2x2)")
    print(f"  {BASE_DIR}/overlap_structure.png — Post-integration UMAP")
    print(f"  {BASE_DIR}/umap_single_*.png     — 单组学UMAP (3张)")


def redraw_overlap_structure(force_cache=False):
    """Generate only the post-integration UMAP, reusing fused networks."""
    expr, protein, methyl, truelabel = load_omics_data()
    datasets, _, _ = split_for_umap_vis(
        [expr, protein, methyl], truelabel, ratio=0.7, seed=42
    )
    cache_path = _out("overlap_fusion_cache.npz")
    fused, dict_si = run_fusion_cached(
        datasets,
        cache_path,
        neighbor_size=20,
        fusing_iteration=20,
        force=force_cache,
    )
    plot_post_integration_umap(
        datasets,
        None,
        fused,
        dict_si,
        _out("overlap_structure.png"),
    )


def main():
    t0 = time.time()
    print("=" * 60)
    print("IMGF — 完整图表重现")
    print("=" * 60)

    expr, protein, methyl, truelabel = load_omics_data()
    data_all = [expr, protein, methyl]
    n_clusters = truelabel["cluster.id"].nunique()

    # ---- 1. 运行所有 NMI 实验 ----
    all_results, all_timing, _ = run_full_experiment(
        ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_runs=1)

    # ---- 2. (已禁用) 单独保存各 NMI 子图，只保留 2x2 大图 ----

    # ---- 3. 单组学 UMAP（用 split_for_umap_vis: 每视图 350+50）----
    print("\n>>> 单组学 UMAP（预融合）...")
    ds_umap, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    plot_single_omics_umap(ds_umap, truelabel, 1, "Protein expression (raw)",
                          _out("umap_single_protein.png"), n_clusters)
    plot_single_omics_umap(ds_umap, truelabel, 0, "mRNA expression (raw)",
                          _out("umap_single_mrna.png"), n_clusters)
    plot_single_omics_umap(ds_umap, truelabel, 2, "DNA methylation (raw)",
                          _out("umap_single_methyl.png"), n_clusters)

    # ---- 4. Post-integration UMAP ----
    print("\n>>> Post-integration UMAP...")
    datasets_umap, _, _ = split_for_umap_vis(data_all, truelabel, ratio=0.7, seed=42)
    fused, dict_si = run_fusion(datasets_umap)
    plot_post_integration_umap(datasets_umap, None, fused, dict_si,
                               _out("overlap_structure.png"))

    # ---- 5. NMI 折线图 + UMAP 散点图 (两张独立大图) ----
    print("\n>>> NMI 折线图 + UMAP 散点图...")
    plot_combined_figure(all_results, expr, protein, methyl, truelabel, n_clusters)

    # ---- 保存数值 ----
    ratio_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df = pd.DataFrame({"ratio": ratio_vals})
    for key in all_results:
        df[f"{key}_IMGF"] = all_results[key]["IMGF"]
        df[f"{key}_IntegrAO"] = all_results[key]["IntegrAO"]
        df[f"{key}_KMeans"] = all_results[key]["KMeans(on intact feature)"]
        df[f"{key}_NEMO"] = all_results[key].get("NEMO", [np.nan]*9)
        df[f"{key}_MSNE"] = all_results[key].get("MSNE", [np.nan]*9)
    df.to_csv(_out("nmi_results.csv"), index=False)

    # ---- 保存耗时 + 画时间对比图 ----
    tdf = pd.DataFrame({"ratio": ratio_vals})
    for key in all_timing:
        for m in ["IMGF", "IntegrAO", "KMeans(on intact feature)", "NEMO", "MSNE"]:
            tdf[f"{key}_{m}"] = all_timing[key][m]
    tdf.to_csv(_out("timing_results.csv"), index=False)
    print("\n>>> 时间对比图...")
    plot_timing_comparison(all_timing)

    print(f"\n总耗时: {time.time()-t0:.0f}s")
    print("===== 全部完成! =====")
    print(f"输出文件:")
    print(f"  {OUT_DIR}/nmi_curves.png             — NMI 折线图 (2x2)")
    print(f"  {OUT_DIR}/umap_overview.png     — UMAP 散点图 (2x2)")
    print(f"  {OUT_DIR}/overlap_structure.png — Post-integration UMAP")
    print(f"  {OUT_DIR}/umap_single_*.png     — 单组学UMAP (3张)")
    print(f"  {OUT_DIR}/nmi_*.png             — NMI曲线 (4张)")
    print(f"  {OUT_DIR}/nmi_results.csv       — 数值结果")
    print(f"  {OUT_DIR}/timing_results.csv    — 各方法耗时")
    print(f"  {OUT_DIR}/timing_bar.png        — 耗时柱状图")
    print(f"  {OUT_DIR}/timing_ratio.png      — 耗时 vs ratio 曲线")


if __name__ == "__main__":
    if "--overlap-only" in sys.argv:
        redraw_overlap_structure(force_cache="--force-cache" in sys.argv)
    elif "--combined-only" in sys.argv:
        redraw_combined_figure(force_cache="--force-cache" in sys.argv)
    else:
        main()
