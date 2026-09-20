"""
NEMO 和 MSNE 多组学整合方法的 Python 实现。

NEMO (Rappoport & Shamir, 2019): 相对相似性变换 + 平均 + 谱聚类
MSNE (Xu et al., 2020): 跨网络随机游走 + SVD 嵌入 + KMeans
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import cdist


# ============================================================
# 共享工具
# ============================================================

def _extract_true_labels(union_labels, all_names):
    """
    从 union_labels 中按 all_names 顺序提取真实标签数组。
    兼容两种格式：
    - 字符串索引（Scenario A）：直接按名称查找
    - 整数索引（Scenario B）：从名称中提取原始 ID 后查找
    """
    if hasattr(union_labels.index, 'dtype') and union_labels.index.dtype.kind in ('i', 'u'):
        # 整数索引（0..499）
        labels = []
        for name in all_names:
            orig_id = int(name.split('_')[1])
            labels.append(union_labels[orig_id])
        return np.array(labels)
    else:
        # 字符串索引
        return np.array([union_labels[n] for n in all_names])


def _build_adaptive_similarity(X, k):
    """
    自适应带宽 RBF 核相似性矩阵（NEMO 和 MSNE 共用）。

    S(i,j) = exp(-||xi-xj||^2 / (2 * sigma_ij^2)) / sqrt(2*pi*sigma_ij^2)

    sigma_ij^2 = (avg_dist_sq(i) + avg_dist_sq(j) + dist(i,j)^2) / 3
    avg_dist_sq(i) = (1/k) * sum_{r in N_k(i)} ||xi - xr||^2
    """
    n = X.shape[0]
    k_eff = min(k + 1, n)

    nn = NearestNeighbors(n_neighbors=k_eff, metric='euclidean')
    nn.fit(X)
    dists, indices = nn.kneighbors(X)

    # 每个样本到其 k 近邻的平均平方距离
    avg_dist_sq = np.zeros(n)
    for i in range(n):
        avg_dist_sq[i] = np.mean(dists[i, 1:] ** 2)

    S = np.zeros((n, n))
    for i in range(n):
        for j_idx in range(1, k_eff):
            j = indices[i, j_idx]
            if i >= j:
                continue  # 对上三角计算，后面镜像
            d_sq = dists[i, j_idx] ** 2
            sigma_sq = (avg_dist_sq[i] + avg_dist_sq[j] + d_sq) / 3.0
            sigma_sq = max(sigma_sq, 1e-12)
            val = np.exp(-d_sq / (2.0 * sigma_sq)) / np.sqrt(2.0 * np.pi * sigma_sq)
            S[i, j] = val
            S[j, i] = val

    return S, indices, dists


def _build_relative_similarity(X, k):
    """
    计算一个组学视图的相对相似性 RS 矩阵（NEMO 算法核心）。

    RS_l(i,j) = S(i,j)/sum_r S(i,r) * I(j in N_k(i))
               + S(i,j)/sum_r S(j,r) * I(i in N_k(j))

    RS 值域 [0, 2]。
    """
    n = X.shape[0]
    k_eff = min(k + 1, n)

    nn = NearestNeighbors(n_neighbors=k_eff, metric='euclidean')
    nn.fit(X)
    dists, indices = nn.kneighbors(X)

    avg_dist_sq = np.zeros(n)
    for i in range(n):
        avg_dist_sq[i] = np.mean(dists[i, 1:] ** 2)

    # 预计算所有近邻对之间的 S(i,j)
    S_neighbors = {}  # (i,j) -> S_val
    neighbor_sets = []
    for i in range(n):
        eta_i = set(indices[i, 1:])
        neighbor_sets.append(eta_i)
        S_sum_i = 0.0
        S_i_vals = {}
        for j_idx in range(1, k_eff):
            j = indices[i, j_idx]
            d_sq = dists[i, j_idx] ** 2
            sigma_sq = (avg_dist_sq[i] + avg_dist_sq[j] + d_sq) / 3.0
            sigma_sq = max(sigma_sq, 1e-12)
            val = np.exp(-d_sq / (2.0 * sigma_sq)) / np.sqrt(2.0 * np.pi * sigma_sq)
            S_i_vals[j] = val
            S_sum_i += val
            S_neighbors[(i, j)] = val
        S_neighbors[(i, 'sum')] = S_sum_i

    RS = np.zeros((n, n))
    for i in range(n):
        eta_i = neighbor_sets[i]
        sum_i = S_neighbors.get((i, 'sum'), 1e-12)
        for j in eta_i:
            S_ij = S_neighbors.get((i, j), 0.0)
            # 第一项：j 在 i 的邻域中
            RS[i, j] += S_ij / max(sum_i, 1e-12)
        # 第二项：i 在 j 的邻域中
        for j in range(n):
            if i != j and i in neighbor_sets[j]:
                S_ji = S_neighbors.get((j, i), 0.0)
                sum_j = S_neighbors.get((j, 'sum'), 1e-12)
                RS[i, j] += S_ji / max(sum_j, 1e-12)

    return RS


# ============================================================
# NEMO 完整流程
# ============================================================

def nemo_nmi(datasets, union_labels, n_clusters):
    """
    NEMO: 相对相似性 → 平均 ARS → 谱聚类 → NMI。

    参数:
        datasets: DataFrame 列表，每项为一个组学视图
        union_labels: Series，样本名 → 真实簇标签
        n_clusters: 聚类数

    返回:
        NMI 分数
    """
    n_views = len(datasets)

    # 构建联合样本空间
    all_names = sorted(set().union(*(set(d.index) for d in datasets)))
    name_to_union = {name: i for i, name in enumerate(all_names)}
    n_union = len(all_names)

    # k = n / K（NEMO 默认）
    k = max(5, n_union // n_clusters)

    RS_matrices = []
    view_name_lists = []

    for df in datasets:
        X = df.values
        names = df.index.tolist()
        RS = _build_relative_similarity(X, k)
        RS_matrices.append(RS)
        view_name_lists.append(names)

    # 在联合空间中求平均 ARS
    ARS = np.zeros((n_union, n_union))
    count = np.zeros((n_union, n_union))

    for RS, names in zip(RS_matrices, view_name_lists):
        local_idx = [name_to_union[n] for n in names]
        n_local = len(names)
        for li in range(n_local):
            u = local_idx[li]
            for lj in range(n_local):
                v = local_idx[lj]
                ARS[u, v] += RS[li, lj]
                count[u, v] += 1

    count[count == 0] = 1.0
    ARS /= count

    # 谱聚类
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                            assign_labels='discretize', random_state=42)
    try:
        pred = sc.fit_predict(ARS)
    except Exception:
        return np.nan

    # NMI（样本按 all_names 顺序）
    true_arr = _extract_true_labels(union_labels, all_names)
    return normalized_mutual_info_score(true_arr, pred)


# ============================================================
# MSNE 完整流程
# ============================================================

def _cross_network_random_walks(Q_matrices, view_name_lists, name_to_union,
                                 n_union, num_walks=100, walk_length=20):
    """
    跨多重相似性网络的随机游走（MSNE 核心创新）。

    每次游走：
    1. 从随机节点开始
    2. 每步先选网络（均匀），再按边权选下一个节点
    3. 若当前节点在某网络中无邻居，跳过该网络
    """
    n_views = len(Q_matrices)

    # 为每个视图建立：union_idx -> local_idx 映射
    union_to_local = []
    for names in view_name_lists:
        mapping = {}
        for li, name in enumerate(names):
            mapping[name_to_union[name]] = li
        union_to_local.append(mapping)

    # 为每个视图预计算邻居列表和采样权重（仅非零边）
    neighbor_info = []
    for v_idx, Q in enumerate(Q_matrices):
        n_local = Q.shape[0]
        neighbors = []
        weights = []
        for i in range(n_local):
            row = Q[i]
            nz = np.where(row > 0)[0]
            # 排除自身
            nz = nz[nz != i]
            if len(nz) == 0:
                neighbors.append(np.array([i]))
                weights.append(np.array([1.0]))
            else:
                neighbors.append(nz)
                w = row[nz]
                w = w / w.sum()
                weights.append(w)
        neighbor_info.append((neighbors, weights))

    # 每个节点进行 num_walks 次随机游走
    all_walks = []
    for start_union in range(n_union):
        for _ in range(num_walks):
            walk = [start_union]
            current_union = start_union

            for _ in range(walk_length - 1):
                # 找出包含当前节点的网络
                available_views = []
                for v_idx in range(n_views):
                    if current_union in union_to_local[v_idx]:
                        available_views.append(v_idx)

                if not available_views:
                    break

                # 随机选一个网络
                v_chosen = available_views[np.random.randint(len(available_views))]
                local_idx = union_to_local[v_chosen][current_union]

                # 在选定网络中按边权采样下一个节点
                neigh, wgt = neighbor_info[v_chosen]
                next_local = np.random.choice(neigh[local_idx], p=wgt[local_idx])

                # 映射回 union 空间
                local_name = view_name_lists[v_chosen][next_local]
                next_union = name_to_union[local_name]

                walk.append(next_union)
                current_union = next_union

            all_walks.append(walk)

    return all_walks


def _walks_to_embeddings(walks, n_nodes, n_components=100, window_size=10):
    """
    从随机游走序列构建共现矩阵 → PMI → SVD 嵌入。
    等价于 DeepWalk / MSNE 的矩阵分解形式。
    """
    # 构建共现矩阵
    cooc = np.zeros((n_nodes, n_nodes))
    for walk in walks:
        for pos_i, node_i in enumerate(walk):
            win_start = max(0, pos_i - window_size)
            win_end = min(len(walk), pos_i + window_size + 1)
            for pos_j in range(win_start, win_end):
                if pos_i != pos_j:
                    cooc[node_i, walk[pos_j]] += 1

    # PMI 变换 (Positive Pointwise Mutual Information)
    total = cooc.sum()
    if total == 0:
        return np.random.randn(n_nodes, n_components) * 0.01

    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)

    expected = (row_sum @ col_sum) / total
    expected[expected == 0] = 1.0

    PMI = cooc / expected
    PMI = np.maximum(np.log(np.maximum(PMI, 1e-12)), 0)  # PPMI

    # SVD
    try:
        U, S, Vt = np.linalg.svd(PMI, full_matrices=False)
        n_comp = min(n_components, len(S))
        embeddings = U[:, :n_comp] * np.sqrt(S[:n_comp])
    except np.linalg.LinAlgError:
        embeddings = np.random.randn(n_nodes, n_components) * 0.01

    return embeddings


def msne_nmi(datasets, union_labels, n_clusters, k=20, num_walks=100,
             walk_length=20, embed_size=100, window_size=10):
    """
    MSNE: 跨网络随机游走 → SVD 嵌入 → KMeans → NMI。

    参数:
        datasets: DataFrame 列表，每项为一个组学视图
        union_labels: Series，样本名 → 真实簇标签
        n_clusters: 聚类数
        k: 局部邻域大小
        num_walks: 每节点游走次数
        walk_length: 每次游走长度
        embed_size: 嵌入维度
        window_size: Skip-gram 窗口大小

    返回:
        NMI 分数
    """
    n_views = len(datasets)

    # 联合样本空间
    all_names = sorted(set().union(*(set(d.index) for d in datasets)))
    name_to_union = {name: i for i, name in enumerate(all_names)}
    n_union = len(all_names)

    # Step 1-4: 为每个组学构建 Q 矩阵
    Q_matrices = []
    view_name_lists = []

    for df in datasets:
        X = df.values
        names = df.index.tolist()
        n_local = len(X)

        # 自适应带宽相似性
        S, indices, _ = _build_adaptive_similarity(X, k)

        # 稀疏化：只保留 top-k 邻居
        S_sparse = np.zeros_like(S)
        for i in range(n_local):
            for j_idx in range(1, min(k + 1, n_local)):
                j = indices[i, j_idx]
                S_sparse[i, j] = S[i, j]

        # 归一化为转移概率
        row_sum = S_sparse.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        P = S_sparse / row_sum

        # 马尔可夫扩散：P^2
        Pn = P @ P

        # 对称化
        Q = (Pn + Pn.T) / 2.0
        Q_matrices.append(Q)
        view_name_lists.append(names)

    # Step 5: 跨网络随机游走
    walks = _cross_network_random_walks(
        Q_matrices, view_name_lists, name_to_union, n_union,
        num_walks=num_walks, walk_length=walk_length)

    # Step 6: 嵌入
    embeddings = _walks_to_embeddings(walks, n_union,
                                       n_components=embed_size,
                                       window_size=window_size)

    # Step 7: KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    pred = km.fit_predict(embeddings)

    # NMI
    true_arr = _extract_true_labels(union_labels, all_names)
    return normalized_mutual_info_score(true_arr, pred)
