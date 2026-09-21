# IMGF

IMGF（Integrative Multi-omics Graph Fusion）是一种面向部分重叠多组学数据的无监督整合与患者分型方法。

方法流程：**网络扩散融合 → Diffusion Map 降维 → KMeans 聚类**。它能够在不做样本剔除或插补的前提下，融合 DNA 甲基化、mRNA 表达、蛋白质表达三个组学视图，对仅部分重叠的患者样本进行聚类分型。

## 方法对比

本仓库同时实现了若干基线/对比方法：

| 方法 | 流程 | 说明 |
|------|------|------|
| **IMGF** | 融合 → Diffusion Map → KMeans | 本方法 |
| IntegrAO | 融合 → GNN → 谱聚类 | 论文原版 baseline（依赖 [IntegrAO](https://github.com/bowang-lab/IntegrAO) 包） |
| NEMO | 相对相似性变换 → 谱聚类 | Rappoport & Shamir, 2019 |
| MSNE | 跨网络随机游走 → SVD → KMeans | Xu et al., 2020 |
| KMeans | 单组学原始特征 KMeans | 基线 |

## 目录结构

```
.
├── reproduce_figures.py            # 主脚本：融合、聚类、NMI 实验、出图
├── nemo_msne.py                    # NEMO / MSNE 对比方法实现
├── generate_intersim.R             # 用 InterSIM 生成模拟多组学数据
├── make_final_figure.py            # 生成最终 2x4 组合大图（NMI + UMAP）
├── BRCA/                           # 乳腺癌下游分析：GSEA、生存、分类器
├── CRC/                            # 结直肠癌下游分析：GSEA、生存、分类器
├── brca_survival_analysis.py       # IMGF 聚类 + 生存分析
├── crc_survival_analysis.py        # IMGF 聚类 + 生存分析
├── make_BRCA_classifier_all_metrics.R  # 多组学分类器对比
├── CRC_classifier_comparison.R     # 多组学分类器对比
└── README.md
```

## 环境依赖

- Python 3.10
- PyTorch 2.1.0（CPU 版即可）
- torch-geometric 2.7.0
- snfpy、umap-learn、scikit-learn、numpy、pandas、scipy
- R ≥ 4.3（仅数据生成需要，需安装 [InterSIM](https://CRAN.R-project.org/package=InterSIM) 包）
- [IntegrAO](https://github.com/bowang-lab/IntegrAO) 包（baseline 方法依赖）

```bash
pip install integrao snfpy umap-learn scikit-learn numpy pandas scipy
pip install torch==2.1.0 torch-geometric==2.7.0
```

## 运行方法

### 1. 生成模拟数据

用 InterSIM 生成 500 样本 × 15 亚型、三个组学的模拟数据，保存为 IntegrAO 格式：

```bash
Rscript generate_intersim.R <delta> <outdir> [seed] [p.DMP]
```

- `delta`：信噪比（默认约 0.9，越小越难分）
- `p.DMP`：差分特征比例（默认 0.2，越小信号越稀疏）

### 2. 运行对比实验与出图

```bash
OMICS_DATA_DIR=<数据目录> OUT_TAG=<输出后缀> \
INTEGRAO_EPOCHS=1000 IMGF_NDM=8 \
python reproduce_figures.py
```

主要环境变量：

| 变量 | 含义 | 默认 |
|------|------|------|
| `OMICS_DATA_DIR` | 组学数据目录（含 omics1/2/3.txt 与 clusters.txt） | IntegrAO 自带数据 |
| `OUT_TAG` | 输出文件名后缀 | 空 |
| `INTEGRAO_EPOCHS` | IntegrAO GNN 训练轮数 | 1000 |
| `IMGF_NDM` | IMGF 的 Diffusion Map 维度 | 3 |

### 3. 生成最终组合大图

```bash
python make_final_figure.py
```

生成 `final_figure_*.png/pdf`（上排 NMI vs overlap ratio，下排 UMAP）。

## 数据说明

### 模拟数据

模拟数据由 InterSIM R 包生成，基于 TCGA 卵巢癌数据的真实协方差结构，生成 DNA 甲基化 + mRNA 表达 + 蛋白质表达三个互相关联的组学视图，并指定亚型标签。

### 真实数据

在真实癌症数据上验证 IMGF 的分型效果，包括下游的生存分析（KM 曲线）、GSEA 富集分析和多组学分类器：

- **BRCA**：TCGA 乳腺癌 + METABRIC 乳腺癌
- **CRC**：GEO GSE39582 结直肠癌

原始数据文件较大，未随仓库托管，需自行从 TCGA / GEO 下载后运行对应脚本。
