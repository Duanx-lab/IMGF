################################################################
# crc_GSEA_K4.R
# CRC IntegrAO K=4 聚类 → GSEA 通路富集分析 (fgsea)
# 热力图颜色与 CRC_KM_K4.png 生存曲线一致
################################################################

library(limma)
library(fgsea)
library(dplyr)
library(pheatmap)
library(cowplot)
library(data.table)

# ============================================================
# 0. 颜色定义（与生存曲线 plt.cm.tab10(np.linspace(0,1,4)) 一致）
# ============================================================
# tab10 索引: 0, 3, 5, 7  (均匀间隔4色)
K4_COLORS <- c("Cluster1" = "#F9C74F",   # CS1 金黄色
               "Cluster2" = "#577590",   # CS2 蓝灰色
               "Cluster3" = "#F3722C",   # CS3 橙色
               "Cluster4" = "#6A994E")   # CS4 绿色

# ============================================================
# 1. 加载基因表达数据
# ============================================================
cat("\n>>> 加载 CRC 基因表达数据...\n")
# 从原始 RData 加载（在 RStudio 中可用）
load("D:/Mysoftware/R/data/CRC.RData")  # mydatGE (297 x 20531)

# 转置：基因 x 样本
if (is.matrix(mydatGE)) {
  data <- t(mydatGE)
} else {
  data <- t(as.matrix(mydatGE))
}

# 提取基因符号（mydatGE 列名格式: "SYMBOL|ENSEMBL"）
col_names <- colnames(mydatGE)
gene_symbols <- sapply(col_names, function(x) {
  strsplit(x, "|", fixed = TRUE)[[1]][1]
})
rownames(data) <- gene_symbols

# 去重：保留第一个出现的基因
keep <- !duplicated(rownames(data))
data <- data[keep, ]

cat(sprintf("  表达矩阵: %d 基因 x %d 样本 (去重后)\n", nrow(data), ncol(data)))

# ============================================================
# 2. 加载 K=4 聚类标签
# ============================================================
cat("\n>>> 加载 K=4 聚类标签...\n")
labels_df <- read.csv("CRC/crc_k4_labels.csv")

# 建立 原始索引 -> 聚类标签 映射
idx_to_cluster <- setNames(labels_df$cluster, labels_df$orig_index)

rna_sample_count <- ncol(data)
cat(sprintf("  表达谱样本数: %d, 标签样本数: %d\n", rna_sample_count, nrow(labels_df)))

# ============================================================
# 3. limma 差异分析：每个 Cluster vs Others
# ============================================================
cat("\n>>> limma 差异分析 (K=4)...\n")

# 为表达谱中每个样本分配聚类标签
rna_labels <- as.character(idx_to_cluster[as.character(0:(rna_sample_count - 1))] + 1)

cluster_ids <- sort(unique(na.omit(rna_labels)))
cat(sprintf("  聚类数: %d\n", length(cluster_ids)))

limma_results <- list()
for (cl in cluster_ids) {
  cat(sprintf("  处理 Cluster %s...\n", cl))
  group <- rep("0", ncol(data))
  group[which(rna_labels == cl)] <- "1"
  group <- as.factor(group)

  fit <- lmFit(data, model.matrix(~ group))
  fit <- eBayes(fit)
  tbl <- topTable(fit, coef = 2, number = Inf, adjust.method = "BH")
  limma_results[[paste0("Cluster", cl)]] <- tbl
  cat(sprintf("    显著基因数 (adj.P<0.05): %d\n", sum(tbl$adj.P.Val < 0.05)))
}

# ============================================================
# 4. 加载 genesets（保留 SYMBOL，不转 ENTREZID）
# ============================================================
cat("\n>>> 加载基因集...\n")
nf <- max(count.fields("CRC/genesets.gmt", sep = "\t"))
geneset_raw <- read.table("CRC/genesets.gmt",
                          stringsAsFactors = FALSE, fill = TRUE, col.names = 1:nf)
geneset <- lapply(1:nrow(geneset_raw), function(i) {
  gs1 <- geneset_raw[i, , drop = TRUE] %>% unlist
  gs1 <- gs1[2:length(gs1)]
  gs1 <- gs1[gs1 != ""]
  unique(gs1)
})
names(geneset) <- geneset_raw$X1
cat(sprintf("  基因集数量: %d\n", length(geneset)))

# ============================================================
# 5. GSEA 分析 (fgsea)
# ============================================================
cat("\n>>> GSEA 富集分析 (fgsea)...\n")

GSEA.rslt <- list()
set.seed(42)

for (cn in names(limma_results)) {
  cat(sprintf("  fgsea: %s...\n", cn))
  tbl <- limma_results[[cn]]
  ranks <- tbl$logFC
  names(ranks) <- rownames(tbl)
  ranks <- sort(ranks, decreasing = TRUE)
  ranks <- ranks[!duplicated(names(ranks))]

  fgsea_res <- fgsea(pathways = geneset, stats = ranks,
                     minSize = 1, maxSize = Inf, nPermSimple = 10000,
                     nproc = 2)
  GSEA.rslt[[cn]] <- fgsea_res
}

save(GSEA.rslt, file = "CRC/crc_data_gsea_K4.RData")

# 保存 fgsea 通路名
writeLines(GSEA.rslt[[1]]$pathway, "CRC/crc_fgsea_pathways.txt")
cat(sprintf("  已保存通路名 (%d 条)\n", length(GSEA.rslt[[1]]$pathway)))

# ============================================================
# 6. 构建热力图数据
# ============================================================
cat("\n>>> 构建热力图数据...\n")

cluster_names <- paste0("Cluster", cluster_ids)
all_pathways <- GSEA.rslt[[1]]$pathway
cat(sprintf("  fgsea 返回通路数: %d\n", length(all_pathways)))

gsea_list <- lapply(GSEA.rslt, function(g) {
  df <- as.data.frame(g)
  rownames(df) <- df$pathway
  df$Pvalue_trans <- -log10(df$pval + 1e-5)
  for (i in 1:nrow(df)) {
    if (!is.na(df[i, "NES"]) && df[i, "NES"] < 0) {
      df[i, "Pvalue_trans"] <- -df[i, "Pvalue_trans"]
    }
  }
  df[all_pathways, ]
})

# 构建热力图矩阵
heat.mat <- do.call(cbind, lapply(gsea_list, function(g) g$Pvalue_trans))
rownames(heat.mat) <- all_pathways
colnames(heat.mat) <- cluster_names

# 截断值
heat.mat[heat.mat <= -2] <- -2
heat.mat[heat.mat >= 2] <- 2

save(heat.mat, file = "CRC/crc_pathway_heat_K4.RData")

# ============================================================
# 7. 提取各维度通路
# ============================================================

## Signature
Signature.heat <- heat.mat[match(c("EPITH_LOBODA",
                                    "WNT_FLIER",
                                    "MYC_TARGETS_ZELLER",
                                    "MESENCH_LOBODA",
                                    "EMT_CORE_GENES",
                                    "TGFB_KEGG",
                                    "MATRIX_REMODEL_REACTOME",
                                    "WOUND_RESPONSE_GO_BP",
                                    "CSC_BATLLE"), rownames(heat.mat)), ]
rownames(Signature.heat) <- c("Epithelial", "WNT targets", "MYC targets",
                               "Mesenchymal", "EMT activation",
                               "TGFB-activation", "Matrix remodeling",
                               "Wound response", "Cancer stem cell")

## Pathways
Pathways.heat <- heat.mat[match(c("MAPK_KEGG",
                                   "PI3K_ACT_REACTOME",
                                   "SRC_ACT_BILD",
                                   "JAK_STAT_KEGG",
                                   "CASPASE_BIOCARTA",
                                   "PROTEASOME_KEGG",
                                   "KEGG_CELL_CYCLE",
                                   "TRANSLATION_RIBOS_REACTOME",
                                   "INTEGRIN_BETA3_CP",
                                   "VEGF_VEGFR_REACTOME"), rownames(heat.mat)), ]
rownames(Pathways.heat) <- c("MAPK", "PI3K", "SRC", "JAK-STAT", "Caspases",
                              "Proteosome", "Cell cycle", "Translation ribosome",
                              "Integrin-B3", "VEGF VEGFR")

## Estimate
Estimate.heat <- heat.mat[match(c("IMMUNE_ESTIMATE",
                                   "STROMAL_ESTIMATE"), rownames(heat.mat)), ]
rownames(Estimate.heat) <- c("Immune infiltration", "Stromal infiltration")

## Immune
Immune.heat <- heat.mat[match(c("IMMUNE_RESP_GO_BP",
                                 "PD1_REACTOME",
                                 "IMMUNE_NKC_BREAST",
                                 "IMMUNE_TH1_GALON",
                                 "IMMUNE_THF_BREAST",
                                 "IMMUNE_TH17_GOUNARI",
                                 "IMMUNE_TREG_GALON",
                                 "COMPLEMENT_COAG_KEGG"), rownames(heat.mat)), ]
rownames(Immune.heat) <- c("Immune response", "PD1 activation",
                            "NK cell infiltration", "TH1 infiltration",
                            "TFH infiltration", "TH17 activation",
                            "Treg activation", "Complement activation")

## Metabolism
Metabolism.heat <- heat.mat[match(c("AMINO_SUGAR_NUCLEO_METAB_KEGG",
                                     "PENTOSE_GLUC_METAB_KEGG",
                                     "FRUTOSE_MANNOSE_METAB_KEGG",
                                     "GALACTOSE_METAB_KEGG",
                                     "GLUTAMINE_GO_BP",
                                     "GLUTATHIONE_KEGG",
                                     "NITROGEN_METAB_KEGG",
                                     "GLYCEROPHOSPHOLIPID_GO_BP",
                                     "LYSOPHOSPHOLIPID_PID",
                                     "FATTY_ACID_METAB_KEGG"), rownames(heat.mat)), ]
rownames(Metabolism.heat) <- c("Sugar aa nucleotide", "Glucose pentose",
                                "Fructose mannose", "Galactose",
                                "Glutamine", "Glutathione",
                                "Nitrogen", "Glycerophospholipid",
                                "Lysophospholipid", "Fatty acid")

# ============================================================
# 8. 绘制热力图（颜色与生存曲线一致）
# ============================================================
cat("\n>>> 绘制热力图...\n")

annotation_col <- data.frame(
  Class = factor(cluster_names, levels = cluster_names)
)
rownames(annotation_col) <- cluster_names

ann_colors <- list(Class = K4_COLORS)
names(ann_colors$Class) <- cluster_names

breaks <- seq(-2, 2, length.out = 100)
heatmap_color <- colorRampPalette(c("#92C5DE", "#FFFFFF", "#F4A582"))(100)

common_args <- list(
  fontsize = 10, breaks = breaks,
  show_rownames = TRUE,
  annotation_col = annotation_col,
  annotation_colors = ann_colors,
  color = heatmap_color,
  annotation_names_col = FALSE,
  show_colnames = FALSE,
  cluster_cols = FALSE,
  cluster_rows = FALSE,
  fontsize_col = 12,
  cellwidth = 20,
  cellheight = 20
)

## Signature
Signature.p <- do.call(pheatmap, c(list(mat = Signature.heat,
  main = "Signatures", legend = TRUE, annotation_legend = TRUE), common_args))

## Pathways
Pathways.p <- do.call(pheatmap, c(list(mat = Pathways.heat,
  main = "Pathways", legend = FALSE, annotation_legend = FALSE), common_args))

## Estimate
Estimate.p <- do.call(pheatmap, c(list(mat = Estimate.heat,
  main = "Estimate", legend = FALSE, annotation_legend = FALSE), common_args))

## Immune
Immune.p <- do.call(pheatmap, c(list(mat = Immune.heat,
  main = "Immune", legend = FALSE, annotation_legend = FALSE), common_args))

## Metabolism
Metabolism.p <- do.call(pheatmap, c(list(mat = Metabolism.heat,
  main = "Metabolism", legend = FALSE, annotation_legend = FALSE), common_args))

# ============================================================
# 9. 组合图输出
# ============================================================
cat("\n>>> 输出组合图...\n")

out_dir <- "CRC"
save(Signature.p, Pathways.p, Immune.p, Metabolism.p, Estimate.p,
     file = file.path(out_dir, "crc_Figure_K4.RData"))

pdf(file.path(out_dir, "crc_pathway_K4.pdf"), height = 12, width = 10)
plot_grid(Signature.p$gtable, Pathways.p$gtable, Estimate.p$gtable,
          Metabolism.p$gtable, Immune.p$gtable, labels = "auto",
          ncol = 2, rel_heights = c(3, 3, 2.2),
          align = "v")
dev.off()
cat("  PDF 已保存: crc_pathway_K4.pdf\n")

png(file.path(out_dir, "crc_pathway_K4.png"), height = 12, width = 10,
    units = "in", res = 200)
plot_grid(Signature.p$gtable, Pathways.p$gtable, Estimate.p$gtable,
          Metabolism.p$gtable, Immune.p$gtable, labels = "auto",
          ncol = 2, rel_heights = c(3, 3, 2.2),
          align = "v")
dev.off()
cat("  PNG 已保存: crc_pathway_K4.png\n")

cat("\n===== CRC GSEA K=4 分析完成! =====\n")
