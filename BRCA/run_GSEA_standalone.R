# ============================================================
# GSEA 分析 — 自包含版本 (无需 org.Hs.eg.db / HTSanalyzeR2)
# 输入: genesets.gmt, BRCA1.RData
# 输出: R 文件夹下所有结果
# ============================================================

SCRIPT_DIR <- normalizePath(dirname(sub("--file=", "",
    commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))])))
if (is.na(SCRIPT_DIR) || length(SCRIPT_DIR) == 0) {
  SCRIPT_DIR <- normalizePath(".")
}
setwd(SCRIPT_DIR)
cat("工作目录:", SCRIPT_DIR, "\n")

suppressPackageStartupMessages({
  library(Biobase)
  library(SummarizedExperiment)
  library(limma)
  library(dplyr)
  library(pheatmap)
  library(cowplot)
})

# ============================================================
# 1. 基因集富集分析 (基于秩和检验, 快速)
# ============================================================
gsea_ranksum <- function(gene_stats, gene_sets) {
  # gene_stats: named numeric vector of per-gene statistics (e.g. logFC)
  # gene_sets: named list of gene symbol vectors
  # Returns: data.frame with signed enrichment score and p-value

  gene_names <- names(gene_stats)
  # 用 Wilcoxon rank-sum test
  results <- data.frame(
    Gene.Set = names(gene_sets),
    Size = NA_integer_,
    ES = NA_real_,
    NES = NA_real_,
    Pvalue = NA_real_,
    Observed.score = NA_real_,
    stringsAsFactors = FALSE
  )

  # 全局排名
  ranks <- rank(gene_stats)
  n_total <- length(ranks)

  for (i in seq_along(gene_sets)) {
    gs_genes <- intersect(gene_sets[[i]], gene_names)
    n_gs <- length(gs_genes)
    results$Size[i] <- n_gs

    if (n_gs < 2) {
      results$ES[i] <- 0
      results$NES[i] <- 0
      results$Pvalue[i] <- 1
      results$Observed.score[i] <- 0
      next
    }

    in_set <- gene_names %in% gs_genes
    ranks_in <- ranks[in_set]
    ranks_out <- ranks[!in_set]

    # Wilcoxon 秩和检验
    wt <- wilcox.test(ranks_in, ranks_out, alternative = "two.sided")
    pval <- wt$p.value

    # 富集分数: 平均秩的标准化差
    mean_rank_in <- mean(ranks_in)
    mean_rank_out <- mean(ranks_out)
    sd_pooled <- sd(ranks) * sqrt(1/n_gs + 1/(n_total - n_gs))
    if (sd_pooled < 1e-8) sd_pooled <- 1e-8
    z_score <- (mean_rank_in - mean_rank_out) / sd_pooled

    # 方向: 基因集内基因的 logFC 均值符号
    direction <- sign(mean(gene_stats[gs_genes]))

    results$ES[i] <- direction * abs(z_score)
    results$NES[i] <- z_score
    results$Pvalue[i] <- pval
    results$Observed.score[i] <- direction * abs(z_score)
  }

  results <- results[order(results$Pvalue), ]
  rownames(results) <- results$Gene.Set
  results
}

# ============================================================
# 2. 加载基因集
# ============================================================
cat("\n>>> 加载基因集...\n")
nf <- max(count.fields(file.path(SCRIPT_DIR, "genesets.gmt"), sep = "\t"))
geneset_raw <- read.table(file.path(SCRIPT_DIR, "genesets.gmt"),
                          stringsAsFactors = FALSE, fill = TRUE, col.names = 1:nf)

gene_sets <- lapply(1:nrow(geneset_raw), function(i) {
  gs <- geneset_raw[i, , drop = TRUE] %>% unlist()
  gs <- gs[2:length(gs)]
  gs <- gs[gs != ""]
  unique(gs)
})
names(gene_sets) <- geneset_raw$X1
cat("  基因集:", length(gene_sets), "个\n")
cat("  平均大小:", round(mean(lengths(gene_sets))), "个基因\n")

# ============================================================
# 3. 加载表达数据
# ============================================================
cat("\n>>> 加载表达数据...\n")
load("D:/Mysoftware/R/data/BRCA1.RData")
cat("  已加载:", paste(ls(), collapse = ", "), "\n")

if (is.matrix(mydatGE)) {
  data <- t(mydatGE)
} else {
  data <- t(as.matrix(mydatGE))
}
cat("  表达矩阵:", nrow(data), "genes x", ncol(data), "samples\n")

# ============================================================
# 4. 样本标签 (从 CSV 读取 K-means 6分类)
# ============================================================
labels_csv <- file.path(SCRIPT_DIR, "brca_labels.csv")
if (file.exists(labels_csv)) {
  label_df <- read.csv(labels_csv, stringsAsFactors = FALSE)
  ws <- as.character(label_df$class_id)
  names(ws) <- label_df$sample
  cat("  从 brca_labels.csv 读取标签\n")
} else if (exists("lab")) {
  ws <- lab
  cat("  从 RData 读取 lab\n")
} else {
  ws <- as.character(as.numeric(as.factor(survival$PAM50)))
  cat("  从 survival$PAM50 推导标签\n")
}
ws <- ws[intersect(names(ws), colnames(data))]
data <- data[, names(ws)]
cat("  标签分布:\n")
print(table(ws))
classes_present <- sort(unique(ws))
n_classes <- length(classes_present)

# ============================================================
# 5. Limma 差异分析
# ============================================================
cat("\n>>> Limma 差异分析...\n")
limma.rslt <- list()
for (cl in classes_present) {
  group <- rep("0", ncol(data))
  group[which(ws == cl)] <- "1"
  group <- as.factor(group)

  fit <- lmFit(data, model.matrix(~ group))
  fit <- eBayes(fit)
  res <- topTable(fit, coef = 2, number = Inf, adjust.method = "BH")
  limma.rslt[[paste0("class", cl, ".limma")]] <- res
  cat(sprintf("  Class %s: %d DE genes (adj.P < 0.05)\n",
              cl, sum(res$adj.P.Val < 0.05)))
}

# ============================================================
# 6. 基因集富集分析 (秩和检验)
# ============================================================
cat("\n>>> 基因集富集分析 (秩和检验)...\n")
GSEA.rslt <- list()
for (i in seq_along(limma.rslt)) {
  cl_name <- names(limma.rslt)[i]
  res <- limma.rslt[[i]]
  gene_stats <- res$logFC
  names(gene_stats) <- rownames(res)
  gene_stats <- gene_stats[order(abs(gene_stats), decreasing = TRUE)]
  gene_stats <- gene_stats[!duplicated(names(gene_stats))]

  gsea_out <- gsea_ranksum(gene_stats, gene_sets)
  GSEA.rslt[[cl_name]] <- gsea_out
  cat(sprintf("  %s: %d / %d sets significant (P < 0.05)\n",
              cl_name, sum(gsea_out$Pvalue < 0.05), nrow(gsea_out)))
}

save(GSEA.rslt, file = file.path(SCRIPT_DIR, "data.gsea.RData"))
cat("  data.gsea.RData 已保存\n")

# ============================================================
# 7. 构建热图数据
# ============================================================
cat("\n>>> 构建热图数据...\n")
gsea_list <- lapply(seq_along(GSEA.rslt), function(i) {
  df <- GSEA.rslt[[i]]
  signed_logp <- -log10(df$Pvalue + 1e-5)
  signed_logp[df$Observed.score < 0] <- -signed_logp[df$Observed.score < 0]
  data.frame(row.names = df$Gene.Set, Score = signed_logp)
})

all_pathways <- unique(unlist(lapply(gsea_list, rownames)))
heat.dat <- do.call(cbind, lapply(gsea_list, function(x) {
  x[all_pathways, "Score"]
}))
rownames(heat.dat) <- all_pathways
colnames(heat.dat) <- paste0("Class", classes_present)
heat.dat[is.na(heat.dat)] <- 0
save(heat.dat, file = file.path(SCRIPT_DIR, "pathway.heat.RData"))
cat("  pathway.heat.RData 已保存, 维度:", nrow(heat.dat), "x", ncol(heat.dat), "\n")

# ============================================================
# 8. 生成热图
# ============================================================
cat("\n>>> 生成热图...\n")

heat.dat[heat.dat <= -2] <- -2
heat.dat[heat.dat >= 2] <- 2

annotation_col <- data.frame(
  Class = as.factor(paste0("Class", classes_present)),
  row.names = colnames(heat.dat)
)

color_list <- c("#0073C2FF", "#EFC000FF", "#868686FF", "#CD534CFF",
                "#7AA6DCFF", "#003C67FF", "#8F7700FF", "#3B3B3BFF")
class_colors <- setNames(color_list[1:n_classes], paste0("Class", classes_present))
ann_colors <- list(Class = class_colors)

breaks <- c(seq(-2, 2, length.out = 100))
heat_colors <- colorRampPalette(c("#92C5DE", "#FFFFFF", "#F4A582"))(100)

plot_heatmap_safe <- function(heat_data, filename, main_title,
                               show_legend = TRUE, show_annotation_legend = TRUE) {
  if (is.null(heat_data) || nrow(heat_data) == 0) {
    cat("  跳过", main_title, ": 无匹配通路\n")
    return(NULL)
  }
  heat_data <- heat_data[rowSums(is.na(heat_data)) == 0, , drop = FALSE]
  if (nrow(heat_data) == 0) {
    cat("  跳过", main_title, ": 无有效数据\n")
    return(NULL)
  }
  # 保存单个热图到 PNG
  png(file.path(SCRIPT_DIR, filename), width = 8, height = max(3, nrow(heat_data) * 0.4),
      units = "in", res = 200)
  p <- pheatmap(heat_data,
           fontsize = 10, breaks = breaks,
           show_rownames = TRUE,
           annotation_col = annotation_col,
           annotation_colors = ann_colors,
           color = heat_colors,
           annotation_names_col = FALSE, show_colnames = FALSE,
           cluster_cols = FALSE, cluster_rows = FALSE, fontsize_col = 12,
           cellwidth = 20, cellheight = 20,
           legend = show_legend,
           annotation_legend = show_annotation_legend,
           main = main_title)
  dev.off()
  return(p)
}

extract_pathways <- function(heat_data, gene_names, rename_vec) {
  idx <- match(gene_names, rownames(heat_data))
  valid <- !is.na(idx)
  if (sum(valid) == 0) return(NULL)
  out <- heat_data[idx[valid], , drop = FALSE]
  rownames(out) <- rename_vec[valid]
  out
}

# Signature 热图
sig_genes <- c("EPITH_LOBODA", "WNT_FLIER", "MYC_TARGETS_ZELLER",
               "MESENCH_LOBODA", "EMT_CORE_GENES", "TGFB_KEGG",
               "MATRIX_REMODEL_REACTOME", "WOUND_RESPONSE_GO_BP", "CSC_BATLLE")
sig_rename <- c("Epithelial", "WNT targets", "MYC targets", "Mesenchymal",
                "EMT activation", "TGFB-activation",
                "Matrix remodeling", "Wound response", "Cancer stem cell")
Signature.heat <- extract_pathways(heat.dat, sig_genes, sig_rename)
Signature.p <- plot_heatmap_safe(Signature.heat, "Signature.png", "Signatures")

# Pathways 热图
path_genes <- c("MAPK_KEGG", "PI3K_ACT_REACTOME", "SRC_ACT_BILD",
                "JAK_STAT_KEGG", "CASPASE_BIOCARTA", "PROTEASOME_KEGG",
                "KEGG_CELL_CYCLE", "TRANSLATION_RIBOS_REACTOME",
                "INTEGRIN_BETA3_CP", "VEGF_VEGFR_REACTOME")
path_rename <- c("MAPK", "PI3K", "SRC", "JAK-STAT", "Caspases", "Proteosome",
                 "Cell cycle", "Translation ribosome", "Integrin-B3", "VEGF VEGFR")
Pathways.heat <- extract_pathways(heat.dat, path_genes, path_rename)
Pathways.p <- plot_heatmap_safe(Pathways.heat, "Pathways.png", "Pathways", FALSE, FALSE)

# Estimate 热图
est_genes <- c("IMMUNE_ESTIMATE", "STROMAL_ESTIMATE")
est_rename <- c("Immune infiltration", "Stromal infiltration")
Estimate.heat <- extract_pathways(heat.dat, est_genes, est_rename)
Estimate.p <- plot_heatmap_safe(Estimate.heat, "Estimate.png", "Estimate", FALSE, FALSE)

# Immune 热图
imm_genes <- c("IMMUNE_RESP_GO_BP", "PD1_REACTOME", "IMMUNE_NKC_BREAST",
               "IMMUNE_TH1_GALON", "IMMUNE_THF_BREAST", "IMMUNE_TH17_GOUNARI",
               "IMMUNE_TREG_GALON", "COMPLEMENT_COAG_KEGG")
imm_rename <- c("Immune response", "PD1 activation", "NK cell infiltration",
                "TH1 infiltration", "TFH infiltration", "TH17 activation",
                "Treg activation", "Complement activation")
Immune.heat <- extract_pathways(heat.dat, imm_genes, imm_rename)
Immune.p <- plot_heatmap_safe(Immune.heat, "Immune.png", "Immune", FALSE, FALSE)

# Metabolism 热图
met_genes <- c("AMINO_SUGAR_NUCLEO_METAB_KEGG", "PENTOSE_GLUC_METAB_KEGG",
               "FRUTOSE_MANNOSE_METAB_KEGG", "GALACTOSE_METAB_KEGG",
               "GLUTAMINE_GO_BP", "GLUTATHIONE_KEGG",
               "NITROGEN_METAB_KEGG", "GLYCEROPHOSPHOLIPID_GO_BP",
               "LYSOPHOSPHOLIPID_PID", "FATTY_ACID_METAB_KEGG")
met_rename <- c("Sugar aa nucleotide", "Glucose pentose", "Fructose mannose",
                "Galactose", "Glutamine", "Glutathione",
                "Nitrogen", "Glycerophospholipid", "Lysophospholipid", "Fatty acid")
Metabolism.heat <- extract_pathways(heat.dat, met_genes, met_rename)
Metabolism.p <- plot_heatmap_safe(Metabolism.heat, "Metabolism.png", "Metabolism", FALSE, FALSE)

# ---- 保存和组合 ----
save(Signature.p, Pathways.p, Immune.p, Metabolism.p, Estimate.p,
     file = file.path(SCRIPT_DIR, "Figure.RData"))

cat("\n>>> 组合热图...\n")
gtable_list <- list()
if (!is.null(Signature.p)) gtable_list$Signature <- Signature.p$gtable
if (!is.null(Pathways.p))  gtable_list$Pathways  <- Pathways.p$gtable
if (!is.null(Estimate.p))  gtable_list$Estimate   <- Estimate.p$gtable
if (!is.null(Metabolism.p)) gtable_list$Metabolism <- Metabolism.p$gtable
if (!is.null(Immune.p))    gtable_list$Immune     <- Immune.p$gtable

if (length(gtable_list) > 0) {
  png(file.path(SCRIPT_DIR, "pathway_combined.png"), height = 14, width = 12,
      units = "in", res = 200)
  args <- c(gtable_list, list(ncol = 2))
  do.call(gridExtra::grid.arrange, args)
  dev.off()
  cat("  pathway_combined.png 已保存\n")
}

# ---- 完成 ----
cat("\n===== GSEA 分析完成 =====\n")
cat("输出文件 (", SCRIPT_DIR, "):\n", sep = "")
for (f in dir(SCRIPT_DIR, pattern = "\\.(png|RData)$")) {
  cat("  ", f, "\n")
}
