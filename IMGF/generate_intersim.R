#!/usr/bin/env Rscript
# 用 InterSIM 生成 15 簇模拟多组学数据，保存为 IntegrAO 格式
# 用法: Rscript generate_intersim.R <delta> <outdir> [seed]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("用法: Rscript generate_intersim.R <delta> <outdir> [seed] [p.DMP]")
delta  <- as.numeric(args[1])
outdir <- args[2]
seed   <- if (length(args) >= 3) as.integer(args[3]) else 42L
p_dmp  <- if (length(args) >= 4) as.numeric(args[4]) else 0.2

suppressMessages(library(InterSIM))

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(seed)
prop <- rep(1 / 15, 15)
res <- InterSIM(
    n.sample = 500, cluster.sample.prop = prop,
    delta.methyl = delta, delta.expr = delta, delta.protein = delta,
    p.DMP = p_dmp,
    do.plot = FALSE, sample.cluster = FALSE, feature.cluster = FALSE
)

save_omics <- function(mat, file, prefix) {
    # mat: 样本 x 特征
    if (is.null(rownames(mat))) rownames(mat) <- paste0("subject", seq_len(nrow(mat)))
    mat_t <- t(mat)  # 特征 x 样本
    feat <- colnames(mat)
    if (is.null(feat) || !length(feat)) feat <- paste0(prefix, seq_len(ncol(mat)))
    df <- data.frame(probe = feat, mat_t, check.names = FALSE, stringsAsFactors = FALSE)
    write.table(df, file = file, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
}

save_omics(res$dat.methyl,  file.path(outdir, "omics1.txt"), "cg")
save_omics(res$dat.expr,    file.path(outdir, "omics2.txt"), "gene")
save_omics(res$dat.protein, file.path(outdir, "omics3.txt"), "protein")

clust <- res$clustering.assignment
write.table(clust, file = file.path(outdir, "clusters.txt"),
            sep = "\t", quote = TRUE, row.names = TRUE, col.names = TRUE)

cat("done:", outdir, "delta:", delta, "\n")
