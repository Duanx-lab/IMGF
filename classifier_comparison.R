# ============================================================
# BRCA 三组学 (GE + MI + ME) 分类器对比分析
# Diffusion Map 降维 → 7 种特征组合 → 5 种分类器 → 指标对比
# 外部验证: METABRIC GE 数据 (特征对齐 → RF 预测 → PAM50 对比)
# ============================================================

# ---- 0. 环境准备 ----
rm(list = ls())   # 清理残留变量，保持干净环境

# 设置国内镜像 (若默认 CRAN 不通)
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

cran_pkgs <- c("e1071", "randomForest", "xgboost", "nnet", "class",
               "caret", "ggplot2", "reshape2", "survival", "survminer",
               "gridExtra", "pdist", "mclust")
for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# destiny 在 Bioconductor, 不在 CRAN
if (!requireNamespace("destiny", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("destiny", update = FALSE, ask = FALSE)
}
# destiny 可能需要 smoother 包作为依赖
tryCatch({
  library(destiny)
  USE_DESTINY <- TRUE
}, error = function(e) {
  cat("destiny 加载失败:", conditionMessage(e), "\n")
  cat("尝试安装缺失依赖...\n")
  tryCatch({
    install.packages("smoother")
    library(destiny)
    USE_DESTINY <- TRUE
  }, error = function(e2) {
    cat("将使用手动 Diffusion Map 实现\n")
    USE_DESTINY <<- FALSE
  })
})

# ---- 自定义 Diffusion Map (备用: 若 destiny 不可用时使用) ----
diffusion_map_manual <- function(X, ndim = 2) {
  # 标准 Diffusion Map 算法:
  #   1. 欧氏距离矩阵 → 2. 高斯核 → 3. 归一化 → 4. 特征分解
  X <- as.matrix(X)
  d <- as.matrix(dist(X))
  sigma <- median(d[upper.tri(d)])
  K <- exp(-d^2 / (2 * sigma^2))
  # 归一化: D^{-1} K → Markov 转移矩阵 P
  row_sum <- rowSums(K)
  P <- sweep(K, 1, row_sum, "/")
  # 特征分解, 取第 2 ~ ndim+1 个特征向量 (第1个是平凡解)
  eig <- eigen(P, symmetric = FALSE)
  ev <- Re(eig$vectors[, 2:(ndim + 1), drop = FALSE])
  colnames(ev) <- paste0("DC", seq_len(ndim))
  structure(list(eigenvectors = ev, n_components = ndim), class = "dm_manual")
}

if (!USE_DESTINY) cat("注意: 使用手动 Diffusion Map 实现\n")

set.seed(42)

# ============================================================
# 1. 加载 BRCA 训练数据
# ============================================================
cat("=== 1. 加载 BRCA 数据 ===\n")
load("D:/Mysoftware/R/data/BRCA1.RData")     # mydatGE, mydatME, mydatMI, survival
labels_df <- read.csv("BRCA/brca_k5_labels.csv")

stopifnot(nrow(mydatGE) == 628, nrow(mydatME) == 628,
          nrow(mydatMI) == 628, nrow(labels_df) == 628)

y <- factor(labels_df$cluster)                # 5 类标签 0-4
cat(sprintf("样本: %d  类别: %d  分布: %s\n",
            length(y), length(levels(y)), paste(table(y), collapse = ", ")))

# ============================================================
# 2. 加载 METABRIC 外部验证数据 + 基因对齐
# ============================================================
cat("\n=== 2. 加载 METABRIC 外部数据 + 基因对齐 ===\n")

# 外部 GE 表达矩阵
load("metabric_exp.RData")                    # metabric_exp: 1980 × 23008
# 外部临床数据 (含 PAM50 标签)
clin_all <- read.csv("brca_clin_all.csv", stringsAsFactors = FALSE)
clin_meta <- subset(clin_all, data_source == "metabric")
rownames(clin_meta) <- clin_meta$sample_ID

# ---- 基因对齐: 取 BRCA GE 和 METABRIC GE 的共同基因 ----
genes_brca  <- colnames(mydatGE)
genes_metab <- colnames(metabric_exp)
common_genes <- intersect(genes_brca, genes_metab)
cat(sprintf("BRCA GE: %d 基因  METABRIC: %d 基因  共同: %d 基因\n",
            length(genes_brca), length(genes_metab), length(common_genes)))

# 两个 GE 矩阵都取共同基因子集
ge_brca_train <- mydatGE[, common_genes, drop = FALSE]       # 628 × 13897
ge_metab_ext  <- metabric_exp[, common_genes, drop = FALSE]  # 1980 × 13897

# ---- 从 METABRIC 中随机取 500 样本作为外部测试集 ----
n_ext <- 500
idx_ext <- sample(seq_len(nrow(ge_metab_ext)), n_ext)
X_new_ge_raw <- ge_metab_ext[idx_ext, common_genes, drop = FALSE]

# 获取这些样本的 PAM50 标签 (作为 ground truth)
ext_ids <- rownames(X_new_ge_raw)
y_pam50 <- clin_meta[ext_ids, "pam50"]
# 去除 NC (Not Classified) 样本
keep_idx <- y_pam50 != "NC" & !is.na(y_pam50)
X_new_ge_raw <- X_new_ge_raw[keep_idx, , drop = FALSE]
y_pam50 <- factor(y_pam50[keep_idx])
cat(sprintf("外部验证样本: %d (去除 NC 后)  PAM50 分布: %s\n",
            length(y_pam50), paste(table(y_pam50), collapse = ", ")))

# ============================================================
# 3. Diffusion Map 降维 (各组学独立, 取前 2 个 DC)
#    GE 用共同基因子集
# ============================================================
cat("\n=== 3. Diffusion Map 降维 ===\n")

dm_reduce <- function(X, ndim = 2, label = "") {
  m <- as.matrix(X)
  if (USE_DESTINY) {
    dm <- DiffusionMap(m, distance = "euclidean", suppress_dpt = TRUE)
    ev <- dm@eigenvectors[, seq_len(ndim), drop = FALSE]
  } else {
    dm <- diffusion_map_manual(m, ndim)
    ev <- dm$eigenvectors
  }
  colnames(ev) <- paste0(label, "_DC", seq_len(ndim))
  cat(sprintf("  %-5s %d x %d  →  %d DCs\n", label, nrow(m), ncol(m), ndim))
  ev
}

# BRCA 训练数据的 DM (GE 使用共同基因)
ge_dm <- dm_reduce(ge_brca_train, 2, "GE")
mi_dm <- dm_reduce(mydatMI,      2, "MI")
me_dm <- dm_reduce(mydatME,      2, "ME")

# 保存完整 GE DM 模型 (用于后续 Nyström 投影)
dm_ge_full <- if (USE_DESTINY) {
  DiffusionMap(as.matrix(ge_brca_train), distance = "euclidean",
               suppress_dpt = TRUE)
} else {
  diffusion_map_manual(as.matrix(ge_brca_train), 2)
}

# ============================================================
# 4. 7 种特征组合
# ============================================================
cat("\n=== 4. 特征组合 ===\n")

combos <- list(
  GE         = ge_dm,
  MI         = mi_dm,
  ME         = me_dm,
  `GE+MI`    = cbind(ge_dm, mi_dm),
  `GE+ME`    = cbind(ge_dm, me_dm),
  `MI+ME`    = cbind(mi_dm, me_dm),
  `GE+MI+ME` = cbind(ge_dm, mi_dm, me_dm)
)
for (nm in names(combos))
  cat(sprintf("  %-10s %d 特征\n", nm, ncol(combos[[nm]])))

# ============================================================
# 5. 训练集 / 测试集 划分
# ============================================================
n_total    <- nrow(ge_dm)
idx_train  <- sort(sample(seq_len(n_total), 400))
idx_test   <- setdiff(seq_len(n_total), idx_train)
cat(sprintf("\n训练: %d  测试: %d  (总样本: %d)\n",
            length(idx_train), length(idx_test), n_total))

# ============================================================
# 6. 分类器定义 & 主循环
# ============================================================
cat("\n=== 6. 训练 & 评估 ===\n")

my_classifiers <- list(
  RF = function(xtr, ytr, xte) {
    rf <- randomForest(x = xtr, y = ytr, ntree = 500)
    predict(rf, xte)
  },
  SVM = function(xtr, ytr, xte) {
    m <- svm(x = xtr, y = ytr, kernel = "radial", probability = FALSE)
    predict(m, xte)
  },
  XGBoost = function(xtr, ytr, xte) {
    xtr_mat <- matrix(as.numeric(xtr), nrow = nrow(xtr), ncol = ncol(xtr))
    xte_mat <- matrix(as.numeric(xte), nrow = nrow(xte), ncol = ncol(xte))
    y_int <- as.integer(ytr) - 1L
    dtrain <- xgb.DMatrix(xtr_mat, label = y_int)
    dtest  <- xgb.DMatrix(xte_mat)
    par <- list(objective = "multi:softmax", num_class = 5L, max_depth = 4,
                eta = 0.1, subsample = 0.8)
    m <- xgb.train(params = par, data = dtrain, nrounds = 100, verbose = 0)
    factor(levels(ytr)[predict(m, dtest) + 1L], levels = levels(ytr))
  },
  MLP = function(xtr, ytr, xte) {
    m <- nnet(xtr, class.ind(ytr), size = 10, maxit = 300,
              trace = FALSE, linout = TRUE, decay = 0.01)
    factor(levels(ytr)[max.col(predict(m, xte))], levels = levels(ytr))
  },
  KNN = function(xtr, ytr, xte) {
    knn(train = xtr, test = xte, cl = ytr, k = 5)
  }
)

calc_metrics <- function(true, pred) {
  stopifnot(length(true) == length(pred))
  cm <- confusionMatrix(pred, true)
  f1 <- cm$byClass[, "F1"]; f1[is.na(f1)] <- 0
  c(Accuracy     = unname(cm$overall["Accuracy"]),
    F1_macro     = mean(f1),
    F1_weighted  = sum(f1 * as.numeric(table(true))) / length(true))
}

results <- list()

for (cn in names(combos)) {
  X <- as.data.frame(combos[[cn]])
  xs <- scale(X)
  xtr <- xs[idx_train, , drop = FALSE]
  xte <- xs[idx_test,  , drop = FALSE]
  ytr <- y[idx_train]
  yte <- y[idx_test]

  cat(sprintf("\n[%s] %d 特征\n", cn, ncol(xtr)))

  for (clf in names(my_classifiers)) {
    cat(sprintf("  %-7s ", clf))
    flush.console()
    pred <- my_classifiers[[clf]](xtr, ytr, xte)
    pred <- factor(pred, levels = levels(yte))
    m <- calc_metrics(yte, pred)
    results[[length(results) + 1L]] <- data.frame(
      Combination = cn, Classifier = clf,
      Accuracy = m["Accuracy"], F1_macro = m["F1_macro"],
      F1_weighted = m["F1_weighted"],
      stringsAsFactors = FALSE, row.names = NULL
    )
    cat(sprintf("Acc=%.3f  F1m=%.3f  F1w=%.3f\n",
                m["Accuracy"], m["F1_macro"], m["F1_weighted"]))
  }
}

res <- do.call(rbind, results)

# ============================================================
# 7. 分组柱状图
# ============================================================
cat("\n=== 7. 绘图 ===\n")

res$Combination <- factor(res$Combination,
  levels = c("GE","MI","ME","GE+MI","GE+ME","MI+ME","GE+MI+ME"))
res$Classifier <- factor(res$Classifier,
  levels = c("RF","SVM","XGBoost","MLP","KNN"))

# 暖色系配色 (不与生存曲线重叠: 生存曲线用红/蓝/绿/深蓝/橙)
clr <- c("#8B5CF6","#F59E0B","#06B6D4","#84CC16","#EC4899","#78716C","#14B8A6")

make_plot <- function(metric_col, metric_label) {
  p <- ggplot(res, aes(x = Classifier, y = .data[[metric_col]], fill = Combination)) +
    geom_col(position = position_dodge(0.85), width = 0.7,
             color = "white", linewidth = 0.2) +
    scale_fill_manual(values = clr) +
    labs(title = paste0("BRCA ", metric_label, " 对比"),
         subtitle = "Diffusion Map 2 DCs x 7 组合 x 5 分类器",
         x = NULL, y = metric_label, fill = "特征组合") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14),
          plot.subtitle = element_text(size = 9, color = "grey40"),
          panel.grid.major.x = element_blank()) +
    ylim(0, 1)
  ggsave(paste0("classifier_", metric_col, ".png"), p, width = 12, height = 6, dpi = 300)
  p
}

p1 <- make_plot("Accuracy",    "Accuracy")
p2 <- make_plot("F1_macro",    "F1-macro")
p3 <- make_plot("F1_weighted", "F1-weighted")

png("classifier_all_metrics.png", width = 12, height = 16, units = "in", res = 300)
gridExtra::grid.arrange(p1, p2, p3, ncol = 1)
dev.off()

cat("\n=== 结果汇总 ===\n")
print(res, digits = 3)
write.csv(res, "classifier_results.csv", row.names = FALSE, fileEncoding = "UTF-8")

# ============================================================
# 8. 外部验证: METABRIC GE 数据 → RF 分类 → PAM50 对比
# ============================================================
cat("\n=== 8. 外部验证: METABRIC GE → RF 预测 ===\n")

# ---- 8a. 在原 BRCA 全部 628 样本上训练 GE-only RF ----
X_ge <- as.data.frame(ge_dm)
X_ge_s <- scale(X_ge)
rf_ge_full <- randomForest(x = X_ge_s, y = y, ntree = 500)
cat(sprintf("GE-only RF (BRCA 628 样本) OOB 误差: %.3f\n",
            rf_ge_full$err.rate[500, "OOB"]))

# ---- 8b. Nyström 投影: METABRIC GE → BRCA GE 的 DM 空间 ----
cat("\n--- Nyström 特征对齐 ---\n")

nystrom_project <- function(X_orig, X_new, dm_obj, ndim = 2) {
  # X_orig: 原始 BRCA GE (628 × common_genes)
  # X_new:  外部 METABRIC GE (m × common_genes)
  # dm_obj: dm_ge_full 对象
  X_orig <- as.matrix(X_orig)
  X_new  <- as.matrix(X_new)

  # 高斯核带宽 = 原数据 median pairwise distance
  d_orig <- as.matrix(dist(X_orig))
  sigma  <- median(d_orig[upper.tri(d_orig)])

  # 新-原 交叉高斯核
  d_cross <- as.matrix(pdist(X_new, X_orig))
  K_cross <- exp(-d_cross^2 / (2 * sigma^2))

  # 原数据的核矩阵 & 归一化权重
  K_orig  <- exp(-d_orig^2 / (2 * sigma^2))
  row_sum <- rowSums(K_orig)

  # Markov 转移 → 投影到原特征向量
  P_cross <- sweep(K_cross, 2, row_sum, "/")
  ev <- if (USE_DESTINY) {
    dm_obj@eigenvectors[, seq_len(ndim), drop = FALSE]
  } else {
    dm_obj$eigenvectors[, seq_len(ndim), drop = FALSE]
  }
  proj <- P_cross %*% ev
  colnames(proj) <- paste0("GE_DC", seq_len(ndim))
  proj
}

X_new_dm <- nystrom_project(ge_brca_train, X_new_ge_raw, dm_ge_full, ndim = 2)
cat(sprintf("Nyström 投影完成: %d x %d\n", nrow(X_new_dm), ncol(X_new_dm)))

# ---- 8c. Z-score 标准化 (用 BRCA 训练集的 mean/sd) ----
X_new_dm <- as.data.frame(X_new_dm)
X_new_s  <- scale(X_new_dm,
                  center = attr(X_ge_s, "scaled:center"),
                  scale  = attr(X_ge_s, "scaled:scale"))

# ---- 8d. RF 预测 ----
pred_new <- predict(rf_ge_full, X_new_s)
prob_new <- predict(rf_ge_full, X_new_s, type = "prob")

cat(sprintf("\n预测完成: %d 样本\n", length(pred_new)))
cat("预测类别分布 (BRCA cluster 标签):\n")
print(table(pred_new))

# ---- 8e. 与 PAM50 真实标签对比 ----
cat("\n--- PAM50 标签对比 ---\n")

# PAM50 是字符串标签，BRCA 聚类标签是 0-4 数值
# 无法直接计算 Accuracy (标签体系不同)，但可以输出交叉表看一致性
cat("预测标签 vs PAM50 交叉表:\n")
cross_tab <- table(Predicted = pred_new, PAM50 = y_pam50)
print(cross_tab)

# 计算 Adjusted Rand Index (ARI) 衡量两种标签的一致性
ari <- mclust::adjustedRandIndex(as.integer(pred_new), as.integer(y_pam50))
cat(sprintf("\nAdjusted Rand Index (预测 vs PAM50): %.4f\n", ari))
cat("(ARI 越接近 1 表示两种分类越一致, 越接近 0 表示随机)\n")

# ---- 8f. 导出预测结果 ----
out <- data.frame(
  sample_id       = rownames(X_new_ge_raw),
  predicted_cluster = pred_new,
  PAM50_true       = y_pam50,
  prob_new,
  stringsAsFactors = FALSE
)
write.csv(out, "external_validation_predictions.csv",
          row.names = FALSE, fileEncoding = "UTF-8")

cat("\n预测结果已导出到 external_validation_predictions.csv\n")

# ---- 8g. 生存曲线: 基于预测标签绘制 METABRIC 样本的 KM 曲线 ----
cat("\n--- 生存曲线 (基于预测标签) ---\n")

if (!requireNamespace("survminer", quietly = TRUE))
  install.packages("survminer")
library(survival)
library(survminer)

# 取生存数据
surv_meta <- clin_meta[rownames(X_new_ge_raw), ]
time <- as.numeric(surv_meta$dfs_month)
event <- as.numeric(surv_meta$dfs_event)
# 截尾 > 240 月
event[time > 240 & event == 1] <- 0
time[time > 240] <- 240

surv_obj <- Surv(time, event)
pred_group <- pred_new

n_groups <- length(unique(pred_group))
if (n_groups > 1) {
  km_fit <- survfit(surv_obj ~ pred_group)
  p_km <- ggsurvplot(km_fit,
    data = data.frame(time, event, pred_group),
    palette = c("#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"),
    legend.title = "Predicted Cluster",
    legend.labs = paste("Cluster", sort(unique(pred_group))),
    xlab = "DFS (Months)", ylab = "Survival Probability",
    pval = TRUE, risk.table = TRUE)
  ggsave("external_KM_predicted.png", p_km$plot,
         width = 8, height = 6, dpi = 300)
} else {
  cat(sprintf("警告: 所有样本预测为同一类别 (cluster %s), 无法绘制分组 KM 曲线\n",
              unique(pred_group)))
}

# ============================================================
# 9. 实例说明
# ============================================================
cat("\n=== 9. 实例说明 ===\n")
cat("
外部验证流程:
  输入:  METABRIC GE 矩阵 (~500 样本 x 23008 基因)
  步骤1: 基因对齐 → 取共同基因 (13897 个)
  步骤2: Nyström 投影 → 映射到 BRCA Diffusion Map 空间 (2 DCs)
  步骤3: Z-score 标准化 (使用 BRCA 训练集的 mean/sd)
  步骤4: GE-only RF 预测 → 输出亚型标签 + 概率
  步骤5: 与 PAM50 金标准标签对比 (交叉表 + ARI)
  步骤6: 基于预测标签绘制 KM 生存曲线
\n")

cat("前 10 个预测示例:\n")
print(head(out, 10))

cat("\n=== 脚本执行完毕 ===\n")
