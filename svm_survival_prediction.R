# ============================================================
# BRCA & CRC SVM 预测生存曲线
# BRCA: metabric_exp.RData + brca_clin_all.csv (PAM50标签)
# CRC:  CRC_GSE39582.raw.RData + crc_k4_labels.csv (IntegrAO K4标签)
# 分类器: SVM
# ============================================================

rm(list = ls())

options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

cran_pkgs <- c("e1071", "randomForest", "ggplot2", "reshape2", "survival", "survminer",
               "caret")
for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

set.seed(42)

# ============================================================
# 配色定义
# ============================================================
SURV_COLORS_BRCA <- c("#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F")
SURV_COLORS_CRC  <- c("#F9C74F", "#577590", "#F3722C", "#6A994E")

# CRC classifier can be switched to "SVM", "RF", or "XGBoost".
# RF is the default here because it is usually more stable for small-n, high-p omics data.
CRC_CLASSIFIER <- "RF"

predict_crc_classifier <- function(classifier, xtr, ytr, xte) {
  if (classifier == "SVM") {
    model <- svm(x = xtr, y = ytr, kernel = "radial", probability = TRUE)
    return(predict(model, xte))
  }

  if (classifier == "RF") {
    model <- randomForest(x = xtr, y = ytr, ntree = 500, importance = TRUE)
    return(predict(model, xte))
  }

  if (classifier == "XGBoost") {
    if (!requireNamespace("xgboost", quietly = TRUE)) install.packages("xgboost")
    library(xgboost)
    xtr_mat <- as.matrix(xtr)
    xte_mat <- as.matrix(xte)
    dtrain <- xgb.DMatrix(xtr_mat, label = as.integer(ytr) - 1L)
    dtest <- xgb.DMatrix(xte_mat)
    params <- list(
      objective = "multi:softmax",
      num_class = length(levels(ytr)),
      max_depth = 3,
      eta = 0.05,
      subsample = 0.8,
      colsample_bytree = 0.8,
      eval_metric = "mlogloss"
    )
    model <- xgb.train(params = params, data = dtrain, nrounds = 200, verbose = 0)
    return(factor(levels(ytr)[predict(model, dtest) + 1L], levels = levels(ytr)))
  }

  stop("Unsupported CRC_CLASSIFIER: ", classifier)
}

crc_caret_method <- function(classifier) {
  switch(classifier,
         SVM = "svmRadial",
         RF = "rf",
         XGBoost = "xgbTree",
         stop("Unsupported CRC_CLASSIFIER: ", classifier))
}

logrank_p <- function(time, event, group) {
  lr <- survdiff(Surv(time, event) ~ group)
  pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)
}

# ============================================================
# PART 1: BRCA — METABRIC PAM50 标签 + SVM 生存曲线
# ============================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("PART 1: BRCA (METABRIC) — PAM50 标签 + SVM\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# --- 1a. 加载数据 ---
cat("\n[1a] 加载数据\n")
load("metabric_exp.RData")                           # metabric_exp
clin_all <- read.csv("brca_clin_all.csv", stringsAsFactors = FALSE)
clin_meta <- subset(clin_all, data_source == "metabric")
rownames(clin_meta) <- clin_meta$sample_ID

# --- 1b. 样本筛选: 有 PAM50 标签 + 有生存数据 ---
cat("\n[1b] 样本筛选 (PAM50 + 生存数据)\n")
clin_meta$pam50_valid <- clin_meta$pam50 != "NC" & !is.na(clin_meta$pam50)
clin_meta$surv_valid  <- !is.na(clin_meta$dfs_month) & !is.na(clin_meta$dfs_event)

common_ids <- intersect(rownames(metabric_exp), rownames(clin_meta))
keep_ids <- common_ids[clin_meta[common_ids, "pam50_valid"] &
                       clin_meta[common_ids, "surv_valid"]]
cat(sprintf("有效样本: %d (PAM50 + 生存数据齐全)\n", length(keep_ids)))

# PAM50 标签
y_brca <- factor(clin_meta[keep_ids, "pam50"])
cat(sprintf("PAM50 分布: %s\n", paste(paste(levels(y_brca), table(y_brca), sep = "="), collapse = ", ")))

# --- 1c. 表达矩阵: top 2000 变量基因 ---
cat("\n[1c] 特征筛选 (top 2000 变量基因)\n")
ge_brca <- as.matrix(metabric_exp[keep_ids, ])
variances <- apply(ge_brca, 2, var, na.rm = TRUE)
top_genes <- names(sort(variances, decreasing = TRUE)[1:min(2000, length(variances))])
X_brca <- scale(ge_brca[, top_genes])
cat(sprintf("表达矩阵: %d x %d\n", nrow(X_brca), ncol(X_brca)))

# --- 1d. 训练/测试划分 ---
cat("\n[1d] 训练/测试划分 (70/30)\n")
n_brca <- nrow(X_brca)
idx_train <- sort(sample(seq_len(n_brca), round(n_brca * 0.7)))
idx_test  <- setdiff(seq_len(n_brca), idx_train)
cat(sprintf("训练: %d  测试: %d\n", length(idx_train), length(idx_test)))

# --- 1e. SVM 训练 ---
cat("\n[1e] SVM 训练\n")
svm_brca <- svm(x = X_brca[idx_train, ], y = y_brca[idx_train],
                kernel = "radial", probability = TRUE)
pred_train <- predict(svm_brca, X_brca[idx_train, ])
cat(sprintf("训练集 Accuracy: %.4f\n", mean(pred_train == y_brca[idx_train])))

# --- 1f. 预测测试集 ---
cat("\n[1f] 预测测试集\n")
pred_brca <- predict(svm_brca, X_brca[idx_test, ])
pred_brca <- factor(pred_brca, levels = levels(y_brca))
y_test_brca <- y_brca[idx_test]
acc_brca <- mean(pred_brca == y_test_brca)
cat(sprintf("测试集 Accuracy: %.4f\n", acc_brca))
cat("混淆矩阵:\n")
print(table(Predicted = pred_brca, Actual = y_test_brca))

# --- 1g. KM 生存曲线 ---
cat("\n[1g] BRCA SVM 预测生存曲线\n")
time_brca <- as.numeric(clin_meta[keep_ids[idx_test], "dfs_month"])
event_brca <- as.numeric(clin_meta[keep_ids[idx_test], "dfs_event"])
event_brca[time_brca > 240 & event_brca == 1] <- 0
time_brca[time_brca > 240] <- 240

valid <- !is.na(time_brca) & !is.na(event_brca)
time_brca <- time_brca[valid]
event_brca <- event_brca[valid]
pred_brca_surv <- pred_brca[valid]

n_groups <- length(unique(pred_brca_surv))
fit_brca <- survfit(Surv(time_brca, event_brca) ~ pred_brca_surv)
p_brca <- ggsurvplot(fit_brca,
  data = data.frame(time_brca, event_brca, pred_brca_surv),
  palette = SURV_COLORS_BRCA[1:n_groups],
  legend.title = "SVM Predicted (PAM50)",
  legend.labs = paste(levels(pred_brca_surv)),
  xlab = "DFS (Months)", ylab = "Survival Probability",
  title = "BRCA (METABRIC) — SVM Predicted Survival (PAM50 subtypes)",
  pval = TRUE, risk.table = TRUE)
ggsave("BRCA/BRCA_SVM_KM_predicted.png", p_brca$plot,
       width = 10, height = 7, dpi = 300)
ggsave("BRCA/BRCA_SVM_KM_predicted.pdf", p_brca$plot,
       width = 10, height = 7, dpi = 300)
cat("  -> BRCA/BRCA_SVM_KM_predicted.png/pdf 已保存\n")

# ============================================================
# PART 2: CRC — IntegrAO K4 标签 + SVM 生存曲线
# ============================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat(sprintf("PART 2: CRC — IntegrAO K4 标签 + %s\n", CRC_CLASSIFIER))
cat(paste(rep("=", 60), collapse = ""), "\n")

# --- 2a. 加载数据 ---
cat("\n[2a] 加载 CRC 数据\n")
load("CRC_GSE39582.raw.RData")       # GSE39582_ge.syms, GSE39582_info

crc_labels <- read.csv("CRC/crc_k4_labels.csv")
crc_surv   <- read.csv("CRC/crc_survival.csv")
crc_mrna   <- read.csv("CRC/crc_mRNA.csv")
colnames(crc_surv) <- c("CMS", "time", "status")
y_crc <- factor(crc_labels$cluster)
cat(sprintf("CRC: %d 样本, K4 分布: %s\n",
            length(y_crc), paste(table(y_crc), collapse = ", ")))

# --- 2b. 特征筛选 + 训练/测试划分 ---
cat("\n[2b] 特征筛选 (top 2000)\n")
variances <- apply(crc_mrna, 2, var, na.rm = TRUE)
top_genes <- names(sort(variances, decreasing = TRUE)[1:min(2000, ncol(crc_mrna))])
X_crc <- scale(as.matrix(crc_mrna[, top_genes]))
cat(sprintf("表达矩阵: %d x %d\n", nrow(X_crc), ncol(X_crc)))

n_crc <- nrow(X_crc)
idx_train_crc <- sort(sample(seq_len(n_crc), round(n_crc * 0.7)))
idx_test_crc  <- setdiff(seq_len(n_crc), idx_train_crc)
cat(sprintf("训练: %d  测试: %d\n", length(idx_train_crc), length(idx_test_crc)))

# --- 2c. 分类器训练 & 预测 ---
cat(sprintf("\n[2c] %s 训练 & 预测\n", CRC_CLASSIFIER))
pred_crc <- predict_crc_classifier(
  CRC_CLASSIFIER,
  X_crc[idx_train_crc, ],
  y_crc[idx_train_crc],
  X_crc[idx_test_crc, ]
)
pred_crc <- factor(pred_crc, levels = levels(y_crc))
y_test_crc <- y_crc[idx_test_crc]
acc_crc <- mean(pred_crc == y_test_crc)
cm <- confusionMatrix(pred_crc, y_test_crc)
cat(sprintf("测试集 Accuracy: %.4f  F1-macro: %.4f\n",
            acc_crc, mean(cm$byClass[, "F1"], na.rm = TRUE)))
cat("混淆矩阵:\n")
print(table(Predicted = pred_crc, Actual = y_test_crc))

# --- 2d. KM 生存曲线 ---
cat(sprintf("\n[2d] CRC %s 预测生存曲线\n", CRC_CLASSIFIER))
time_crc <- as.numeric(crc_surv$time[idx_test_crc]) / 30.44
event_crc <- as.numeric(crc_surv$status[idx_test_crc])
valid <- !is.na(time_crc) & !is.na(event_crc)
time_crc <- time_crc[valid]
event_crc <- event_crc[valid]
pred_crc_surv <- pred_crc[valid]
crc_test_p <- logrank_p(time_crc, event_crc, pred_crc_surv)
cat(sprintf("测试集 log-rank p = %.4g\n", crc_test_p))

fit_crc <- survfit(Surv(time_crc, event_crc) ~ pred_crc_surv)
p_crc <- ggsurvplot(fit_crc,
  data = data.frame(time_crc, event_crc, pred_crc_surv),
  palette = SURV_COLORS_CRC,
  legend.title = paste(CRC_CLASSIFIER, "Predicted Cluster"),
  legend.labs = paste("Cluster", levels(pred_crc_surv)),
  xlab = "Time (Months)", ylab = "Survival Probability",
  title = paste0("CRC — ", CRC_CLASSIFIER, " Predicted Survival (IntegrAO K=4)"),
  pval = TRUE, risk.table = TRUE)
crc_test_png <- paste0("CRC/CRC_", CRC_CLASSIFIER, "_KM_predicted.png")
crc_test_pdf <- paste0("CRC/CRC_", CRC_CLASSIFIER, "_KM_predicted.pdf")
ggsave(crc_test_png, p_crc$plot,
       width = 10, height = 7, dpi = 300)
ggsave(crc_test_pdf, p_crc$plot,
       width = 10, height = 7, dpi = 300)
cat(sprintf("  -> %s/pdf 已保存\n", sub("\\.png$", ".png", crc_test_png)))

# --- 2e. 全样本 5-fold CV ---
cat("\n[2e] 全样本 5-fold CV + KM\n")
ctrl <- trainControl(method = "cv", number = 5, savePredictions = "final")
cv_model <- train(x = X_crc, y = y_crc, method = crc_caret_method(CRC_CLASSIFIER),
                  trControl = ctrl, trace = FALSE)
pred_crc_cv <- cv_model$pred[order(cv_model$pred$rowIndex), "pred"]
pred_crc_cv <- factor(pred_crc_cv, levels = levels(y_crc))

cat(sprintf("5-fold CV Accuracy: %.4f\n", mean(pred_crc_cv == y_crc)))

time_crc_all <- as.numeric(crc_surv$time) / 30.44
event_crc_all <- as.numeric(crc_surv$status)
valid <- !is.na(time_crc_all) & !is.na(event_crc_all)
time_crc_all <- time_crc_all[valid]
event_crc_all <- event_crc_all[valid]
pred_crc_cv_surv <- pred_crc_cv[valid]
crc_cv_p <- logrank_p(time_crc_all, event_crc_all, pred_crc_cv_surv)
cat(sprintf("5-fold CV 全样本 log-rank p = %.4g\n", crc_cv_p))

fit_crc_cv <- survfit(Surv(time_crc_all, event_crc_all) ~ pred_crc_cv_surv)
p_crc_cv <- ggsurvplot(fit_crc_cv,
  data = data.frame(time_crc_all, event_crc_all, pred_crc_cv_surv),
  palette = SURV_COLORS_CRC,
  legend.title = paste(CRC_CLASSIFIER, "CV Cluster"),
  legend.labs = paste("Cluster", levels(pred_crc_cv_surv)),
  xlab = "Time (Months)", ylab = "Survival Probability",
  title = paste0("CRC — ", CRC_CLASSIFIER, " 5-fold CV Predicted Survival (K=4, n=297)"),
  pval = TRUE, risk.table = TRUE)
crc_cv_png <- paste0("CRC/CRC_", CRC_CLASSIFIER, "_KM_predicted_all.png")
crc_cv_pdf <- paste0("CRC/CRC_", CRC_CLASSIFIER, "_KM_predicted_all.pdf")
ggsave(crc_cv_png, p_crc_cv$plot,
       width = 10, height = 7, dpi = 300)
ggsave(crc_cv_pdf, p_crc_cv$plot,
       width = 10, height = 7, dpi = 300)
cat(sprintf("  -> %s/pdf 已保存\n", sub("\\.png$", ".png", crc_cv_png)))

# ============================================================
# 汇总
# ============================================================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("汇总\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat(sprintf("\nBRCA (METABRIC + PAM50, n=%d):\n", n_brca))
cat(sprintf("  测试集 Accuracy: %.4f\n", acc_brca))
cat(sprintf("  生存曲线: BRCA/BRCA_SVM_KM_predicted.png\n"))

cat(sprintf("\nCRC (IntegrAO K4, n=%d):\n", n_crc))
cat(sprintf("  测试集 Accuracy: %.4f  |  CV Accuracy: %.4f\n",
            acc_crc, mean(pred_crc_cv == y_crc)))
cat(sprintf("  测试集 log-rank p: %.4g  |  CV全样本 log-rank p: %.4g\n",
            crc_test_p, crc_cv_p))
cat(sprintf("  生存曲线 (测试集): %s\n", crc_test_png))
cat(sprintf("  生存曲线 (CV全样本): %s\n", crc_cv_png))

cat("\n===== 完成 =====\n")
