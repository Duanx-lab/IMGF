# ============================================================
# CRC multi-omics classifier comparison
# CRC.RData: mydatGE + mydatMI + mydatME, IntegrAO K=4 labels
# Diffusion Map -> first 2 DCs per omics -> 7 omics combinations
# Classifiers: RF, SVM, XGBoost, MLP, KNN
# Metrics: Accuracy, F1-macro, F1-weighted
# ============================================================

rm(list = ls())

options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

cran_pkgs <- c(
  "e1071", "randomForest", "xgboost", "nnet", "class",
  "ggplot2", "reshape2", "gridExtra"
)
for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

set.seed(42)

CRC_RDATA <- "D:/Mysoftware/R/data/CRC.RData"
LABEL_FILE <- "CRC/crc_k4_labels.csv"
TOP_FEATURES <- 2000
N_DM <- 2

# CRC has 297 samples, so 400 training samples is not possible.
# Use 70% train / 30% test by default.
TRAIN_N_REQUESTED <- 400

# ============================================================
# 1. Load data and labels
# ============================================================
cat("=== 1. Load CRC data ===\n")
load(CRC_RDATA)  # mydatGE, mydatMI, mydatME, survival
labels_df <- read.csv(LABEL_FILE, stringsAsFactors = FALSE)

stopifnot(nrow(mydatGE) == nrow(mydatMI), nrow(mydatGE) == nrow(mydatME))
stopifnot(nrow(labels_df) == nrow(mydatGE))

y <- factor(labels_df$cluster)
n_total <- length(y)
cat(sprintf(
  "CRC samples: %d | GE=%d, MI=%d, ME=%d | classes: %s\n",
  n_total, ncol(mydatGE), ncol(mydatMI), ncol(mydatME),
  paste(paste(levels(y), table(y), sep = "="), collapse = ", ")
))

# ============================================================
# 2. Feature filtering and Diffusion Map
# ============================================================
top_var_matrix <- function(X, n_top = 2000, name = "") {
  X <- as.matrix(X)
  vars <- apply(X, 2, var, na.rm = TRUE)
  vars[is.na(vars)] <- 0
  keep <- names(sort(vars, decreasing = TRUE))[seq_len(min(n_top, ncol(X)))]
  out <- X[, keep, drop = FALSE]
  out <- scale(out)
  out[is.na(out)] <- 0
  cat(sprintf("  %-2s top features: %d x %d\n", name, nrow(out), ncol(out)))
  out
}

diffusion_map_manual <- function(X, ndim = 2, label = "") {
  X <- as.matrix(X)
  d <- as.matrix(dist(X))
  sigma <- median(d[upper.tri(d)])
  if (!is.finite(sigma) || sigma <= 0) sigma <- 1
  K <- exp(-d^2 / (2 * sigma^2))
  row_sum <- rowSums(K)
  P <- sweep(K, 1, row_sum, "/")
  eig <- eigen(P, symmetric = FALSE)
  ev <- Re(eig$vectors[, 2:(ndim + 1), drop = FALSE])
  colnames(ev) <- paste0(label, "_DC", seq_len(ndim))
  list(
    embedding = ev,
    train_X = X,
    sigma = sigma,
    eigenvectors = ev,
    label = label
  )
}

cat("\n=== 2. Diffusion Map ===\n")
GE_raw <- top_var_matrix(mydatGE, TOP_FEATURES, "GE")
MI_raw <- top_var_matrix(mydatMI, TOP_FEATURES, "MI")
ME_raw <- top_var_matrix(mydatME, TOP_FEATURES, "ME")

GE_dm_obj <- diffusion_map_manual(GE_raw, N_DM, "GE")
MI_dm_obj <- diffusion_map_manual(MI_raw, N_DM, "MI")
ME_dm_obj <- diffusion_map_manual(ME_raw, N_DM, "ME")

GE_dm <- GE_dm_obj$embedding
MI_dm <- MI_dm_obj$embedding
ME_dm <- ME_dm_obj$embedding

combos <- list(
  GE = GE_dm,
  MI = MI_dm,
  ME = ME_dm,
  `GE+MI` = cbind(GE_dm, MI_dm),
  `GE+ME` = cbind(GE_dm, ME_dm),
  `MI+ME` = cbind(MI_dm, ME_dm),
  `GE+MI+ME` = cbind(GE_dm, MI_dm, ME_dm)
)

for (nm in names(combos)) {
  cat(sprintf("  %-8s features: %d\n", nm, ncol(combos[[nm]])))
}

# ============================================================
# 3. Train/test split
# ============================================================
train_n <- min(TRAIN_N_REQUESTED, floor(n_total * 0.7))
idx_train <- sort(sample(seq_len(n_total), train_n))
idx_test <- setdiff(seq_len(n_total), idx_train)
cat(sprintf("\nTrain: %d | Test: %d\n", length(idx_train), length(idx_test)))

# ============================================================
# 4. Classifiers and metrics
# ============================================================
calc_metrics <- function(true, pred) {
  true <- factor(true)
  pred <- factor(pred, levels = levels(true))
  levs <- levels(true)

  f1 <- numeric(length(levs))
  names(f1) <- levs
  for (lab in levs) {
    tp <- sum(true == lab & pred == lab)
    fp <- sum(true != lab & pred == lab)
    fn <- sum(true == lab & pred != lab)
    precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1[lab] <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  }

  class_weight <- as.numeric(table(true)[levs]) / length(true)
  c(
    Accuracy = mean(true == pred),
    F1_macro = mean(f1),
    F1_weighted = sum(f1 * class_weight)
  )
}

my_classifiers <- list(
  RF = function(xtr, ytr, xte) {
    model <- randomForest(x = xtr, y = ytr, ntree = 500)
    predict(model, xte)
  },
  SVM = function(xtr, ytr, xte) {
    model <- svm(x = xtr, y = ytr, kernel = "radial", probability = FALSE)
    predict(model, xte)
  },
  XGBoost = function(xtr, ytr, xte) {
    dtrain <- xgb.DMatrix(as.matrix(xtr), label = as.integer(ytr) - 1L)
    dtest <- xgb.DMatrix(as.matrix(xte))
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
    factor(levels(ytr)[predict(model, dtest) + 1L], levels = levels(ytr))
  },
  MLP = function(xtr, ytr, xte) {
    model <- nnet(
      xtr, class.ind(ytr),
      size = 8, decay = 0.01, maxit = 500,
      softmax = TRUE, trace = FALSE, MaxNWts = 50000
    )
    factor(levels(ytr)[max.col(predict(model, xte))], levels = levels(ytr))
  },
  KNN = function(xtr, ytr, xte) {
    knn(train = xtr, test = xte, cl = ytr, k = 5)
  }
)

cat("\n=== 3. Train and evaluate classifiers ===\n")
results <- list()
rf_ge_full <- NULL

for (cn in names(combos)) {
  X <- scale(as.matrix(combos[[cn]]))
  X[is.na(X)] <- 0
  xtr <- X[idx_train, , drop = FALSE]
  xte <- X[idx_test, , drop = FALSE]
  ytr <- y[idx_train]
  yte <- y[idx_test]

  cat(sprintf("\n[%s] %d features\n", cn, ncol(X)))

  for (clf in names(my_classifiers)) {
    cat(sprintf("  %-7s ", clf))
    pred <- my_classifiers[[clf]](xtr, ytr, xte)
    pred <- factor(pred, levels = levels(y))
    m <- calc_metrics(yte, pred)
    results[[length(results) + 1L]] <- data.frame(
      Combination = cn,
      Classifier = clf,
      Accuracy = unname(m["Accuracy"]),
      F1_macro = unname(m["F1_macro"]),
      F1_weighted = unname(m["F1_weighted"]),
      stringsAsFactors = FALSE
    )
    cat(sprintf("Acc=%.3f F1_macro=%.3f F1_weighted=%.3f\n",
                m["Accuracy"], m["F1_macro"], m["F1_weighted"]))
  }
}

res <- do.call(rbind, results)
write.csv(res, "CRC/CRC_classifier_results.csv", row.names = FALSE, fileEncoding = "UTF-8")

# ============================================================
# 5. Plot grouped bar charts
# ============================================================
cat("\n=== 4. Plot classifier comparison ===\n")

res$Combination <- factor(
  res$Combination,
  levels = c("GE", "MI", "ME", "GE+MI", "GE+ME", "MI+ME", "GE+MI+ME")
)
res$Classifier <- factor(
  res$Classifier,
  levels = c("RF", "SVM", "XGBoost", "MLP", "KNN")
)

plot_colors <- c(
  "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
  "#76B7B2", "#EDC948", "#B07AA1"
)

make_metric_plot <- function(metric_col, metric_label) {
  p <- ggplot(res, aes(x = Classifier, y = .data[[metric_col]], fill = Combination)) +
    geom_col(position = position_dodge(width = 0.82), width = 0.72,
             color = "white", linewidth = 0.2) +
    scale_fill_manual(values = plot_colors) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title = paste0("CRC ", metric_label, " comparison"),
      x = NULL, y = metric_label, fill = "Feature set"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      panel.grid.major.x = element_blank(),
      legend.position = "bottom"
    )

  ggsave(paste0("CRC/CRC_classifier_", metric_col, ".png"),
         p, width = 12, height = 6, dpi = 300)
  p
}

p1 <- make_metric_plot("Accuracy", "Accuracy")
p2 <- make_metric_plot("F1_macro", "F1-macro")
p3 <- make_metric_plot("F1_weighted", "F1-weighted")

png("CRC/CRC_classifier_all_metrics.png", width = 12, height = 16, units = "in", res = 300)
gridExtra::grid.arrange(p1, p2, p3, ncol = 1)
dev.off()

cat("\nTop results by F1_weighted:\n")
print(res[order(-res$F1_weighted), ][1:10, ], row.names = FALSE)

# ============================================================
# 6. Example: classify 500 new GE-only samples with GE-only RF
# ============================================================
# If you have a real external GE matrix, replace this example with:
# new_GE_raw <- read.csv("new_crc_GE_500.csv", row.names = 1, check.names = FALSE)
# The columns must match the selected GE genes used above.

cat("\n=== 5. GE-only RF example for 500 new samples ===\n")

project_dm_manual <- function(X_new, dm_obj) {
  X_train <- dm_obj$train_X
  common_cols <- intersect(colnames(X_train), colnames(X_new))
  if (length(common_cols) < 2) {
    stop("New GE matrix does not share enough features with the CRC GE training matrix.")
  }
  X_new <- as.matrix(X_new[, common_cols, drop = FALSE])
  X_train <- as.matrix(X_train[, common_cols, drop = FALSE])
  X_new <- scale(X_new)
  X_new[is.na(X_new)] <- 0

  d_cross <- as.matrix(dist(rbind(X_new, X_train)))[
    seq_len(nrow(X_new)),
    nrow(X_new) + seq_len(nrow(X_train)),
    drop = FALSE
  ]
  K_cross <- exp(-d_cross^2 / (2 * dm_obj$sigma^2))
  P_cross <- sweep(K_cross, 1, rowSums(K_cross), "/")
  proj <- P_cross %*% dm_obj$eigenvectors
  colnames(proj) <- colnames(dm_obj$eigenvectors)
  proj
}

# Example only: sample 500 CRC GE profiles with replacement as pseudo-new data.
example_idx <- sample(seq_len(nrow(GE_raw)), 500, replace = TRUE)
new_GE_raw <- GE_raw[example_idx, , drop = FALSE]

GE_train_scaled <- scale(GE_dm)
GE_train_scaled[is.na(GE_train_scaled)] <- 0
rf_ge_full <- randomForest(x = GE_train_scaled, y = y, ntree = 500)

new_GE_dm <- project_dm_manual(new_GE_raw, GE_dm_obj)
new_GE_dm_scaled <- scale(new_GE_dm)
new_GE_dm_scaled[is.na(new_GE_dm_scaled)] <- 0

new_pred <- predict(rf_ge_full, new_GE_dm_scaled)
new_prob <- predict(rf_ge_full, new_GE_dm_scaled, type = "prob")

new_pred_df <- data.frame(
  Sample_ID = paste0("New_", seq_len(length(new_pred))),
  Predicted_Cluster = new_pred,
  new_prob,
  check.names = FALSE
)
write.csv(new_pred_df, "CRC/CRC_RF_GE_new500_predictions_example.csv",
          row.names = FALSE, fileEncoding = "UTF-8")

cat("Saved:\n")
cat("  CRC/CRC_classifier_results.csv\n")
cat("  CRC/CRC_classifier_all_metrics.png\n")
cat("  CRC/CRC_classifier_Accuracy.png\n")
cat("  CRC/CRC_classifier_F1_macro.png\n")
cat("  CRC/CRC_classifier_F1_weighted.png\n")
cat("  CRC/CRC_RF_GE_new500_predictions_example.csv\n")
cat("\n===== Done =====\n")
