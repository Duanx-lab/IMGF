# ============================================================
# Build BRCA classifier all-metrics bar plot
# Input: classifier_results.csv
# Output: BRCA/BRCA_classifier_all_metrics.png/pdf
# ============================================================


options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

cran_pkgs <- c("ggplot2", "gridExtra")
for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

res <- read.csv("classifier_results.csv", stringsAsFactors = FALSE)

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
  ggplot(res, aes(x = Classifier, y = .data[[metric_col]], fill = Combination)) +
    geom_col(
      position = position_dodge(width = 0.82),
      width = 0.72,
      color = "white",
      linewidth = 0.2
    ) +
    scale_fill_manual(values = plot_colors) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title = paste0("BRCA ", metric_label, " comparison"),
      x = NULL,
      y = metric_label,
      fill = "Feature set"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      panel.grid.major.x = element_blank(),
      legend.position = "bottom"
    )
}

p1 <- make_metric_plot("Accuracy", "Accuracy")
p2 <- make_metric_plot("F1_macro", "F1-macro")
p3 <- make_metric_plot("F1_weighted", "F1-weighted")

png("BRCA/BRCA_classifier_all_metrics.png", width = 12, height = 16, units = "in", res = 300)
gridExtra::grid.arrange(p1, p2, p3, ncol = 1)
dev.off()

pdf("BRCA/BRCA_classifier_all_metrics.pdf", width = 12, height = 16)
gridExtra::grid.arrange(p1, p2, p3, ncol = 1)
dev.off()

cat("Saved:\n")
cat("  BRCA/BRCA_classifier_all_metrics.png\n")
cat("  BRCA/BRCA_classifier_all_metrics.pdf\n")
