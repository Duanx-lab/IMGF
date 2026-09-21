# ============================================================
# CRC RF classifier survival curve on GSE39582 raw data
# Data: D:/integrAO/vs/CRC_GSE39582.raw.RData
# Classifier labels: GSE39582_Rfcms$RF.predictedCMS
# Survival: GSE39582_info$os.delay + GSE39582_info$os.event
# ============================================================

setwd("D:/integrAO/vs")
rm(list = ls())

options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

cran_pkgs <- c("survival", "survminer", "ggplot2")
for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

set.seed(42)

RAW_FILE <- "D:/integrAO/vs/CRC_GSE39582.raw.RData"
OUT_PNG <- "CRC/CRC_RF_GSE39582_OS_KM.png"
OUT_PDF <- "CRC/CRC_RF_GSE39582_OS_KM.pdf"
OUT_CSV <- "CRC/CRC_RF_GSE39582_predictions_survival.csv"

SURV_COLORS_CRC <- c(
  CMS1 = "#F9C74F",
  CMS2 = "#577590",
  CMS3 = "#F3722C",
  CMS4 = "#6A994E"
)

logrank_p <- function(time, event, group) {
  lr <- survdiff(Surv(time, event) ~ group)
  pchisq(lr$chisq, df = length(lr$n) - 1, lower.tail = FALSE)
}

cat("=== Load GSE39582 raw data ===\n")
load(RAW_FILE)

stopifnot(exists("GSE39582_info"))
stopifnot(exists("GSE39582_Rfcms"))

common_ids <- intersect(rownames(GSE39582_info), rownames(GSE39582_Rfcms))
info <- GSE39582_info[common_ids, , drop = FALSE]
rf <- GSE39582_Rfcms[common_ids, , drop = FALSE]

plot_df <- data.frame(
  sample_id = common_ids,
  RF_predicted_CMS = rf$RF.predictedCMS,
  RF_nearest_CMS = rf$RF.nearestCMS,
  os_time_month = as.numeric(info$os.delay),
  os_event = as.numeric(info$os.event),
  stringsAsFactors = FALSE
)

prob_cols <- grep("posteriorProb", colnames(rf), value = TRUE)
plot_df <- cbind(plot_df, rf[, prob_cols, drop = FALSE])

valid <- !is.na(plot_df$RF_predicted_CMS) &
  !is.na(plot_df$os_time_month) &
  !is.na(plot_df$os_event)
plot_df <- plot_df[valid, , drop = FALSE]
plot_df$RF_predicted_CMS <- factor(
  plot_df$RF_predicted_CMS,
  levels = c("CMS1", "CMS2", "CMS3", "CMS4")
)

cat(sprintf("Valid samples: %d\n", nrow(plot_df)))
cat("RF predicted CMS distribution:\n")
print(table(plot_df$RF_predicted_CMS))

p_value <- logrank_p(
  plot_df$os_time_month,
  plot_df$os_event,
  plot_df$RF_predicted_CMS
)
cat(sprintf("OS log-rank p = %.4g\n", p_value))

fit <- survfit(Surv(os_time_month, os_event) ~ RF_predicted_CMS, data = plot_df)

p <- ggsurvplot(
  fit,
  data = plot_df,
  palette = unname(SURV_COLORS_CRC),
  legend.title = "RF Predicted CMS",
  legend.labs = levels(plot_df$RF_predicted_CMS),
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  title = "CRC GSE39582 - RF Predicted CMS Survival",
  pval = TRUE,
  risk.table = TRUE,
  risk.table.height = 0.25,
  break.time.by = 24
)

png(OUT_PNG, width = 10, height = 8, units = "in", res = 300)
print(p)
dev.off()

pdf(OUT_PDF, width = 10, height = 8)
print(p)
dev.off()
write.csv(plot_df, OUT_CSV, row.names = FALSE, fileEncoding = "UTF-8")

cat("Saved:\n")
cat(sprintf("  %s\n", OUT_PNG))
cat(sprintf("  %s\n", OUT_PDF))
cat(sprintf("  %s\n", OUT_CSV))
cat("\n===== Done =====\n")
