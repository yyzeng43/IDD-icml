# To use a specific Python interpreter, set the RETICULATE_PYTHON environment
# variable (e.g. export RETICULATE_PYTHON=/path/to/.venv/bin/python) before
# running this script. If unset, reticulate auto-detects an available Python.

suppressPackageStartupMessages({
  library(reticulate)
  library(funcharts)
  library(dplyr)
  library(jsonlite)
  library(ggplot2)
})

`%||%` <- function(a, b) {
  # NULL → use default
  if (is.null(a)) return(b)
  # empty vector → use default
  if (length(a) == 0L) return(b)
  # scalar NA → use default
  if (length(a) == 1L && is.na(a)) return(b)
  # otherwise, use a as-is (can be vector)
  a
}

# ---------- numpy loader (matrix) ----------
np <- reticulate::import("numpy")
load_np_matrix <- function(path) {
  x <- reticulate::py_to_r(np$load(path, allow_pickle = TRUE))
  if (is.numeric(x) && length(dim(x)) == 2L) return(as.matrix(x))
  if (is.numeric(x) && is.null(dim(x)))      return(matrix(x, nrow = 1L))
  if (is.list(x) && all(vapply(x, is.numeric, logical(1)))) {
    return(do.call(rbind, lapply(x, as.numeric)))
  }
  stop("Unsupported .npy at: ", path)
}

# ---------- reshape flattened (n × Gd) -> list of d matrices (n × G) ----------
to_mfd_list <- function(MX, G, d, prefix="coord") {
  stopifnot(ncol(MX) == G * d)
  comps <- vector("list", d)
  for (j in seq_len(d)) {
    idx <- ((j - 1L) * G + 1L):(j * G)
    comps[[j]] <- MX[, idx, drop = FALSE]  # n × G
  }
  names(comps) <- paste0(prefix, seq_len(d))
  comps
}

# ---------- build mfd (grid-based; no pre-smoothing basis) ----------
build_mfd_grid <- function(list_of_mats) {
  funcharts::get_mfd_list(list_of_mats)
}

# ---------- Run length: 1..H if alarm, H+1 if none ----------
run_length <- function(x, UCL, H = length(x)) {
  idx <- which(x[seq_len(H)] > UCL)
  if (length(idx) == 0L) return(H + 1L)
  idx[1L]
}

# ---------- MAIN: single run, no replicates ----------
run_once <- function(base,
                     K = NULL, alpha_overall = 0.05,
                     tot_var = 0.95, single_min = 0.00,
                     limits = c("standard","cv"),
                     out_csv, out_json,
                     out_plot_png = NULL,
                     out_plot_csv = NULL,
                     out_limits_csv = NULL,
                     timing_path = NULL,
                     save_plot_png = TRUE) {

  limits <- match.arg(limits)

  # Expect files: base.Xb.npy, base.PhaseI.npy, base.PhaseII.npy
  Xb0  <- load_np_matrix(paste0(base, ".Xb.npy"))
  TI0  <- load_np_matrix(paste0(base, ".PhaseI.npy"))
  TII0 <- load_np_matrix(paste0(base, ".PhaseII.npy"))

  G <- nrow(Xb0); d <- ncol(Xb0)
  stopifnot(ncol(TI0) == G * d, ncol(TII0) == G * d)

  # ---- PRE-STANDARDIZE on Phase I with epsilon floor ----
  mu  <- colMeans(TI0)
  sdv <- apply(TI0, 2, sd)
  eps <- 1e-8
  sdv[!is.finite(sdv) | sdv < eps] <- eps

  TI0s  <- sweep(sweep(TI0,  2, mu, "-"), 2, sdv, "/")
  TII0s <- sweep(sweep(TII0, 2, mu, "-"), 2, sdv, "/")

  # Build mfd objects
  mfd_train0  <- build_mfd_grid(to_mfd_list(TI0s,  G, d, "coord"))
  mfd_phase20 <- build_mfd_grid(to_mfd_list(TII0s, G, d, "coord"))

  # ==== Time MFPCA fit + control chart construction ====
  t0 <- proc.time()

  # We already standardized → disable internal scaling
  pca0 <- funcharts::pca_mfd(mfd_train0, scale = FALSE)
  comps_arg <- if (!is.null(K)) 1:as.integer(K) else NULL

  cc_I0 <- control_charts_pca(
    pca0, components = comps_arg,
    tuning_data = mfd_train0, newdata = mfd_train0,
    alpha = alpha_overall, limits = limits,
    tot_variance_explained = tot_var,
    single_min_variance_explained = single_min
  )

  cc_II0 <- control_charts_pca(
    pca0, components = comps_arg,
    tuning_data = mfd_train0, newdata = mfd_phase20,
    alpha = alpha_overall, limits = limits,
    tot_variance_explained = tot_var,
    single_min_variance_explained = single_min
  )

  t_fit_mfpca <- as.numeric((proc.time() - t0)[["elapsed"]])
  cat(sprintf("[R][TIMING] mfpca_fit_sec=%.6f\n", t_fit_mfpca))

  if (!is.null(timing_path)) {
    write(
      jsonlite::toJSON(list(mfpca_fit_sec = t_fit_mfpca),
                       pretty = TRUE, auto_unbox = TRUE),
      timing_path
    )
  }

  # ---- Combine Phase I + II for plotting / CSV ----
  cc_all0 <- bind_rows(
    cc_I0 %>% mutate(phase = "I"),
    cc_II0 %>% mutate(phase = "II")
  )

  # Control charts plot (optional)
  if (isTRUE(save_plot_png) && !is.null(out_plot_png)) {
    p <- plot_control_charts(cc_all0, nobsI = nrow(cc_I0))
    ggsave(filename = out_plot_png, plot = p, width = 10, height = 5, dpi = 200)
  }

  # Save full control-chart table (Phase I + II)
  message("[R] Writing CC table to: ", out_csv)
  write.csv(cc_all0, out_csv, row.names = FALSE)

  # ---- Extract fixed UCL/LCL from Phase I ----
  UCL_T2  <- (cc_I0$T2_lim  %||% NA_real_)[1]
  UCL_SPE <- (cc_I0$spe_lim %||% NA_real_)[1]
  LCL_T2  <- (cc_I0$T2_lcl  %||% 0.0)[1]
  LCL_SPE <- (cc_I0$spe_lcl %||% 0.0)[1]

  # ---- Run length on Phase II only (single run) ----
  H0       <- nrow(cc_II0)
  RL_T2    <- run_length(cc_II0$T2,  UCL_T2,  H = H0)
  RL_SPE   <- run_length(cc_II0$spe, UCL_SPE, H = H0)

  # ---- Plot-ready series CSV (idx, phase, stats, limits) ----
  nI  <- nrow(cc_I0)
  nII <- nrow(cc_II0)

  plot_I <- cc_I0 %>%
    mutate(idx = seq_len(nI),
           phase = "I",
           UCL_T2  = UCL_T2,
           LCL_T2  = LCL_T2,
           UCL_SPE = UCL_SPE,
           LCL_SPE = LCL_SPE) %>%
    select(idx, phase, T2, SPE = spe, UCL_T2, LCL_T2, UCL_SPE, LCL_SPE)

  plot_II <- cc_II0 %>%
    mutate(idx = nI + seq_len(nII),
           phase = "II",
           UCL_T2  = UCL_T2,
           LCL_T2  = LCL_T2,
           UCL_SPE = UCL_SPE,
           LCL_SPE = LCL_SPE) %>%
    select(idx, phase, T2, SPE = spe, UCL_T2, LCL_T2, UCL_SPE, LCL_SPE)

  plot_df <- bind_rows(plot_I, plot_II)

  if (!is.null(out_plot_csv)) {
    write.csv(plot_df, out_plot_csv, row.names = FALSE)
  }

  # ---- Limits CSV ----
  if (!is.null(out_limits_csv)) {
    limits_df <- data.frame(
      stat = c("T2","SPE"),
      UCL  = c(UCL_T2, UCL_SPE),
      LCL  = c(LCL_T2, LCL_SPE),
      source = "PhaseI_fixed"
    )
    write.csv(limits_df, out_limits_csv, row.names = FALSE)
  }

  # ---- JSON summary ----
  summary_list <- list(
    dims = list(G = G, d = d, n0 = nrow(TI0), nT = nrow(TII0)),
    reference = list(
      base = base,
      UCL = list(T2 = UCL_T2, SPE = UCL_SPE),
      LCL = list(T2 = LCL_T2, SPE = LCL_SPE)
    ),
    run_length = list(
      T2  = RL_T2,
      SPE = RL_SPE
    )
  )
  write(jsonlite::toJSON(summary_list, pretty = TRUE, auto_unbox = TRUE), out_json)

  message("[OK] Charts written for: ", out_csv)
  message("[OK] Summary JSON: ", out_json)

  invisible(summary_list)
}

# ---- CLI ----
args <- commandArgs(trailingOnly = TRUE)
base     <- args[1]           # e.g., ".../gauss_shift_d2" (no _rep)

K        <- if (length(args) >= 2L && nzchar(args[2])) as.integer(args[2]) else NULL
alpha    <- if (length(args) >= 3L && nzchar(args[3])) as.numeric(args[3]) else 0.05
limits   <- if (length(args) >= 4L && nzchar(args[4])) args[4] else "standard"

# explicit out_dir and export_stem
out_dir_cli     <- if (length(args) >= 5L && nzchar(args[5])) args[5] else NA_character_
export_stem_cli <- if (length(args) >= 6L && nzchar(args[6])) args[6] else NA_character_

set.seed(123)

out_dir    <- if (!is.na(out_dir_cli)) out_dir_cli else dirname(base)
export_stem<- if (!is.na(export_stem_cli)) export_stem_cli else basename(base)

out_dir    <- normalizePath(out_dir, winslash = "/", mustWork = FALSE)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_csv        <- file.path(out_dir, paste0(export_stem, "_cc_all.csv"))
out_summary    <- file.path(out_dir, paste0(export_stem, "_summary.json"))
out_plot_png   <- file.path(out_dir, paste0(export_stem, "_control_charts.png"))
out_plot_csv   <- file.path(out_dir, paste0(export_stem, "_plot_series.csv"))
out_limits_csv <- file.path(out_dir, paste0(export_stem, "_limits.csv"))
timing_path    <- file.path(out_dir, paste0(export_stem, "_timing.json"))

message("[R] base       : ", base)
message("[R] out_dir    : ", out_dir)
message("[R] export_stem: ", export_stem)

res <- run_once(
  base           = base,
  K              = K,
  alpha_overall  = alpha,
  limits         = limits,
  out_csv        = out_csv,
  out_json       = out_summary,
  out_plot_png   = out_plot_png,
  out_plot_csv   = out_plot_csv,
  out_limits_csv = out_limits_csv,
  timing_path    = timing_path,
  save_plot_png  = TRUE
)
