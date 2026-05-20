# To use a specific Python interpreter, set the RETICULATE_PYTHON environment
# variable (e.g. export RETICULATE_PYTHON=/path/to/.venv/bin/python) before
# running this script. If unset, reticulate auto-detects an available Python.

suppressPackageStartupMessages({
  library(reticulate)
  library(funcharts)
  library(dplyr)
  library(jsonlite)
})

`%||%` <- function(a, b) {
  if (is.null(a)) return(b)
  if (length(a) == 0L) return(b)
  if (length(a) == 1L && is.na(a)) return(b)
  a
}

# ---------- numpy loader ----------
np <- reticulate::import("numpy")

load_np_matrix <- function(path) {
  x <- reticulate::py_to_r(np$load(path, allow_pickle = TRUE))
  if (is.numeric(x) && length(dim(x)) == 2L) return(as.matrix(x))
  if (is.numeric(x) && is.null(dim(x)))      return(matrix(x, nrow = 1L))
  if (is.list(x) && all(vapply(x, is.numeric, logical(1L)))) {
    return(do.call(rbind, lapply(x, as.numeric)))
  }
  stop("Unsupported .npy at: ", path)
}

# ---------- reshape: (n × Gd) -> list of d matrices (n × G) ----------
to_mfd_list <- function(MX, G, d, prefix = "coord") {
  stopifnot(ncol(MX) == G * d)
  comps <- vector("list", d)
  for (j in seq_len(d)) {
    idx <- ((j - 1L) * G + 1L):(j * G)
    comps[[j]] <- MX[, idx, drop = FALSE]
  }
  names(comps) <- paste0(prefix, seq_len(d))
  comps
}

build_mfd_grid <- function(list_of_mats) {
  funcharts::get_mfd_list(list_of_mats)
}

standardize_with_phaseI <- function(TI0, TII0 = NULL) {
  mu  <- colMeans(TI0)
  sdv <- apply(TI0, 2, sd)
  eps <- 1e-8
  sdv[!is.finite(sdv) | sdv < eps] <- eps

  TI0s  <- sweep(sweep(TI0,  2, mu, "-"), 2, sdv, "/")
  if (is.null(TII0)) {
    return(list(TI0s = TI0s, mu = mu, sdv = sdv, TII0s = NULL))
  }
  TII0s <- sweep(sweep(TII0, 2, mu, "-"), 2, sdv, "/")
  list(TI0s = TI0s, TII0s = TII0s, mu = mu, sdv = sdv)
}

# ------------------------------------------------------------
# FIT: train on Phase I only, save model.rds, output Phase I scores
# ------------------------------------------------------------
fit_once <- function(base_prefix,
                     out_dir,
                     export_stem,
                     model_rds,
                     K = NULL,
                     tot_var = 0.95,
                     single_min = 0.00) {

  Xb_path    <- paste0(base_prefix, ".Xb.npy")
  phaseI_path <- paste0(base_prefix, ".PhaseI.npy")

  Xb  <- load_np_matrix(Xb_path)
  TI0 <- load_np_matrix(phaseI_path)

  G <- nrow(Xb)
  d <- ncol(Xb)

  if (ncol(TI0) != G * d) stop("PhaseI ncol mismatch with Xb: ", phaseI_path)

  std <- standardize_with_phaseI(TI0, NULL)
  TI0s <- std$TI0s

  mfd_train <- build_mfd_grid(to_mfd_list(TI0s,  G, d, "coord"))

  t0 <- proc.time()
  pca0 <- funcharts::pca_mfd(mfd_train, scale = FALSE)
  comps_arg <- if (!is.null(K)) 1:as.integer(K) else NULL

  cc_I <- funcharts::control_charts_pca(
    pca0,
    components = comps_arg,
    tuning_data = mfd_train,
    newdata = mfd_train,
    alpha = 0.05,
    limits = "standard",
    tot_variance_explained = tot_var,
    single_min_variance_explained = single_min
  )
  t_fit <- as.numeric((proc.time() - t0)[["elapsed"]])

  # save model for reuse (includes standardization + training grid)
  dir.create(dirname(model_rds), recursive = TRUE, showWarnings = FALSE)
  saveRDS(list(
    pca0 = pca0,
    mfd_train = mfd_train,
    mu = std$mu,
    sdv = std$sdv,
    G = G,
    d = d,
    K = K,
    tot_var = tot_var,
    single_min = single_min
  ), model_rds)

  scores_I <- data.frame(
    idx   = seq_len(nrow(cc_I)),
    phase = "I",
    T2    = cc_I$T2,
    SPE   = cc_I$spe
  )

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_csv   <- file.path(out_dir, paste0(export_stem, "_scores_phaseI.csv"))
  timing_js <- file.path(out_dir, paste0(export_stem, "_timing_fit.json"))

  write.csv(scores_I, out_csv, row.names = FALSE)
  write(
    jsonlite::toJSON(list(mfpca_fit_sec = t_fit), pretty = TRUE, auto_unbox = TRUE),
    timing_js
  )
  invisible(NULL)
}

# ------------------------------------------------------------
# SCORE: load model.rds, score Phase II only, output Phase II scores
# ------------------------------------------------------------
score_once <- function(base_prefix,
                       out_dir,
                       export_stem,
                       model_rds) {

  Xb_path     <- paste0(base_prefix, ".Xb.npy")
  phaseII_path <- paste0(base_prefix, ".PhaseII.npy")

  Xb   <- load_np_matrix(Xb_path)
  TII0 <- load_np_matrix(phaseII_path)

  mdl <- readRDS(model_rds)
  G <- mdl$G
  d <- mdl$d

  # shape checks
  if (nrow(Xb) != G || ncol(Xb) != d) {
    stop("Xb shape mismatch with saved model: expected (", G, ",", d, ")")
  }
  if (ncol(TII0) != G * d) stop("PhaseII ncol mismatch with Xb: ", phaseII_path)

  # standardize with saved mu/sdv
  mu <- mdl$mu
  sdv <- mdl$sdv
  eps <- 1e-8
  sdv[!is.finite(sdv) | sdv < eps] <- eps
  TII0s <- sweep(sweep(TII0, 2, mu, "-"), 2, sdv, "/")

  mfd_phase2 <- build_mfd_grid(to_mfd_list(TII0s, G, d, "coord"))

  t0 <- proc.time()
  comps_arg <- if (!is.null(mdl$K)) 1:as.integer(mdl$K) else NULL

  cc_II <- funcharts::control_charts_pca(
    mdl$pca0,
    components = comps_arg,
    tuning_data = mdl$mfd_train,
    newdata = mfd_phase2,
    alpha = 0.05,
    limits = "standard",
    tot_variance_explained = mdl$tot_var,
    single_min_variance_explained = mdl$single_min
  )
  t_score <- as.numeric((proc.time() - t0)[["elapsed"]])

  scores_II <- data.frame(
    idx   = seq_len(nrow(cc_II)),
    phase = "II",
    T2    = cc_II$T2,
    SPE   = cc_II$spe
  )

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  out_csv   <- file.path(out_dir, paste0(export_stem, "_scores_phaseII.csv"))
  timing_js <- file.path(out_dir, paste0(export_stem, "_timing_score.json"))

  write.csv(scores_II, out_csv, row.names = FALSE)
  write(
    jsonlite::toJSON(list(mfpca_score_sec = t_score), pretty = TRUE, auto_unbox = TRUE),
    timing_js
  )
  invisible(NULL)
}

# ------------------------------------------------------------
# Backward-compatible "run_once": fits on Phase I and scores Phase II in one shot
# ------------------------------------------------------------
run_once <- function(base_prefix,
                     out_dir,
                     export_stem,
                     K = NULL,
                     tot_var = 0.95,
                     single_min = 0.00) {

  Xb_path   <- paste0(base_prefix, ".Xb.npy")
  phaseI_path  <- paste0(base_prefix, ".PhaseI.npy")
  phaseII_path <- paste0(base_prefix, ".PhaseII.npy")

  Xb  <- load_np_matrix(Xb_path)
  TI0 <- load_np_matrix(phaseI_path)
  TII0<- load_np_matrix(phaseII_path)

  G <- nrow(Xb)
  d <- ncol(Xb)

  if (ncol(TI0) != G * d) stop("PhaseI ncol mismatch with Xb: ", phaseI_path)
  if (ncol(TII0) != G * d) stop("PhaseII ncol mismatch with Xb: ", phaseII_path)

  # ---- standardize on Phase I only ----
  std <- standardize_with_phaseI(TI0, TII0)
  TI0s  <- std$TI0s
  TII0s <- std$TII0s

  mfd_train  <- build_mfd_grid(to_mfd_list(TI0s,  G, d, "coord"))
  mfd_phase2 <- build_mfd_grid(to_mfd_list(TII0s, G, d, "coord"))

  t0 <- proc.time()

  pca0 <- funcharts::pca_mfd(mfd_train, scale = FALSE)
  comps_arg <- if (!is.null(K)) 1:as.integer(K) else NULL

  cc_I <- funcharts::control_charts_pca(
    pca0,
    components = comps_arg,
    tuning_data = mfd_train,
    newdata = mfd_train,
    alpha = 0.05,
    limits = "standard",
    tot_variance_explained = tot_var,
    single_min_variance_explained = single_min
  )

  cc_II <- funcharts::control_charts_pca(
    pca0,
    components = comps_arg,
    tuning_data = mfd_train,
    newdata = mfd_phase2,
    alpha = 0.05,
    limits = "standard",
    tot_variance_explained = tot_var,
    single_min_variance_explained = single_min
  )

  t_fit <- as.numeric((proc.time() - t0)[["elapsed"]])

  nI  <- nrow(cc_I)
  nII <- nrow(cc_II)

  scores_I <- data.frame(
    idx   = seq_len(nI),
    phase = "I",
    T2    = cc_I$T2,
    SPE   = cc_I$spe
  )

  scores_II <- data.frame(
    idx   = nI + seq_len(nII),
    phase = "II",
    T2    = cc_II$T2,
    SPE   = cc_II$spe
  )

  scores_all <- bind_rows(scores_I, scores_II)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  out_csv   <- file.path(out_dir, paste0(export_stem, "_scores.csv"))
  timing_js <- file.path(out_dir, paste0(export_stem, "_timing.json"))

  write.csv(scores_all, out_csv, row.names = FALSE)
  write(
    jsonlite::toJSON(list(mfpca_fit_sec = t_fit), pretty = TRUE, auto_unbox = TRUE),
    timing_js
  )

  invisible(NULL)
}

# -------- CLI --------
# New usage:
#   Fit:   Rscript run_mfpca.R fit   base_prefix out_dir export_stem model_rds [K]
#   Score: Rscript run_mfpca.R score base_prefix out_dir export_stem model_rds
# Backward-compatible:
#   Rscript run_mfpca.R base_prefix out_dir export_stem [K]
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1L && tolower(args[[1]]) %in% c("fit", "score")) {
  mode <- tolower(args[[1]])
  if (mode == "fit") {
    if (length(args) < 5L) stop("Usage: Rscript run_mfpca.R fit base_prefix out_dir export_stem model_rds [K]")
    base_prefix <- args[[2]]
    out_dir     <- args[[3]]
    export_stem <- args[[4]]
    model_rds   <- args[[5]]
    K <- if (length(args) >= 6L && nzchar(args[[6]])) as.integer(args[[6]]) else NULL
    fit_once(base_prefix, out_dir, export_stem, model_rds, K = K)
  } else {
    if (length(args) < 5L) stop("Usage: Rscript run_mfpca.R score base_prefix out_dir export_stem model_rds")
    base_prefix <- args[[2]]
    out_dir     <- args[[3]]
    export_stem <- args[[4]]
    model_rds   <- args[[5]]
    score_once(base_prefix, out_dir, export_stem, model_rds)
  }
} else {
  if (length(args) < 3L) {
    stop("Usage: Rscript run_mfpca.R base_prefix out_dir export_stem [K]")
  }
  base_prefix <- args[[1]]
  out_dir     <- args[[2]]
  export_stem <- args[[3]]
  K <- if (length(args) >= 4L && nzchar(args[[4]])) as.integer(args[[4]]) else NULL
  run_once(base_prefix, out_dir, export_stem, K = K)
}
