Sys.setenv(RETICULATE_PYTHON = "C:/Users/zengyy/Research/DIDO_CC/DIDO_CC/.venv/Scripts/python.exe")


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
  mu  <- colMeans(TI0)
  sdv <- apply(TI0, 2, sd)
  eps <- 1e-8
  sdv[!is.finite(sdv) | sdv < eps] <- eps

  TI0s  <- sweep(sweep(TI0,  2, mu, "-"), 2, sdv, "/")
  TII0s <- sweep(sweep(TII0, 2, mu, "-"), 2, sdv, "/")

  mfd_train  <- build_mfd_grid(to_mfd_list(TI0s,  G, d, "coord"))
  mfd_phase2 <- build_mfd_grid(to_mfd_list(TII0s, G, d, "coord"))

  t0 <- proc.time()

  pca0 <- funcharts::pca_mfd(mfd_train, scale = FALSE)
  comps_arg <- if (!is.null(K)) 1:as.integer(K) else NULL

  # alpha is irrelevant here; we ignore the limits it returns
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
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop("Usage: Rscript run_mfpca_scores.R base_prefix out_dir export_stem [K]")
}
base_prefix <- args[[1]]
out_dir     <- args[[2]]
export_stem <- args[[3]]
K <- if (length(args) >= 4L && nzchar(args[[4]])) as.integer(args[[4]]) else NULL

run_once(base_prefix, out_dir, export_stem, K = K)
