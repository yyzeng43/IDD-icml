import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from sklearn.decomposition import PCA
import time
import json
from numpy.linalg import LinAlgError
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
import csv as _csv
from pathlib import Path
import matplotlib.pyplot as _plt

# You already have these somewhere:
# - embed_by_full_kde(phase, X_eval, bandwidth=None, l1_normalize=True)
# - _emp_ucl
# - _first_alarm


# ===============================
# Utilities
# ===============================

def _as2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return X

# TODO: change the bandwidth
def scott_bandwidth(samples: np.ndarray) -> float:
    X = _as2d(samples)
    n, d = X.shape
    if n <= 1:
        return 1.0
    sd = X.std(axis=0, ddof=1)
    sd = np.maximum(sd, 1e-8)
    scale = float(np.mean(sd))
    # print(max(scale * n ** (-1.0 / (d + 4.0)), 1e-6))
    return max(scale * n ** (-1.0 / (d + 4.0)), 1e-6)

def _first_alarm(stat: np.ndarray, UCL: float) -> Optional[int]:
    idx = np.where(np.asarray(stat, float) > float(UCL))[0]
    return int(idx[0]) + 1 if idx.size else None  # 1-based RL

def _emp_ucl(v: np.ndarray, alpha: float) -> float:
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    q = float(np.quantile(v, 1.0 - float(alpha)))
    # Ensure strictly above typical numerical floor
    return max(q, 0.0)

def embed_by_full_kde(clouds: List[np.ndarray], X_eval: np.ndarray, bandwidth: Optional[float] = None,
                      l1_normalize: bool = True) -> Tuple[np.ndarray, float]:
    X_eval = _as2d(X_eval).astype(np.float64, copy=False)
    if bandwidth is None:
        bandwidth = scott_bandwidth(np.vstack(clouds))
    print('Used Bandwith', bandwidth)
    mats = []
    for C in clouds:
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(_as2d(C))
        p = np.exp(kde.score_samples(X_eval))
        if l1_normalize:
            s = float(p.sum())
            if s > 0:
                p = p / s
        mats.append(p)
    return np.vstack(mats), float(bandwidth)


def _read_plot_series_csv(csv_path: Path | str):
    """
    Read a plot-series CSV with columns:
      idx, phase, T2, SPE, UCL_T2, LCL_T2, UCL_SPE, LCL_SPE
    """
    csv_path = Path(csv_path)
    idx, phase, T2, SPE = [], [], [], []
    UCL_T2 = UCL_SPE = LCL_T2 = LCL_SPE = None

    with csv_path.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            idx.append(int(row["idx"]))
            phase.append(row["phase"].strip())
            T2.append(float(row["T2"]))
            SPE.append(float(row["SPE"]))
            # The limits are repeated on every row; grab the first non-empty.
            if UCL_T2 is None and row.get("UCL_T2", "") != "":
                UCL_T2 = float(row["UCL_T2"])
            if LCL_T2 is None and row.get("LCL_T2", "") != "":
                LCL_T2 = float(row["LCL_T2"])
            if UCL_SPE is None and row.get("UCL_SPE", "") != "":
                UCL_SPE = float(row["UCL_SPE"])
            if LCL_SPE is None and row.get("LCL_SPE", "") != "":
                LCL_SPE = float(row["LCL_SPE"])

    idx = np.asarray(idx, int)
    T2 = np.asarray(T2, float)
    SPE = np.asarray(SPE, float)
    phases = np.asarray(phase, object)
    nI = int(np.sum(phases == "I"))

    return dict(
        idx=idx,
        T2=T2,
        SPE=SPE,
        nI=nI,
        UCL_T2=float(UCL_T2 or 0.0),
        LCL_T2=float(LCL_T2 or 0.0),
        UCL_SPE=float(UCL_SPE or 0.0),
        LCL_SPE=float(LCL_SPE or 0.0),
    )


def _plot_series(ax, x, y, nI, UCL, LCL, ylabel: str, title: str):
    """
    Plot Phase-I (grey) and Phase-II (black) series, with red points for y > UCL.
    """
    if len(x) == 0:
        return

    x = np.asarray(x)
    y = np.asarray(y)

    xI, yI = x[:nI], y[:nI]
    xII, yII = x[nI:], y[nI:]

    # Phase I: grey
    ax.plot(xI, yI, "-", color="#bdbdbd", linewidth=1.2)
    ax.scatter(xI, yI, s=14, color="#bdbdbd", zorder=3)

    # Phase II: black
    ax.plot(xII, yII, "-", color="black", linewidth=1.2)
    ax.scatter(xII, yII, s=14, color="black", zorder=3)

    # Out-of-control points (any phase) in red
    if UCL is not None:
        mask_red = y > UCL
        if np.any(mask_red):
            ax.scatter(
                x[mask_red],
                y[mask_red],
                s=30,
                color="red",
                edgecolor="none",
                zorder=5,
            )

    # Control limits
    if UCL is not None:
        ax.axhline(UCL, color="black", linewidth=1.4)
    if LCL is not None:
        ax.axhline(LCL, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    # Phase-I / Phase-II separator
    ax.axvline(nI + 0.5, color="black", linestyle="--", linewidth=1.0)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x.min(), x.max())


def plot_control_charts_from_csv(
    csv_path: Path | str,
    out_png: Optional[Path | str] = None,
    title_prefix: str = "",
    titles=None,
):
    """
    Two-panel chart: T² (top) + SPE (bottom), from a plot-series CSV.
    """
    if titles is None:
        titles = ("HOTELLING T2 CONTROL CHART", "SPE CONTROL CHART")

    data = _read_plot_series_csv(csv_path)
    idx = data["idx"]
    nI = data["nI"]

    fig, axes = _plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, constrained_layout=True
    )

    _plot_series(
        axes[0],
        idx,
        data["T2"],
        nI,
        data["UCL_T2"],
        data["LCL_T2"],
        ylabel="T2 statistic",
        title=f"{title_prefix}{titles[0]}",
    )
    _plot_series(
        axes[1],
        idx,
        data["SPE"],
        nI,
        data["UCL_SPE"],
        data["LCL_SPE"],
        ylabel="SPE statistic",
        title=f"{title_prefix}{titles[1]}",
    )

    axes[1].set_xlabel("Observation")

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)

    return fig, axes
# ===============================
# log-KDE–MFPCA (full d-variate)
# ===============================

@dataclass
class LogKDEMFPCAFull:
    pca: PCA
    X_eval: np.ndarray
    bandwidth: float
    n_pc: int
    evals: np.ndarray
    UCL_T2: float   # empirical from Phase I
    UCL_SPE: float  # empirical from Phase I
    alpha: float
    eps_log: float  # floor used before log


def _embed_full_log_kde(
    samples: List[np.ndarray],
    X_eval: np.ndarray,
    bandwidth: Optional[float] = None,
    eps_log: float = 1e-8,
    l1_normalize: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Embed each distribution as its *log-density* on a common evaluation grid.

    1) compute KDE on X_eval (possibly choosing bandwidth if None),
    2) enforce positivity with a small floor eps_log,
    3) (optionally) renormalize rows to sum to 1,
    4) take log.

    Returns:
      G : (n_samples, G) matrix of log-densities,
      h : bandwidth actually used.
    """
    # D: densities on grid, shape (n_samples, G)
    D, h = embed_by_full_kde(
        samples,
        X_eval,
        bandwidth=bandwidth,
        l1_normalize=l1_normalize
    )
    D = np.asarray(D, dtype=float)

    # floor to avoid log(0)
    D_clipped = np.maximum(D, eps_log)

    if l1_normalize:
        row_sums = D_clipped.sum(axis=1, keepdims=True)
        # guard against numerical issues
        row_sums[row_sums <= 0] = 1.0
        D_clipped = D_clipped / row_sums

    G = np.log(D_clipped)
    return G, float(h)


def logkde_mfpca_full_fit_phaseI(
    phaseI: List[np.ndarray],
    X_eval: np.ndarray,
    bandwidth: Optional[float] = None,
    var_explained: float = 0.95,
    alpha: float = 0.05,
    eps_log: float = 1e-8,
) -> Tuple[LogKDEMFPCAFull, Dict[str, np.ndarray]]:
    """
    Phase I fit: compute log-KDE embeddings, PCA, and empirical UCLs.
    """
    # 1) embed Phase I clouds as log-densities
    G_I, h = _embed_full_log_kde(
        phaseI,
        X_eval,
        bandwidth=bandwidth,
        eps_log=eps_log,
        l1_normalize=True,
    )

    # 2) PCA on log-densities
    pca = PCA(svd_solver="full").fit(G_I)
    evals = np.asarray(pca.explained_variance_, float)
    csum = np.cumsum(pca.explained_variance_ratio_)
    n_pc = int(np.searchsorted(csum, var_explained) + 1)
    n_pc = max(1, min(n_pc, len(evals)))

    # 3) T² and SPE for Phase I
    Z_I = pca.transform(G_I)
    if n_pc > 0:
        T2_I = np.sum((Z_I[:, :n_pc] ** 2) / (evals[:n_pc] + 1e-12), axis=1)
    else:
        T2_I = np.zeros(G_I.shape[0])
    if n_pc < len(evals):
        SPE_I = np.sum(Z_I[:, n_pc:] ** 2, axis=1)
    else:
        SPE_I = np.zeros(G_I.shape[0])

    UCL_T2  = _emp_ucl(T2_I,  alpha)
    UCL_SPE = _emp_ucl(SPE_I, alpha)

    model = LogKDEMFPCAFull(
        pca=pca,
        X_eval=np.asarray(X_eval, float),
        bandwidth=h,
        n_pc=n_pc,
        evals=evals,
        UCL_T2=float(UCL_T2),
        UCL_SPE=float(UCL_SPE),
        alpha=float(alpha),
        eps_log=float(eps_log),
    )
    return model, dict(T2=T2_I, SPE=SPE_I)


def logkde_mfpca_full_score_phaseI(
    model: LogKDEMFPCAFull,
    phaseI: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Re-score Phase I data with a fitted log-KDE–MFPCA model
    (useful mainly for diagnostics).
    """
    G_I, _ = _embed_full_log_kde(
        phaseI,
        model.X_eval,
        bandwidth=model.bandwidth,
        eps_log=model.eps_log,
        l1_normalize=True,
    )

    Z = model.pca.transform(G_I)
    evals, n_pc = model.evals, int(model.n_pc)

    T2 = np.sum((Z[:, :n_pc] ** 2) / (evals[:n_pc] + 1e-12), axis=1)
    if n_pc < len(evals):
        SPE = np.sum(Z[:, n_pc:] ** 2, axis=1)
    else:
        SPE = np.zeros(G_I.shape[0])

    return dict(T2=T2, SPE=SPE)


def logkde_mfpca_full_score_phaseII(
    model: LogKDEMFPCAFull,
    phaseII: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float, float, int, int, int]:
    """
    Score Phase II sequence with a fitted log-KDE–MFPCA model.

    Returns:
      T2    : T² statistics for each Phase-II distribution
      SPE   : SPE residuals
      UCL_T2, UCL_SPE : control limits (from Phase I)
      H     : number of Phase-II points
      RL_T2 : run length for T² (first > UCL_T2, or H+1)
      RL_SPE: run length for SPE (first > UCL_SPE, or H+1)
    """
    G_II, _ = _embed_full_log_kde(
        phaseII,
        model.X_eval,
        bandwidth=model.bandwidth,
        eps_log=model.eps_log,
        l1_normalize=True,
    )

    Z = model.pca.transform(G_II)
    evals, n_pc = model.evals, int(model.n_pc)

    T2 = np.sum((Z[:, :n_pc] ** 2) / (evals[:n_pc] + 1e-12), axis=1)
    if n_pc < len(evals):
        SPE = np.sum(Z[:, n_pc:] ** 2, axis=1)
    else:
        SPE = np.zeros(G_II.shape[0])

    UCL_T2, UCL_SPE = float(model.UCL_T2), float(model.UCL_SPE)
    H = len(T2)
    RL_T2  = _first_alarm(T2,  UCL_T2) or (H + 1)
    RL_SPE = _first_alarm(SPE, UCL_SPE) or (H + 1)

    return T2, SPE, UCL_T2, UCL_SPE, H, int(RL_T2), int(RL_SPE)



def logkde_mfpca_full_run_and_plot(
    phaseI: List[np.ndarray],
    phaseII: List[np.ndarray],
    X_eval_grid: np.ndarray,
    out_prefix: Path | str,
    bandwidth: float | None = None,
    var_explained: float = 0.95,
    alpha: float = 0.05,
    eps_log: float = 1e-8,
    title_prefix: str = "log-KDE – ",
):
    """
    Fit log-KDE–MFPCA on Phase I, score Phase II, save a *_plot_series.csv,
    and draw control charts with out-of-control points in red.

    Parameters
    ----------
    phaseI, phaseII : list of np.ndarray
        Point clouds (one array per distribution).
    X_eval_grid : (G, d) array
        Common evaluation grid (e.g., OT barycenter support).
    out_prefix : str or Path
        Prefix for outputs; CSV and PNG will be:
          <out_prefix>_plot_series.csv
          <out_prefix>_control_charts.png
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Fit on Phase I ----
    # record the staring time
    t0 = time.perf_counter()

    model, stats_I = logkde_mfpca_full_fit_phaseI(
        phaseI=phaseI,
        X_eval=X_eval_grid,
        bandwidth=bandwidth,
        var_explained=var_explained,
        alpha=alpha,
        eps_log=eps_log,
    )

    T2_I = np.asarray(stats_I["T2"], float)
    SPE_I = np.asarray(stats_I["SPE"], float)
    nI = len(T2_I)
    t_phaseI = time.perf_counter() - t0
    print('KDE estimation time for Phase I {:.3f}s'.format(t_phaseI))


    # ---- 2. Score Phase II ----
    T2_II, SPE_II, UCL_T2, UCL_SPE, H, RL_T2, RL_SPE = \
        logkde_mfpca_full_score_phaseII(model, phaseII)

    T2_II = np.asarray(T2_II, float)
    SPE_II = np.asarray(SPE_II, float)
    nII = len(T2_II)

    t_phaseII = time.perf_counter() - t0
    print('KDE estimation time for Phase II {:.3f}s'.format(t_phaseII))

    # ---- 3. Build combined series (Phase I then II) ----
    idx_I = np.arange(1, nI + 1, dtype=int)
    idx_II = np.arange(nI + 1, nI + nII + 1, dtype=int)

    # LCLs (you can set to 0 for now)
    LCL_T2 = 0.0
    LCL_SPE = 0.0

    rows = []

    # Phase I rows
    for i, t2, spe in zip(idx_I, T2_I, SPE_I):
        rows.append(dict(
            idx=i,
            phase="I",
            T2=t2,
            SPE=spe,
            UCL_T2=UCL_T2,
            LCL_T2=LCL_T2,
            UCL_SPE=UCL_SPE,
            LCL_SPE=LCL_SPE,
        ))

    # Phase II rows
    for i, t2, spe in zip(idx_II, T2_II, SPE_II):
        rows.append(dict(
            idx=i,
            phase="II",
            T2=t2,
            SPE=spe,
            UCL_T2=UCL_T2,
            LCL_T2=LCL_T2,
            UCL_SPE=UCL_SPE,
            LCL_SPE=LCL_SPE,
        ))

    # ---- 4. Save CSV ----
    csv_path = out_prefix.with_suffix("")  # strip any suffix
    csv_path = csv_path.parent / (csv_path.name + "_plot_series.csv")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=["idx", "phase", "T2", "SPE",
                        "UCL_T2", "LCL_T2", "UCL_SPE", "LCL_SPE"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---- 5. Plot control charts from CSV ----
    png_path = out_prefix.with_suffix("")
    png_path = png_path.parent / (png_path.name + "_control_charts.png")

    fig, axes = plot_control_charts_from_csv(
        csv_path,
        out_png=png_path,
        title_prefix=title_prefix,
    )

    return dict(
        model=model,
        csv_path=csv_path,
        png_path=png_path,
        RL_T2=RL_T2,
        RL_SPE=RL_SPE,
        T2_I=T2_I,
        SPE_I=SPE_I,
        T2_II=T2_II,
        SPE_II=SPE_II,
    )


def plot_kde_control_charts(
    T2_I, SPE_I,
    T2_II, SPE_II,
    UCL_T2, UCL_SPE,
    out_png: Path,
    title_prefix: str = "log-KDE – ",
    scenario: str = "IC",
    bandwidth_label: str = "auto",
):
    """
    Plot 2-panel control chart (T2 + SPE) for log-KDE method.
    Phase I + Phase II in one chart, using your existing _plot_series style.
    """
    T2_I   = np.asarray(T2_I, float)
    SPE_I  = np.asarray(SPE_I, float)
    T2_II  = np.asarray(T2_II, float)
    SPE_II = np.asarray(SPE_II, float)

    nI = len(T2_I)
    T2_all  = np.concatenate([T2_I,  T2_II])
    SPE_all = np.concatenate([SPE_I, SPE_II])

    idx = np.arange(1, len(T2_all) + 1)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = _plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)

    title_T2  = f"{title_prefix}T² ({scenario}, h={bandwidth_label})"
    title_SPE = f"{title_prefix}SPE ({scenario}, h={bandwidth_label})"

    # LCL = 0 for T2/SPE
    _plot_series(
        axes[0],
        idx, T2_all,
        nI=nI,
        UCL=UCL_T2,
        LCL=0.0,
        ylabel="T² statistic",
        title=title_T2,
    )
    _plot_series(
        axes[1],
        idx, SPE_all,
        nI=nI,
        UCL=UCL_SPE,
        LCL=0.0,
        ylabel="SPE statistic",
        title=title_SPE,
    )

    axes[1].set_xlabel("Observation")

    fig.savefig(out_png, dpi=200)
    _plt.close(fig)