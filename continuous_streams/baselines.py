from __future__ import annotations
from pathlib import Path
import time, json
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# ============================================================
# Helper: build evaluation grid from Phase-I raw data
# ============================================================

def make_X_eval_from_phaseI(cloudsI,
                            n_eval: int = 512,
                            seed: int = 0,
                            mode: str = "pooled_subsample"):
    rng = np.random.default_rng(seed)
    pooled = np.vstack(cloudsI)
    N, d = pooled.shape
    if mode == "linspace_1d" and d == 1:
        lo, hi = np.quantile(pooled[:, 0], [0.01, 0.99])
        xs = np.linspace(lo, hi, n_eval)
        return xs.reshape(-1, 1)
    if n_eval >= N:
        return pooled.copy()
    idx = rng.choice(N, size=n_eval, replace=False)
    return pooled[idx]


def _as2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return X


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
# ============================================================
# Loader helpers (Phase-I / Phase-II)
# ============================================================

def load_clouds_from_npz(phaseI_file: Path, key="phaseI"):
    z = np.load(phaseI_file, allow_pickle=True)
    raw = list(z[key])
    return [np.asarray(r, float) for r in raw]


# ============================================================
# Core log-KDE evaluation
# ============================================================

def _logkde_eval(clouds, X_eval, bandwidth: float | None,
                eps_log: float = 1e-8, l1_normalize: bool = True):
    """
    Full d-dimensional KDE log-densities on X_eval.
    Returns: (n, G)
    """
    clouds = [np.asarray(c, float) for c in clouds]
    X_eval = np.asarray(X_eval, float)
    pooled = np.vstack(clouds)

    h_used = float(bandwidth) if bandwidth is not None else float(scott_bandwidth(pooled))

    logs = []
    for c in clouds:
        kde = KernelDensity(kernel="gaussian", bandwidth=h_used)
        kde.fit(c)
        logf = kde.score_samples(X_eval)  # (G,)
        if l1_normalize:
            f = np.exp(logf)
            f = np.maximum(f, eps_log)
            s = f.sum()
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            f = f / s
            logf = np.log(f)
        logs.append(logf)

    return np.stack(logs, axis=0), h_used  # (n,G), scalar


def _logkde_eval_marginal(clouds, X_eval, bandwidth: float | None,
                          eps_log: float = 1e-8, l1_normalize: bool = True):
    """
    Marginal KDE per dimension on X_eval[:,j].
    Returns: (n, G*d)
    Also returns bandwidth(s) used: either scalar or (d,) array.
    """
    clouds = [np.asarray(c, float) for c in clouds]
    X_eval = np.asarray(X_eval, float)
    G, d = X_eval.shape
    n = len(clouds)

    logs_all = np.zeros((n, G * d), float)
    h_used = np.zeros(d, float)

    for j in range(d):
        pooled_j = np.vstack([c[:, [j]] for c in clouds])  # (sum n_i, 1)
        hj = float(bandwidth) if bandwidth is not None else float(scott_bandwidth(pooled_j))
        h_used[j] = hj

        xj = X_eval[:, [j]]  # (G,1)
        for i, c in enumerate(clouds):
            kde = KernelDensity(kernel="gaussian", bandwidth=hj)
            kde.fit(c[:, [j]])
            logf = kde.score_samples(xj)  # (G,)
            if l1_normalize:
                f = np.exp(logf)
                f = np.maximum(f, eps_log)
                s = f.sum()
                if not np.isfinite(s) or s <= 0:
                    s = 1.0
                f = f / s
                logf = np.log(f)
            logs_all[i, j * G:(j + 1) * G] = logf

    return logs_all, h_used  # (n, G*d), (d,)


# ============================================================
# Exporter for mFPCA (full + marginal)
# ============================================================

def export_kde_for_mfpca(
    phaseI_file: Path,
    phaseII_file: Path,
    cfg_name: str,
    kde_export_root: Path,
    bandwidth: float | None = None,
    mode: str = "full",   # "full" or "marginal"
    n_eval: int = 512,
) -> tuple[Path, dict]:
    """
    Build KDE-based features for mFPCA for a single replication.
    Saves:
      base.Xb.npy, base.PhaseI.npy, base.PhaseII.npy, base.meta.json

    Timing:
      - t_phaseI : Phase I work (incl. X_eval construction + KDE on Phase I + saving PhaseI/Xb/meta)
      - t_phaseII: Phase II work (KDE on Phase II + saving PhaseII)
      - t_feat   : t_phaseI + t_phaseII (kept for backward compatibility)
    """
    kde_export_root = Path(kde_export_root)
    kde_export_root.mkdir(parents=True, exist_ok=True)

    base_prefix = kde_export_root / cfg_name / phaseI_file.stem.replace("phaseI", f"KDE_{mode}")
    base_prefix.parent.mkdir(parents=True, exist_ok=True)

    cloudsI = load_clouds_from_npz(phaseI_file, "phaseI")
    streamsII = load_clouds_from_npz(phaseII_file, "streams")

    d = cloudsI[0].shape[1]
    X_eval = make_X_eval_from_phaseI(
        cloudsI,
        n_eval=n_eval,
        seed=0,
        mode="pooled_subsample" if d > 1 else "linspace_1d",
    )
    G = X_eval.shape[0]

    # -------------------------
    # Phase I (charged)
    # -------------------------
    t0 = time.perf_counter()

    if mode == "full":
        # Phase I KDE + (possibly) bandwidth selection
        logI, h_used = _logkde_eval(cloudsI, X_eval, bandwidth)
        d_comp = 1
        Xb = np.arange(G, dtype=float).reshape(G, 1)  # (G,1)

    elif mode == "marginal":
        logI, h_used = _logkde_eval_marginal(cloudsI, X_eval, bandwidth)
        d_comp = d
        grid = np.arange(G, dtype=float).reshape(G, 1)
        Xb = np.repeat(grid, d, axis=1)

    else:
        raise ValueError("mode must be 'full' or 'marginal'")

    # If bandwidth is None, reuse Phase-I chosen bandwidth for Phase II
    # bw_used = float(h_used if bandwidth is None else bandwidth)
    val = h_used if bandwidth is None else bandwidth

    # val can be scalar (full KDE) or vector (marginal KDE). Convert safely.
    val = np.asarray(val)
    if val.size == 1:
        bw_used = float(val)
    else:
        v = val[np.isfinite(val)]
        bw_used = float(np.median(v)) if v.size else float(np.median(val))

    # Save Phase I artifacts (counted in Phase I time)
    np.save(str(base_prefix) + ".Xb.npy", Xb.astype(np.float64))
    np.save(str(base_prefix) + ".PhaseI.npy", logI.astype(np.float64))

    meta = {"G": int(G), "d_comp": int(d_comp), "bandwidth": bandwidth, "bandwidth_used": bw_used, "mode": mode}
    with open(str(base_prefix) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    t_phaseI = time.perf_counter() - t0

    # -------------------------
    # Phase II (charged)
    # -------------------------
    t1 = time.perf_counter()

    if mode == "full":
        logII, _ = _logkde_eval(streamsII, X_eval, bw_used)
    else:  # "marginal"
        logII, _ = _logkde_eval_marginal(streamsII, X_eval, bw_used)

    # Save Phase II artifact (counted in Phase II time)
    np.save(str(base_prefix) + ".PhaseII.npy", logII.astype(np.float64))

    t_phaseII = time.perf_counter() - t1

    timing = {
        "t_phaseI": float(t_phaseI),
        "t_phaseII": float(t_phaseII),
        "t_feat": float(t_phaseI + t_phaseII),  # backward compatible
        "G": int(G),
        "d_comp": int(d_comp),
        "bandwidth_used": float(bw_used),
    }
    return base_prefix, timing


# for the mean charts
# ============================================================
# Your plotting style helper (copied from your definition)
# ============================================================

def _plot_series(ax, x, y, nI, UCL, LCL, ylabel: str, title: str):
    if len(x) == 0:
        return
    x = np.asarray(x)
    y = np.asarray(y)

    xI, yI = x[:nI], y[:nI]
    xII, yII = x[nI:], y[nI:]

    ax.plot(xI, yI, "-", color="#bdbdbd", linewidth=1.2)
    ax.scatter(xI, yI, s=14, color="#bdbdbd", zorder=3)

    ax.plot(xII, yII, "-", color="black", linewidth=1.2)
    ax.scatter(xII, yII, s=14, color="black", zorder=3)

    # Out-of-control points in red (two-sided)
    mask_red = np.zeros_like(y, dtype=bool)
    if UCL is not None:
        mask_red |= (y > UCL)
    if LCL is not None:
        mask_red |= (y < LCL)

    if np.any(mask_red):
        ax.scatter(
            x[mask_red],
            y[mask_red],
            s=30,
            color="red",
            edgecolor="none",
            zorder=5,
        )

    if UCL is not None:
        ax.axhline(UCL, color="black", linewidth=1.4)
    if LCL is not None:
        ax.axhline(LCL, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.axvline(nI + 0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x.min(), x.max())


def plot_mean_control_chart(
    df_full: pd.DataFrame,
    UCL: float,
    LCL: float,
    out_png: Path,
    title: str,
    ylabel: str = "Statistic",
):
    """
    df_full must contain: idx, phase in {'I','II'}, stat
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    nI = int((df_full["phase"] == "I").sum())
    x = df_full["idx"].values
    y = df_full["stat"].values

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
    _plot_series(ax, x, y, nI=nI, UCL=UCL, LCL=LCL, ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ============================================================
# Core utilities
# ============================================================

def compute_RL_and_alarms(x, UCL):
    x = np.asarray(x)
    idx = np.where(x > UCL)[0]
    if len(idx) == 0:
        return int(len(x) + 1), 0
    return int(idx[0] + 1), int(len(idx))

def compute_RL_and_alarms_two_sided(x, UCL=None, LCL=None):
    x = np.asarray(x)

    mask = np.zeros_like(x, dtype=bool)
    if UCL is not None:
        mask |= (x > UCL)
    if LCL is not None:
        mask |= (x < LCL)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return int(len(x) + 1), 0
    return int(idx[0] + 1), int(mask.sum())


def load_clouds(npz_file: Path, key: str):
    z = np.load(npz_file, allow_pickle=True)
    return [np.asarray(x, float) for x in z[key]]


def sample_means(clouds):
    return np.stack([c.mean(axis=0) for c in clouds], axis=0)  # (T,d)


def hotelling_T2(X, mu, Sigma_inv):
    D = X - mu[None, :]
    return np.einsum("ti,ij,tj->t", D, Sigma_inv, D)


# ============================================================
# Main function (now saves + plots)
# ============================================================

def compute_mean_charts_for_rep(
    datadir: Path,
    resdir: Path,
    d: int,
    K: int,
    scenarios,
    n_rep: int,
    alpha: float = 0.05,
    save_ic_plots: bool = False,
):
    """
    Mean-based baselines:
      - d==1: Xbar on sample mean
      - d>=2: Hotelling T2 on sample mean vector

    Control limit:
      - empirical UCL from pooled Phase-II IC across replications

    Saves:
      - limits JSON/CSV (per config)
      - per-rep series CSV (Phase I + II)
      - per-rep plots for each OC scenario (T2 only; no SPE here)

    Returns:
      rows list (for your global summary aggregation)
    """
    datadir = Path(datadir)
    resdir = Path(resdir)

    method = "Xbar" if d == 1 else "HotellingT2"
    cfg_tag = f"d{d}_K{K}"
    out_dir = resdir / f"phase_{cfg_tag}" / method
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # --------------------------------------------------------
    # 1) Estimate mu0, Sigma0 from Phase-I across reps (pooled)
    # --------------------------------------------------------
    phaseI_means_all = []
    n0 = None

    for rep in range(n_rep):
        phaseI_file = datadir / f"phaseI_d{d}_K{K}_rep{rep:02d}.npz"
        cloudsI = load_clouds(phaseI_file, "phaseI")
        meansI = sample_means(cloudsI)
        phaseI_means_all.append(meansI)
        if n0 is None:
            n0 = meansI.shape[0]

    phaseI_means_all = np.vstack(phaseI_means_all)  # (B*n0, d)
    mu0 = phaseI_means_all.mean(axis=0)

    Sigma0_inv = None
    if d > 1:
        Sigma0 = np.cov(phaseI_means_all, rowvar=False)
        Sigma0 += 1e-8 * np.eye(d)
        Sigma0_inv = np.linalg.inv(Sigma0)

    # --------------------------------------------------------
    # 2) Empirical UCL from pooled Phase-II IC across reps
    # --------------------------------------------------------
    ic_stats_pool = []
    nT = None

    for rep in range(n_rep):
        f_ic = datadir / f"phaseII_IC_d{d}_K{K}_rep{rep:02d}.npz"
        clouds_ic = load_clouds(f_ic, "streams")
        means_ic = sample_means(clouds_ic)
        if nT is None:
            nT = means_ic.shape[0]

        if d == 1:
            stat_ic = means_ic[:, 0]
        else:
            stat_ic = hotelling_T2(means_ic, mu0, Sigma0_inv)
        ic_stats_pool.append(stat_ic)

    ic_stats_pool = np.concatenate(ic_stats_pool)

    if d == 1:
        # two-sided for X-bar with overall alpha
        UCL = float(np.quantile(ic_stats_pool, 1 - alpha / 2))
        LCL = float(np.quantile(ic_stats_pool, alpha / 2))
    else:
        # one-sided for T² (nonnegative)
        UCL = float(np.quantile(ic_stats_pool, 1 - alpha))
        LCL = 0.0

    # Save limits
    limits = dict(
        method=method,
        cfg=cfg_tag,
        alpha=float(alpha),
        UCL=float(UCL),
        LCL=float(LCL),
        mu0=mu0.tolist(),
        n_rep=int(n_rep),
        n0=int(n0 or 0),
        nT=int(nT or 0),
    )
    (out_dir / "limits.json").write_text(json.dumps(limits, indent=2))
    pd.DataFrame([{"method": method, "cfg": cfg_tag, "alpha": alpha, "UCL": UCL}]).to_csv(
        out_dir / "limits.csv", index=False
    )

    print(f"    [{method}] Empirical UCL from pooled IC Phase-II: {UCL:.6f}")

    # --------------------------------------------------------
    # 3) Per-rep scoring + saving + plotting
    # --------------------------------------------------------
    for rep in range(n_rep):
        rep_tag = f"rep{rep:02d}"

        # Phase I series (stat values)
        phaseI_file = datadir / f"phaseI_d{d}_K{K}_{rep_tag}.npz"
        cloudsI = load_clouds(phaseI_file, "phaseI")
        meansI = sample_means(cloudsI)
        statI = meansI[:, 0] if d == 1 else hotelling_T2(meansI, mu0, Sigma0_inv)

        # IC Phase II (for ARL0 rows if desired)
        phaseII_ic_file = datadir / f"phaseII_IC_d{d}_K{K}_{rep_tag}.npz"
        clouds_ic = load_clouds(phaseII_ic_file, "streams")
        means_ic = sample_means(clouds_ic)
        statII_ic = means_ic[:, 0] if d == 1 else hotelling_T2(means_ic, mu0, Sigma0_inv)

        # Save full IC series table (Phase I + II) to CSV
        df_ic_full = pd.DataFrame({
            "idx": np.arange(1, len(statI) + len(statII_ic) + 1),
            "phase": ["I"] * len(statI) + ["II"] * len(statII_ic),
            "stat": np.concatenate([statI, statII_ic]),
        })
        df_ic_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_IC_series.csv", index=False)

        # IC plot optional
        if save_ic_plots:
            plot_mean_control_chart(
                df_full=df_ic_full,
                UCL=UCL,
                LCL = LCL,
                out_png=out_dir / f"{cfg_tag}_{rep_tag}_IC_{method}.png",
                title=f"{method} ({cfg_tag}) IC {rep_tag}",
                ylabel="Xbar" if d == 1 else "T2",
            )

        # IC ARL row (optional but consistent with your OT/KDE tables)
        RL_ic, n_ic = compute_RL_and_alarms_two_sided(statII_ic, UCL=UCL, LCL=LCL)

        rows.append(dict(
            method=method, bandwidth="NA",
            cfg=cfg_tag, d=d, K=K, rep=rep,
            scenario="IC", stat="Xbar" if d == 1 else "T2",
            ARL=RL_ic, n_alarm=n_ic, runtime=0.0, H=len(statII_ic),
        ))

        # OC scenarios: score + save + plot + rows
        for scen in scenarios:
            if scen == "IC":
                continue

            f_oc = datadir / f"phaseII_{scen}_d{d}_K{K}_{rep_tag}.npz"
            clouds_oc = load_clouds(f_oc, "streams")
            means_oc = sample_means(clouds_oc)
            statII_oc = means_oc[:, 0] if d == 1 else hotelling_T2(means_oc, mu0, Sigma0_inv)

            df_oc_full = pd.DataFrame({
                "idx": np.arange(1, len(statI) + len(statII_oc) + 1),
                "phase": ["I"] * len(statI) + ["II"] * len(statII_oc),
                "stat": np.concatenate([statI, statII_oc]),
            })
            df_oc_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_{scen}_series.csv", index=False)

            plot_mean_control_chart(
                df_full=df_oc_full,
                UCL=UCL,
                LCL = LCL,
                out_png=out_dir / f"{cfg_tag}_{rep_tag}_{scen}_{method}.png",
                title=f"{method} ({cfg_tag}) {scen} {rep_tag}",
                ylabel="Xbar" if d == 1 else "T2",
            )

            RL_oc, n_oc = compute_RL_and_alarms_two_sided(statII_oc, UCL, LCL)
            rows.append(dict(
                method=method, bandwidth="NA",
                cfg=cfg_tag, d=d, K=K, rep=rep,
                scenario=scen, stat="Xbar" if d == 1 else "T2",
                ARL=RL_oc, n_alarm=n_oc, runtime=0.0, H=len(statII_oc),
            ))

    return rows
