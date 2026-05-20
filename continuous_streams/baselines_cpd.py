# -*- coding: utf-8 -*-
"""Batch-stream baselines for online change-point detection (CPD).

Your simulation produces a *stream of batches* (a.k.a. distribution-valued samples):
  X_t = {x_{t,1}, ..., x_{t,n_points}} in R^d.

Most "traditional" online CPD algorithms expect a stream of vectors y_t in R^p.
A standard adaptation is therefore:

  (1) choose a batch-to-vector map psi: R^{n_points x d} -> R^p
      e.g., mean, (mean,cov), random-feature mean embedding, etc.
  (2) apply the CPD algorithm to the vector stream y_t := psi(X_t).

This module provides two concrete baselines compatible with your existing
Phase-I/Phase-II .npz format:
  - NEWMA on batch embeddings
  - Scan-B (kernel MMD) on batch embeddings
  - Fréchet CPD (Dubey & Müller-style statistic) on batch embeddings

It follows the output conventions used in your baselines.py:
  returns a list of row dicts with keys
    {method, bandwidth, cfg, d, K, rep, scenario, stat, ARL, n_alarm, runtime, H}
  and saves per-rep series CSVs + plots + a limits.json.

Dependencies: numpy, pandas, matplotlib, scipy (already used by onlinecp.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

# Reuse implementations you already vendor-copied.
from onlinecp import Newma, ScanB  # noqa: F401


# -------------------------
# IO helpers (same as baselines.py)
# -------------------------

def load_clouds(npz_file: Path, key: str) -> List[np.ndarray]:
    z = np.load(npz_file, allow_pickle=True)
    return [np.asarray(x, float) for x in z[key]]


def compute_RL_and_alarms(x: np.ndarray, UCL: float) -> Tuple[int, int]:
    x = np.asarray(x)
    idx = np.where(x > UCL)[0]
    if len(idx) == 0:
        return int(len(x) + 1), 0
    return int(idx[0] + 1), int(len(idx))


# -------------------------
# Plot helper (keeps style close to your mean-chart plots)
# -------------------------

def _plot_series_one_sided(
    ax,
    x,
    y,
    *,
    nI: int,
    UCL: float | None,
    ylabel: str,
    title: str,
    mark_phase2_only: bool = True,
    shade_phaseI: bool = True,
    grid: bool = True,
):
    """
    Fancy one-sided CPD/control-chart style plot:
      - Phase I: grey (optionally shaded background)
      - Phase II: black
      - Mark y > UCL in red (by default Phase-II only)
      - Horizontal UCL line + vertical Phase separator
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0:
        return

    # Split
    xI, yI = x[:nI], y[:nI]
    xII, yII = x[nI:], y[nI:]

    # Optional shading for Phase I region
    if shade_phaseI and nI > 0:
        ax.axvspan(x.min(), xI.max(), color="#eeeeee", alpha=0.6, zorder=0)

    # Base style
    if grid:
        ax.grid(True, which="major", axis="both", linewidth=0.6, alpha=0.25, zorder=0)

    # Phase I: grey
    if nI > 0:
        ax.plot(xI, yI, "-", color="#9e9e9e", linewidth=1.2, zorder=2)
        ax.scatter(xI, yI, s=16, color="#bdbdbd", edgecolor="none", zorder=3)

    # Phase II: black
    if xII.size > 0:
        ax.plot(xII, yII, "-", color="black", linewidth=1.2, zorder=2)
        ax.scatter(xII, yII, s=16, color="black", edgecolor="none", zorder=3)

    # Control limit line
    if UCL is not None:
        ax.axhline(UCL, color="black", linewidth=1.4, zorder=1)

    # Phase separator (between nI and nI+1)
    if nI > 0 and xII.size > 0:
        ax.axvline(xI.max() + 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)
    elif nI > 0:
        # fallback if only Phase I is present
        ax.axvline(xI.max() + 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

    # Out-of-control points: red (Phase-II only by default)
    if UCL is not None:
        if mark_phase2_only:
            mask_red = (x > nI) & (y > UCL)
        else:
            mask_red = (y > UCL)

        if np.any(mask_red):
            ax.scatter(
                x[mask_red],
                y[mask_red],
                s=44,
                color="red",
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )

    # Labels + limits
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(float(x.min()), float(x.max()))

    # Improve y-limits so UCL/peaks are not jammed against the border
    y_valid = y[np.isfinite(y)]
    if y_valid.size > 0:
        y_min = float(np.min(y_valid))
        y_max = float(np.max(y_valid))
        pad = 0.08 * (y_max - y_min + 1e-12)
        ax.set_ylim(y_min - pad, y_max + pad)


def plot_cpd_chart(
    df_full: pd.DataFrame,
    UCL: float,
    out_png: Path,
    title: str,
    ylabel: str = "Statistic",
    *,
    mark_phase2_only: bool = True,
    shade_phaseI: bool = True,
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    nI = int((df_full["phase"] == "I").sum())
    x = df_full["idx"].to_numpy()
    y = df_full["stat"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=220)
    _plot_series_one_sided(
        ax, x, y,
        nI=nI,
        UCL=UCL,
        ylabel=ylabel,
        title=title,
        mark_phase2_only=mark_phase2_only,
        shade_phaseI=shade_phaseI,
        grid=True,
    )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# -------------------------
# Feature maps: batch -> vector
# -------------------------


def _as2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return X


def batch_mean(X: np.ndarray) -> np.ndarray:
    X = _as2d(X)
    return X.mean(axis=0)



def batch_n_points(X: np.ndarray) -> np.ndarray:
    """Return the batch size (number of points) as a 1D feature.

    Useful when the underlying change is primarily in the *counting process* (e.g., Poisson rate shift),
    or when batch size varies materially across time.
    """
    X = _as2d(X)
    return np.array([float(X.shape[0])], float)


def batch_mean_plus_n_points(X: np.ndarray) -> np.ndarray:
    """Concatenate [n_points, mean(X)].

    This is often a stronger baseline than mean alone when batches have variable sizes, including
    Poisson point-process settings.
    """
    X = _as2d(X)
    return np.concatenate([np.array([float(X.shape[0])], float), X.mean(axis=0)], axis=0)

def _vech(S: np.ndarray) -> np.ndarray:
    """Upper-triangular vectorization (incl diagonal)."""
    S = np.asarray(S, float)
    iu = np.triu_indices(S.shape[0])
    return S[iu]


def batch_mean_cov(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = _as2d(X)
    mu = X.mean(axis=0)
    if X.shape[1] == 1:
        v = float(np.var(X[:, 0], ddof=1)) if X.shape[0] > 1 else 0.0
        cov_vech = np.array([v + eps], float)
    else:
        S = np.cov(X, rowvar=False, ddof=1)
        S = np.asarray(S, float) + eps * np.eye(S.shape[0])
        cov_vech = _vech(S)
    return np.concatenate([mu, cov_vech], axis=0)


def median_heuristic_sigma(points: np.ndarray, max_points: int = 512, seed: int = 0) -> float:
    """Median heuristic on Euclidean distances (subsampled for cost)."""
    X = _as2d(points)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    m = min(max_points, n)
    idx = rng.choice(n, size=m, replace=False)
    Z = X[idx]

    # pairwise squared distances via Gram trick
    s = np.sum(Z * Z, axis=1)
    D2 = s[:, None] + s[None, :] - 2.0 * (Z @ Z.T)
    D2 = np.maximum(D2, 0.0)

    # upper triangle, exclude zeros
    iu = np.triu_indices(m, k=1)
    vals = D2[iu]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(np.median(vals)))


@dataclass
class RFFParams:
    W: np.ndarray  # (m, d)
    b: np.ndarray  # (m,)


def build_rff_params(
    pooled_points: np.ndarray,
    m: int = 256,
    seed: int = 0,
    sigma: Optional[float] = None,
    max_points_sigma: int = 512,
) -> Tuple[RFFParams, float]:
    """Random Fourier Features for an RBF kernel with bandwidth sigma."""
    X = _as2d(pooled_points)
    d = X.shape[1]

    sigma_used = float(sigma) if sigma is not None else median_heuristic_sigma(X, max_points=max_points_sigma, seed=seed)
    sigma_used = max(sigma_used, 1e-8)

    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=1.0 / sigma_used, size=(m, d))
    b = rng.uniform(0.0, 2.0 * np.pi, size=(m,))
    return RFFParams(W=W.astype(float), b=b.astype(float)), sigma_used


def batch_rff_mean_embedding(X: np.ndarray, rff: RFFParams) -> np.ndarray:
    """Mean of cos/sin random features over points in the batch."""
    X = _as2d(X)
    Z = X @ rff.W.T + rff.b[None, :]
    c = np.cos(Z).mean(axis=0)
    s = np.sin(Z).mean(axis=0)
    # scale so dot product approximates RBF kernel
    m = rff.W.shape[0]
    return np.sqrt(2.0 / m) * np.concatenate([c, s], axis=0)


# -------------------------
# CPD baseline runners
# -------------------------


def _collect_phaseI_pooled_points(datadir: Path, d: int, K: int, n_rep: int, max_batches: int = 20) -> np.ndarray:
    """Pool points from a few Phase-I batches across reps (subsampled for speed)."""
    pts = []
    for rep in range(n_rep):
        fI = datadir / f"phaseI_d{d}_K{K}_rep{rep:02d}.npz"
        cloudsI = load_clouds(fI, "phaseI")
        for b in cloudsI[:max_batches]:
            pts.append(np.asarray(b, float))
    return np.vstack(pts) if pts else np.zeros((0, d), float)


def _embed_stream(clouds: Iterable[np.ndarray], psi: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    feats = [np.asarray(psi(c), float).reshape(-1) for c in clouds]
    return np.stack(feats, axis=0)  # (T, p)


def _run_newma_dist(
    Y: np.ndarray,
    lam_slow: float,
    lam_fast: float,
    init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return NEWMA distance series; no thresholding here."""
    Y = np.asarray(Y, float)
    p = Y.shape[1]
    s1 = np.zeros(p, float) if init is None else np.asarray(init, float).copy()
    s2 = s1.copy()

    out = np.zeros(Y.shape[0], float)
    for t in range(Y.shape[0]):
        y = Y[t]
        s1 = (1.0 - lam_slow) * s1 + lam_slow * y
        s2 = (1.0 - lam_fast) * s2 + lam_fast * y
        out[t] = float(np.linalg.norm(s1 - s2))
    return out


def compute_newma_batch_baseline(
    datadir: Path,
    resdir: Path,
    d: int,
    K: int,
    scenarios: Iterable[str],
    n_rep: int,
    *,
    feature: str = "rff",  # "mean" | "mean_cov" | "rff" | "n_points" | "mean_n_points"
    rff_m: int = 256,
    rff_seed: int = 0,
    lam_slow: float = 0.01,
    lam_fast: float = 0.05,
    alpha: float = 0.05,
    save_ic_plots: bool = False,
) -> List[Dict]:
    """NEWMA baseline on batch embeddings (Phase-I UCL, Phase-II monitoring).

    Design:
      - Phase I: estimate IC center mu0 per rep, compute NEWMA distance series on centered embeddings,
        pool (post burn-in) distances across reps, set UCL = quantile_{1-alpha}.
      - Phase II: continue using the same Phase-I center mu0 (per rep) and apply NEWMA to centered
        Phase-II streams; compute RL/alarms on Phase-II only.
    """
    datadir = Path(datadir)
    resdir = Path(resdir)

    cfg_tag = f"d{d}_K{K}"

    # ---- choose feature map ----
    bandwidth_used = "NA"
    if feature == "mean":
        psi = batch_mean
        p = d
    elif feature == "n_points":
        psi = batch_n_points
        p = 1
    elif feature == "mean_n_points":
        psi = batch_mean_plus_n_points
        p = d + 1
    elif feature == "mean_cov":
        psi = batch_mean_cov
        p = d + (d * (d + 1)) // 2
    elif feature == "rff":
        pooled = _collect_phaseI_pooled_points(datadir, d=d, K=K, n_rep=n_rep, max_batches=10)
        rff_params, sigma_used = build_rff_params(pooled, m=rff_m, seed=rff_seed)
        bandwidth_used = float(sigma_used)
        psi = partial(batch_rff_mean_embedding, rff=rff_params)
        p = 2 * rff_m
    else:
        raise ValueError("feature must be one of: 'mean', 'n_points', 'mean_n_points', 'mean_cov', 'rff'")

    method = f"NEWMA_{feature}"
    out_dir = resdir / f"phase_{cfg_tag}" / method
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    # --------------------------------------------------------
    # 1) Empirical UCL from pooled Phase-I distances across reps
    # --------------------------------------------------------
    ic_dist_pool = []
    n0 = None
    nT = None

    for rep in range(n_rep):
        rep_tag = f"rep{rep:02d}"

        fI = datadir / f"phaseI_d{d}_K{K}_{rep_tag}.npz"
        fIC = datadir / f"phaseII_IC_d{d}_K{K}_{rep_tag}.npz"
        cloudsI = load_clouds(fI, "phaseI")
        cloudsIC = load_clouds(fIC, "streams")

        if n0 is None:
            n0 = len(cloudsI)
        if nT is None:
            nT = len(cloudsIC)

        # Phase-I embeddings and IC centering (per rep)
        YI = _embed_stream(cloudsI, psi)
        mu0 = YI.mean(axis=0)

        # Center Phase-I, run NEWMA with init at 0 (consistent, removes big start-up hump)
        YI_c = YI - mu0
        distI = _run_newma_dist(YI_c, lam_slow=lam_slow, lam_fast=lam_fast, init=np.zeros_like(mu0))

        # Burn-in: discard early transient (safe guard so we don't discard all)
        burn = min(20, max(0, len(distI) // 10))
        burn = min(burn, max(0, len(distI) - 1))

        ic_dist_pool.append(distI[burn:])

    ic_dist_pool = np.concatenate(ic_dist_pool, axis=0)
    UCL = float(np.quantile(ic_dist_pool, 1.0 - alpha))

    limits = dict(
        method=method,
        cfg=cfg_tag,
        alpha=float(alpha),
        UCL=float(UCL),
        feature=feature,
        feature_dim=int(p),
        lam_slow=float(lam_slow),
        lam_fast=float(lam_fast),
        bandwidth_used=bandwidth_used,
        n_rep=int(n_rep),
        n0=int(n0 or 0),
        nT=int(nT or 0),
        ucl_source="PhaseI_pooled_post_burnin",
    )
    (out_dir / "limits.json").write_text(json.dumps(limits, indent=2))

    print(f"    [{method}] Empirical UCL from pooled Phase-I (post burn-in): {UCL:.6f}")

    # --------------------------------------------------------
    # 2) Per-rep scoring + saving + plotting
    # --------------------------------------------------------
    for rep in range(n_rep):
        rep_tag = f"rep{rep:02d}"

        # Phase I
        fI = datadir / f"phaseI_d{d}_K{K}_{rep_tag}.npz"
        cloudsI = load_clouds(fI, "phaseI")
        YI = _embed_stream(cloudsI, psi)
        mu0 = YI.mean(axis=0)
        YI_c = YI - mu0

        # IC Phase II
        fIC = datadir / f"phaseII_IC_d{d}_K{K}_{rep_tag}.npz"
        cloudsIC = load_clouds(fIC, "streams")
        YIC = _embed_stream(cloudsIC, psi)
        YIC_c = YIC - mu0

        # Run on concatenated centered stream (consistent with UCL calibration)
        dist_all_ic = _run_newma_dist(
            np.vstack([YI_c, YIC_c]),
            lam_slow=lam_slow,
            lam_fast=lam_fast,
            init=np.zeros_like(mu0),
        )
        distI = dist_all_ic[: len(YI_c)]
        distII_ic = dist_all_ic[len(YI_c) :]

        # Save full IC series
        df_ic_full = pd.DataFrame({
            "idx": np.arange(1, len(distI) + len(distII_ic) + 1),
            "phase": ["I"] * len(distI) + ["II"] * len(distII_ic),
            "stat": np.concatenate([distI, distII_ic]),
        })
        df_ic_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_IC_series.csv", index=False)

        if save_ic_plots:
            plot_cpd_chart(
                df_full=df_ic_full,
                UCL=UCL,
                out_png=out_dir / f"{cfg_tag}_{rep_tag}_IC_{method}.png",
                title=f"{method} ({cfg_tag}) IC {rep_tag}",
                ylabel="NEWMA distance",
            )

        RL_ic, n_ic = compute_RL_and_alarms(distII_ic, UCL)
        rows.append(dict(
            method=method,
            bandwidth=str(bandwidth_used),
            cfg=cfg_tag,
            d=d,
            K=K,
            rep=rep,
            scenario="IC",
            stat="dist",
            ARL=RL_ic,
            n_alarm=n_ic,
            runtime=0.0,
            H=len(distII_ic),
        ))

        # OC scenarios
        for scen in scenarios:
            if scen == "IC":
                continue

            fOC = datadir / f"phaseII_{scen}_d{d}_K{K}_{rep_tag}.npz"
            cloudsOC = load_clouds(fOC, "streams")
            YOC = _embed_stream(cloudsOC, psi)
            YOC_c = YOC - mu0

            dist_all_oc = _run_newma_dist(
                np.vstack([YI_c, YOC_c]),
                lam_slow=lam_slow,
                lam_fast=lam_fast,
                init=np.zeros_like(mu0),
            )
            distII_oc = dist_all_oc[len(YI_c) :]

            df_oc_full = pd.DataFrame({
                "idx": np.arange(1, len(distI) + len(distII_oc) + 1),
                "phase": ["I"] * len(distI) + ["II"] * len(distII_oc),
                "stat": np.concatenate([distI, distII_oc]),
            })
            df_oc_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_{scen}_series.csv", index=False)

            plot_cpd_chart(
                df_full=df_oc_full,
                UCL=UCL,
                out_png=out_dir / f"{cfg_tag}_{rep_tag}_{scen}_{method}.png",
                title=f"{method} ({cfg_tag}) {scen} {rep_tag}",
                ylabel="NEWMA distance",
            )

            RL_oc, n_oc = compute_RL_and_alarms(distII_oc, UCL)
            rows.append(dict(
                method=method,
                bandwidth=str(bandwidth_used),
                cfg=cfg_tag,
                d=d,
                K=K,
                rep=rep,
                scenario=str(scen),
                stat="dist",
                ARL=RL_oc,
                n_alarm=n_oc,
                runtime=0.0,
                H=len(distII_oc),
            ))

    return rows


def compute_scanb_batch_baseline(
    datadir: Path,
    resdir: Path,
    d: int,
    K: int,
    scenarios: Iterable[str],
    n_rep: int,
    *,
    feature: str = "rff",  # "mean" | "mean_cov" | "rff"
    rff_m: int = 256,
    rff_seed: int = 0,
    B: int = 10,
    N: int = 5,
    alpha: float = 0.05,
    save_ic_plots: bool = False,
) -> List[Dict]:
    """Scan-B baseline (kernel MMD) on batch embeddings.

    We use the *linear* kernel on the embedding vectors. For feature=='rff',
    this approximates an RBF kernel between the underlying distributions.

    Thresholding: empirical UCL from pooled Phase-II IC distance series across reps.
    """
    warmup = int((N + 1) * B)  # number of batches before Scan-B statistic is meaningful

    datadir = Path(datadir)
    resdir = Path(resdir)

    cfg_tag = f"d{d}_K{K}"

    bandwidth_used = "NA"
    if feature == "mean":
        psi = batch_mean
        p = d
    elif feature == "mean_cov":
        psi = batch_mean_cov
        p = d + (d * (d + 1)) // 2
    elif feature == "rff":
        pooled = _collect_phaseI_pooled_points(datadir, d=d, K=K, n_rep=n_rep, max_batches=10)
        rff_params, sigma_used = build_rff_params(pooled, m=rff_m, seed=rff_seed)
        bandwidth_used = float(sigma_used)

        psi = partial(batch_rff_mean_embedding, rff=rff_params)

        p = 2 * rff_m
    else:
        raise ValueError("feature must be one of: 'mean', 'mean_cov', 'rff'")

    method = f"ScanB_{feature}"
    out_dir = resdir / f"phase_{cfg_tag}" / method
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    # --------------------------------------------------------
    # 1) Empirical UCL from pooled Phase-II IC across reps
    # --------------------------------------------------------
    ic_dist_pool = []
    n0 = None
    nT = None

    for rep in range(n_rep):
        fI = datadir / f"phaseI_d{d}_K{K}_rep{rep:02d}.npz"
        fIC = datadir / f"phaseII_IC_d{d}_K{K}_rep{rep:02d}.npz"
        cloudsI = load_clouds(fI, "phaseI")
        cloudsIC = load_clouds(fIC, "streams")

        if n0 is None:
            n0 = len(cloudsI)
        if nT is None:
            nT = len(cloudsIC)

        YI = _embed_stream(cloudsI, psi)
        YIC = _embed_stream(cloudsIC, psi)
        Y_all = np.vstack([YI, YIC])

        if Y_all.shape[0] < (N + 1) * B:
            raise ValueError(
                f"Scan-B needs at least (N+1)*B={(N+1)*B} batches; got {Y_all.shape[0]}. "
                f"Increase n0 or reduce B/N."
            )

        scan = ScanB(d=p, B=B, N=N, store_result=True)
        scan.apply_to_data(Y_all)
        dist_all = np.asarray(scan.dist, float)

        # mark warm-up as undefined (avoid artificial zeros in plots/UCL)
        dist_all[:warmup] = np.nan

        distII = dist_all[len(YI):]
        distII = distII[np.isfinite(distII)]  # drop NaNs if warm-up spills into Phase II
        ic_dist_pool.append(distII)

    ic_dist_pool = np.concatenate(ic_dist_pool, axis=0) if ic_dist_pool else np.zeros((0,), float)
    ic_dist_pool = ic_dist_pool[np.isfinite(ic_dist_pool)]
    UCL = float(np.quantile(ic_dist_pool, 1.0 - alpha)) if ic_dist_pool.size > 0 else float("inf")

    limits = dict(
        method=method,
        cfg=cfg_tag,
        alpha=float(alpha),
        UCL=float(UCL),
        feature=feature,
        feature_dim=int(p),
        B=int(B),
        N=int(N),
        bandwidth_used=bandwidth_used,
        n_rep=int(n_rep),
        n0=int(n0 or 0),
        nT=int(nT or 0),
    )
    (out_dir / "limits.json").write_text(json.dumps(limits, indent=2))

    print(f"    [{method}] Empirical UCL from pooled IC Phase-II: {UCL:.6f}")

    # --------------------------------------------------------
    # 2) Per-rep scoring + saving + plotting
    # --------------------------------------------------------
    for rep in range(n_rep):
        rep_tag = f"rep{rep:02d}"

        fI = datadir / f"phaseI_d{d}_K{K}_{rep_tag}.npz"
        cloudsI = load_clouds(fI, "phaseI")
        YI = _embed_stream(cloudsI, psi)

        fIC = datadir / f"phaseII_IC_d{d}_K{K}_{rep_tag}.npz"
        cloudsIC = load_clouds(fIC, "streams")
        YIC = _embed_stream(cloudsIC, psi)

        Y_all_ic = np.vstack([YI, YIC])
        scan_ic = ScanB(d=p, B=B, N=N, store_result=True)
        scan_ic.apply_to_data(Y_all_ic)

        dist_all_ic = np.asarray(scan_ic.dist, float)
        dist_all_ic[:warmup] = np.nan

        distI = dist_all_ic[: len(YI)]
        distII_ic = dist_all_ic[len(YI):]

        df_ic_full = pd.DataFrame({
            "idx": np.arange(1, len(distI) + len(distII_ic) + 1),
            "phase": ["I"] * len(distI) + ["II"] * len(distII_ic),
            "stat": np.concatenate([distI, distII_ic]),
        })
        df_ic_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_IC_series.csv", index=False)

        if save_ic_plots:
            plot_cpd_chart(
                df_full=df_ic_full,
                UCL=UCL,
                out_png=out_dir / f"{cfg_tag}_{rep_tag}_IC_{method}.png",
                title=f"{method} ({cfg_tag}) IC {rep_tag}",
                ylabel="Scan-B MMD",
            )

        distII_ic_valid = distII_ic[np.isfinite(distII_ic)]
        RL_ic, n_ic = compute_RL_and_alarms(distII_ic_valid, UCL)

        rows.append(dict(
            method=method,
            bandwidth=str(bandwidth_used),
            cfg=cfg_tag,
            d=d,
            K=K,
            rep=rep,
            scenario="IC",
            stat="dist",
            ARL=RL_ic,
            n_alarm=n_ic,
            runtime=0.0,
            H=len(distII_ic_valid),
        ))

        for scen in scenarios:
            if scen == "IC":
                continue

            fOC = datadir / f"phaseII_{scen}_d{d}_K{K}_{rep_tag}.npz"
            cloudsOC = load_clouds(fOC, "streams")
            YOC = _embed_stream(cloudsOC, psi)

            Y_all_oc = np.vstack([YI, YOC])
            scan_oc = ScanB(d=p, B=B, N=N, store_result=True)
            scan_oc.apply_to_data(Y_all_oc)

            dist_all_oc = np.asarray(scan_oc.dist, float)
            dist_all_oc[:warmup] = np.nan

            distII_oc = dist_all_oc[len(YI):]
            distII_oc_valid = distII_oc[np.isfinite(distII_oc)]

            df_oc_full = pd.DataFrame({
                "idx": np.arange(1, len(distI) + len(distII_oc) + 1),
                "phase": ["I"] * len(distI) + ["II"] * len(distII_oc),
                "stat": np.concatenate([distI, distII_oc]),
            })
            df_oc_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_{scen}_series.csv", index=False)

            plot_cpd_chart(
                df_full=df_oc_full,
                UCL=UCL,
                out_png=out_dir / f"{cfg_tag}_{rep_tag}_{scen}_{method}.png",
                title=f"{method} ({cfg_tag}) {scen} {rep_tag}",
                ylabel="Scan-B MMD",
            )

            RL_oc, n_oc = compute_RL_and_alarms(distII_oc_valid, UCL)

            rows.append(dict(
                method=method,
                bandwidth=str(bandwidth_used),
                cfg=cfg_tag,
                d=d,
                K=K,
                rep=rep,
                scenario=str(scen),
                stat="dist",
                ARL=RL_oc,
                n_alarm=n_oc,
                runtime=0.0,
                H=len(distII_oc_valid),
            ))

    return rows


# -------------------------
# Fréchet CPD baseline (Dubey & Müller-style statistic in an embedding space)
# -------------------------

def _flatten_objects(X: np.ndarray) -> np.ndarray:
    """Ensure shape (n, p) by flattening each object to a vector."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        return X[:, None]
    if X.ndim == 2:
        return X
    return X.reshape((X.shape[0], -1))


def frechet_stat_scan(x: np.ndarray, *, c: float = 0.1, eps: float = 1e-12) -> float:
    """Compute a windowed Fréchet CPD statistic and scan over split points.

    This is a Euclidean/Hilbert-space instantiation (Fréchet mean = arithmetic mean).
    For a length-L segment, scan t in [ceil(cL), floor((1-c)L)] and take the maximum.
    """
    X = _flatten_objects(x)
    L = X.shape[0]
    if L < 4:
        return 0.0

    t_min = int(np.ceil(c * L))
    t_max = int(np.floor((1.0 - c) * L))
    if t_max <= t_min:
        t_min, t_max = 1, L - 1

    m = X.mean(axis=0)
    dsq = np.sum((X - m) ** 2, axis=1)
    sigma = float(np.mean(dsq ** 2) - (np.mean(dsq) ** 2))
    sigma = max(sigma, eps)

    best = 0.0
    for t in range(t_min, t_max + 1):
        u = t / L
        X0, X1 = X[:t], X[t:]
        if X0.shape[0] < 2 or X1.shape[0] < 2:
            continue

        m0 = X0.mean(axis=0)
        m1 = X1.mean(axis=0)

        V0 = float(np.mean(np.sum((X0 - m0) ** 2, axis=1)))
        V1 = float(np.mean(np.sum((X1 - m1) ** 2, axis=1)))
        V0c = float(np.mean(np.sum((X0 - m1) ** 2, axis=1)))
        V1c = float(np.mean(np.sum((X1 - m0) ** 2, axis=1)))

        add_factor = (V0c - V0) + (V1c - V1)
        stat = float(u * (1.0 - u) * (((V0 - V1) ** 2) + (add_factor ** 2)) / sigma)
        if stat > best:
            best = stat

    return float(best)


def frechet_cpd_series(Y: np.ndarray, *, swl: int = 64, c: float = 0.1) -> np.ndarray:
    """Online-style Fréchet CPD score with a sliding window of length 2*swl.

    The score is undefined for the first 2*swl points; we return NaN there
    (instead of 0) to avoid misleading plots and biased UCL estimation.
    """
    Y = np.asarray(Y, float)
    T = Y.shape[0]
    win = 2 * int(swl)

    out = np.full(T, np.nan, dtype=float)   # <-- change: NaN warm-up region
    for i in range(win, T):
        seg = Y[i - win : i]
        out[i] = frechet_stat_scan(seg, c=c)
    return out


def compute_frechet_batch_baseline(
    datadir: Path,
    resdir: Path,
    d: int,
    K: int,
    scenarios: Iterable[str],
    n_rep: int,
    *,
    feature: str = "rff",
    rff_m: int = 256,
    rff_seed: int = 0,
    swl: int = 64,
    scan_c: float = 0.1,
    alpha: float = 0.05,
) -> List[Dict]:
    """Fréchet CPD baseline on batch embeddings (one-sided UCL)."""
    datadir = Path(datadir)
    resdir = Path(resdir)

    cfg_tag = f"d{d}_K{K}"

    bandwidth_used = "NA"
    if feature == "mean":
        psi = batch_mean
        p = d
    elif feature == "n_points":
        psi = batch_n_points
        p = 1
    elif feature == "mean_n_points":
        psi = batch_mean_plus_n_points
        p = d + 1
    elif feature == "mean_cov":
        psi = batch_mean_cov
        p = d + (d * (d + 1)) // 2
    elif feature == "rff":
        pooled = _collect_phaseI_pooled_points(datadir, d=d, K=K, n_rep=n_rep, max_batches=10)
        rff_params, sigma_used = build_rff_params(pooled, m=rff_m, seed=rff_seed)
        bandwidth_used = float(sigma_used)
        psi = partial(batch_rff_mean_embedding, rff=rff_params)

        p = 2 * rff_m
    else:
        raise ValueError("feature must be one of: 'mean', 'n_points', 'mean_n_points', 'mean_cov', 'rff'")

    method = f"FRECHET_{feature}"
    out_dir = resdir / f"phase_{cfg_tag}" / method
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    warm = 2 * int(swl)

    # ---- UCL from pooled IC phase-II
    ic_pool = []
    n0 = None
    nT = None
    for rep in range(n_rep):
        rep_tag = f"rep{rep:02d}"
        fI = datadir / f"phaseI_d{d}_K{K}_{rep_tag}.npz"
        fIC = datadir / f"phaseII_IC_d{d}_K{K}_{rep_tag}.npz"
        cloudsI = load_clouds(fI, "phaseI")
        cloudsIC = load_clouds(fIC, "streams")

        if n0 is None:
            n0 = len(cloudsI)
        if nT is None:
            nT = len(cloudsIC)

        YI = _embed_stream(cloudsI, psi)
        YIC = _embed_stream(cloudsIC, psi)

        score_all = frechet_cpd_series(np.vstack([YI, YIC]), swl=swl, c=scan_c)
        scoreII = score_all[len(YI) :]

        start = max(0, warm - len(YI))
        ic_pool.append(scoreII[start:])

    ic_pool = np.concatenate(ic_pool, axis=0) if ic_pool else np.zeros((0,), float)
    ic_pool = ic_pool[np.isfinite(ic_pool)]  # drop NaNs from warm-up

    UCL = float(np.quantile(ic_pool, 1.0 - alpha)) if ic_pool.size > 0 else float("inf")

    limits = dict(
        method=method, cfg=cfg_tag, alpha=float(alpha), UCL=float(UCL),
        feature=feature, feature_dim=int(p), swl=int(swl), scan_c=float(scan_c),
        bandwidth_used=bandwidth_used, n_rep=int(n_rep), n0=int(n0 or 0), nT=int(nT or 0),
    )
    (out_dir / "limits.json").write_text(json.dumps(limits, indent=2))
    print(f"    [{method}] Empirical UCL from pooled IC Phase-II: {UCL:.6f}")

    # ---- per-rep scoring
    for rep in range(n_rep):
        rep_tag = f"rep{rep:02d}"

        fI = datadir / f"phaseI_d{d}_K{K}_{rep_tag}.npz"
        cloudsI = load_clouds(fI, "phaseI")
        YI = _embed_stream(cloudsI, psi)

        fIC = datadir / f"phaseII_IC_d{d}_K{K}_{rep_tag}.npz"
        cloudsIC = load_clouds(fIC, "streams")
        YIC = _embed_stream(cloudsIC, psi)

        score_all_ic = frechet_cpd_series(np.vstack([YI, YIC]), swl=swl, c=scan_c)
        scoreI = score_all_ic[: len(YI)]
        scoreII_ic = score_all_ic[len(YI) :]

        df_ic_full = pd.DataFrame({
            "idx": np.arange(1, len(scoreI) + len(scoreII_ic) + 1),
            "phase": ["I"] * len(scoreI) + ["II"] * len(scoreII_ic),
            "stat": np.concatenate([scoreI, scoreII_ic]),
        })
        df_ic_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_IC_series.csv", index=False)
        plot_cpd_chart(df_full=df_ic_full, UCL=UCL,
                       out_png=out_dir / f"{cfg_tag}_{rep_tag}_IC_{method}.png",
                       title=f"{method} ({cfg_tag}) IC {rep_tag}",
                       ylabel="Frechet score")

        scoreII_valid = scoreII_ic[np.isfinite(scoreII_ic)]  # drop NaNs from warm-up
        RL_ic, n_ic = compute_RL_and_alarms(scoreII_valid, UCL)

        rows.append(dict(
            method=method, bandwidth=str(bandwidth_used), cfg=cfg_tag, d=d, K=K, rep=rep,
            scenario="IC", stat="score", ARL=RL_ic, n_alarm=n_ic, runtime=0.0, H=len(scoreII_valid)
        ))


        for scen in scenarios:
            if scen == "IC":
                continue
            fOC = datadir / f"phaseII_{scen}_d{d}_K{K}_{rep_tag}.npz"
            cloudsOC = load_clouds(fOC, "streams")
            YOC = _embed_stream(cloudsOC, psi)

            score_all_oc = frechet_cpd_series(np.vstack([YI, YOC]), swl=swl, c=scan_c)
            scoreII_oc = score_all_oc[len(YI) :]

            df_oc_full = pd.DataFrame({
                "idx": np.arange(1, len(scoreI) + len(scoreII_oc) + 1),
                "phase": ["I"] * len(scoreI) + ["II"] * len(scoreII_oc),
                "stat": np.concatenate([scoreI, scoreII_oc]),
            })
            df_oc_full.to_csv(out_dir / f"{cfg_tag}_{rep_tag}_{scen}_series.csv", index=False)
            plot_cpd_chart(df_full=df_oc_full, UCL=UCL,
                           out_png=out_dir / f"{cfg_tag}_{rep_tag}_{scen}_{method}.png",
                           title=f"{method} ({cfg_tag}) {scen} {rep_tag}",
                           ylabel="Frechet score")

            scoreII_valid = scoreII_oc[np.isfinite(scoreII_oc)]  # drop NaNs from warm-up
            RL_oc, n_oc = compute_RL_and_alarms(scoreII_valid, UCL)
            rows.append(dict(method=method, bandwidth=str(bandwidth_used), cfg=cfg_tag, d=d, K=K, rep=rep,
                             scenario=str(scen), stat="score", ARL=RL_oc, n_alarm=n_oc, runtime=0.0, H=len(scoreII_oc)))


    return rows

