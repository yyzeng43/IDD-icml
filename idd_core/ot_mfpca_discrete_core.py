"""
OT-MFPCA (discrete) core: Phase I/II processing for
  1) 1D counting processes (discrete support on Z_>=0)
  2) Multiclass categorical data (finite support of size m)

Design principles (discrete-specific):
  - Counts (1D): use *quantile barycenter* and *monotone rearrangement* OT map
    (closed-form Wasserstein barycenter/map in 1D).
  - Categorical: work directly on histograms (PMFs) with a user-chosen cost matrix C;
    barycenter via entropic OT barycenter; OT map via coupling -> row-normalized plan.

No flow / SB / Gaussian modes. Two modes only:
  - mode="count_1d"
  - mode="categorical"

Outputs are "tangent features" per time point, suitable for discretized MFPCA:
  - count_1d: tangent is (G, 1)
  - categorical: tangent is (m, m) (domain index = source class; values = R^m)

Requires POT (Python Optimal Transport): pip install POT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import time
import numpy as np
from tqdm import tqdm

import ot  # POT


# ----------------------------
# Helpers
# ----------------------------

def _normalize_weights(a: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    a = np.maximum(a, 0.0)
    s = float(a.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Invalid weights: sum={s}")
    a = a / s
    a = np.maximum(a, eps)
    a = a / a.sum()
    return a


def _quantile_grid(G: int) -> np.ndarray:
    if G <= 1:
        return np.array([0.5], dtype=float)
    return (np.arange(G, dtype=float) + 0.5) / float(G)


def _np_quantile_1d(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("Empty sample.")
    u = np.clip(u, 0.0, 1.0)
    return np.quantile(x, u, method="linear")


def _load_npz_cloud_list(npz_path: Path, key: str) -> List[np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    arr = list(z[key])
    return [np.asarray(a) for a in arr]


def _hist_categorical(labels: np.ndarray, m: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(int)
    labels = labels[(labels >= 0) & (labels < m)]
    w = np.bincount(labels, minlength=m).astype(float)
    return _normalize_weights(w)

# This is for categorical case
def hamming_cost(m: int) -> np.ndarray:
    C = np.ones((m, m), dtype=float)
    np.fill_diagonal(C, 0.0)
    return C

# This is for ordinal values case
def ordinal_cost(m: int, power: float = 2.0) -> np.ndarray:
    """Ordinal ground cost: C_ij = |i-j|^power."""
    idx = np.arange(m, dtype=float)
    D = np.abs(idx[:, None] - idx[None, :])
    return D ** power


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class DiscreteBarycenter:
    mode: str                      # "count_1d" or "categorical"
    Xb: np.ndarray                 # count_1d: (G,1) quantile support; categorical: I_m (m,m)
    wb: np.ndarray                 # count_1d: (G,) uniform; categorical: (m,) barycenter pmf
    meta: Dict                     # mode-specific metadata
    C: Optional[np.ndarray] = None # categorical cost (m,m)
    u: Optional[np.ndarray] = None # count_1d quantile grid (G,)


# ----------------------------
# 1) Fit barycenter from pooled Phase I replications
# ----------------------------

def fit_barycenter_from_phaseI_files(
    phaseI_files: Sequence[Path],
    *,
    mode: str,
    n_bary: int = 256,
    m: Optional[int] = None,
    reg: float = 0.05,
    C: Optional[np.ndarray] = None,
) -> Tuple[DiscreteBarycenter, Dict]:
    """
    Fit a single barycenter from pooled Phase I replications (recommended).
    Returns: (barycenter, timing_dict)
    """
    t0 = time.perf_counter()

    mode = str(mode).lower()
    if mode not in ("count_1d", "categorical", "dust_poisson_1d"):
        raise ValueError("mode must be 'count_1d' or 'categorical'.")

    # Load all Phase I clouds across all reps
    all_phaseI = []
    for f in phaseI_files:
        all_phaseI.extend(_load_npz_cloud_list(f, key="phaseI"))

    if len(all_phaseI) == 0:
        raise ValueError("No Phase I observations found.")

    if mode == "count_1d" or "dust_poisson_1d":
        G = int(n_bary)
        u = _quantile_grid(G)
        # Quantile barycenter: Qbar(u) = mean_t Q_t(u)
        Qs = []
        for Y in all_phaseI:
            y = np.asarray(Y).reshape(-1)
            Qs.append(_np_quantile_1d(y, u))
        Qs = np.stack(Qs, axis=0)               # (n_obs, G)
        Qbar = Qs.mean(axis=0)                  # (G,)
        Xb = Qbar.reshape(-1, 1)
        wb = np.ones(G, dtype=float) / float(G)
        meta = {"mode": mode, "n_bary": G, "n_obs": int(len(all_phaseI))}
        bary = DiscreteBarycenter(mode=mode, Xb=Xb, wb=wb, u=u, meta=meta)

    else:
        # categorical: reduce each cloud to histogram p_t in R^m
        if m is None:
            mx = 0
            for Y in all_phaseI:
                y = np.asarray(Y).reshape(-1).astype(int)
                if y.size:
                    mx = max(mx, int(y.max()))
            m = mx + 1
        m = int(m)
        if m <= 1:
            raise ValueError(f"Invalid m={m}.")

        if C is None:
            C = hamming_cost(m)
        C = np.asarray(C, dtype=float)
        if C.shape != (m, m):
            raise ValueError(f"C must be {(m,m)}, got {C.shape}")

        Ps = np.stack([_hist_categorical(Y, m) for Y in all_phaseI], axis=1)  # (m, n_obs)
        # Uniform weights over observations
        weights = np.ones(Ps.shape[1], dtype=float) / float(Ps.shape[1])
        p_bar = ot.bregman.barycenter(Ps, C, reg=float(reg), weights=weights)
        p_bar = _normalize_weights(p_bar)

        Xb = np.eye(m, dtype=float)
        meta = {"mode": "categorical", "m": m, "reg": float(reg), "n_obs": int(len(all_phaseI))}
        bary = DiscreteBarycenter(mode="categorical", Xb=Xb, wb=p_bar, C=C, meta=meta)

    timing = {"t_barycenter": float(time.perf_counter() - t0)}
    return bary, timing


# ----------------------------
# 2) OT map + tangents for a single Phase I replication file
# ----------------------------

def process_phaseI_discrete(
    file_phaseI: Path,
    bary: DiscreteBarycenter,
    *,
    reg: float = 0.05,
    sinkhorn: bool = True,
    numItermax: int = 5000,
    stopThr: float = 1e-6,
) -> Tuple[np.ndarray, Dict]:
    """
    Given a fitted barycenter, compute tangents for all Phase I time points in one replication.

    Returns:
      tangents: (n0, G, d) where
          - count_1d: (n0, G, 1)
          - categorical: (n0, m, m)
      timing: dict with t_load, t_ot, t_total
    """
    t0 = time.perf_counter()
    phaseI = _load_npz_cloud_list(file_phaseI, key="phaseI")
    t_load = time.perf_counter()

    if bary.mode in ["count_1d", "dust_poisson_1d"]:
        u = bary.u
        Xb = bary.Xb.reshape(-1)  # (G,)
        G = Xb.size
        tang = np.empty((len(phaseI), G, 1), dtype=float)
        for i, Y in enumerate(tqdm(phaseI, desc=f"[OT] Phase I OT maps: {Path(file_phaseI).stem}", leave=False)):
            q = _np_quantile_1d(np.asarray(Y).reshape(-1), u)  # (G,)
            tang[i, :, 0] = q - Xb
        t_ot = time.perf_counter()

    elif bary.mode == "categorical":
        m = int(bary.meta["m"])
        C = np.asarray(bary.C, dtype=float)
        p_src = _normalize_weights(bary.wb)
        tang = np.empty((len(phaseI), m, m), dtype=float)

        for i, Y in enumerate(tqdm(phaseI, desc=f"[OT] Phase I OT maps: {Path(file_phaseI).stem}", leave=False)):
            p_tgt = _hist_categorical(Y, m)
            if sinkhorn:
                Pi = ot.sinkhorn(p_src, p_tgt, C, reg=float(reg),
                                 numItermax=int(numItermax), stopThr=float(stopThr))
            else:
                Pi = ot.emd(p_src, p_tgt, C)
            T = Pi / (p_src[:, None] + 1e-16)  # row-normalized: conditional distribution
            tang[i] = T - np.eye(m)
        t_ot = time.perf_counter()

    else:
        raise ValueError(f"Unknown bary.mode={bary.mode}")

    timing = {
        "t_load": float(t_load - t0),
        "t_ot": float(t_ot - t_load),
        "t_total": float(t_ot - t0),
        "n_obs": int(len(phaseI)),
        "file": str(file_phaseI),
    }
    return tang, timing


# ----------------------------
# 3) OT map + tangents for a single Phase II file
# ----------------------------

def process_phaseII_discrete(
    file_phaseII: Path,
    bary: DiscreteBarycenter,
    *,
    reg: float = 0.05,
    sinkhorn: bool = True,
    numItermax: int = 5000,
    stopThr: float = 1e-6,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute Phase II tangents for all streams in one Phase II file.

    Returns:
      tangents: (nT, G, d) or (nT, m, m)
      timing: dict with t_load, t_ot, t_total
    """
    t0 = time.perf_counter()
    streams = _load_npz_cloud_list(file_phaseII, key="streams")
    t_load = time.perf_counter()

    if bary.mode == "count_1d" or "dust_poisson_1d":
        u = bary.u
        Xb = bary.Xb.reshape(-1)
        G = Xb.size
        tang = np.empty((len(streams), G, 1), dtype=float)
        for i, Y in enumerate(tqdm(streams, desc=f"[OT] Phase II OT maps: {Path(file_phaseII).stem}", leave=False)):
            q = _np_quantile_1d(np.asarray(Y).reshape(-1), u)
            tang[i, :, 0] = q - Xb
        t_ot = time.perf_counter()

    elif bary.mode == "categorical":
        m = int(bary.meta["m"])
        C = np.asarray(bary.C, dtype=float)
        p_src = _normalize_weights(bary.wb)
        tang = np.empty((len(streams), m, m), dtype=float)

        for i, Y in enumerate(tqdm(streams, desc=f"[OT] Phase II OT maps: {Path(file_phaseII).stem}", leave=False)):
            p_tgt = _hist_categorical(Y, m)
            if sinkhorn:
                Pi = ot.sinkhorn(p_src, p_tgt, C, reg=float(reg),
                                 numItermax=int(numItermax), stopThr=float(stopThr))
            else:
                Pi = ot.emd(p_src, p_tgt, C)
            T = Pi / (p_src[:, None] + 1e-16)
            tang[i] = T - np.eye(m)
        t_ot = time.perf_counter()

    else:
        raise ValueError(f"Unknown bary.mode={bary.mode}")

    timing = {
        "t_load": float(t_load - t0),
        "t_ot": float(t_ot - t_load),
        "t_total": float(t_ot - t0),
        "n_obs": int(len(streams)),
        "file": str(file_phaseII),
    }
    return tang, timing


# ----------------------------
# Export helpers for R
# ----------------------------

def export_features_for_r(
    X: np.ndarray,
    out_prefix: Path,
    *,
    meta: Optional[Dict] = None,
) -> Tuple[Path, Dict]:
    """
    Export features as a 2D matrix (n_obs, p) in .npy format so R can read via RcppCNPy.

    X can be 3D (n_obs, G, d). We flatten (G,d) -> p.

    Returns:
      x_path: Path to the .npy file
      info: dict with shapes and p
    """
    t0 = time.perf_counter()
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X2 = X
        G, d = None, None
    elif X.ndim == 3:
        n, G, d = X.shape
        X2 = X.reshape(n, G * d)
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")

    x_path = out_prefix.with_suffix(".npy")
    np.save(x_path, X2)

    info = {
        "x_path": str(x_path),
        "shape_in": list(X.shape),
        "shape_2d": list(X2.shape),
        "p": int(X2.shape[1]),
        "t_export": float(time.perf_counter() - t0),
    }

    if meta is not None:
        meta_path = out_prefix.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        info["meta_path"] = str(meta_path)

    return x_path, info
