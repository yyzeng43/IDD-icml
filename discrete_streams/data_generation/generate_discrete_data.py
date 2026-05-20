"""
Discrete simulation data generator for DIDO_CC control-chart experiments.

Generates Phase I (IC) and Phase II (OC) distribution-valued samples for all
discrete paper scenarios:

Design 1 – Counting process, 1D (Poisson-based):
  (C1) Spike / atom injection at k*
  (C2) Zero-inflation (mass at 0)

Design 2 – Dust-on-screens (Poisson-based, Fig. 2a):
  (D1) Heavy-tail via 2-component Poisson mixture across screens
  (D2) Spike / atom injection at k*

Design 3 – Multiclass categorical, finite support (Fig. 2b):
  (M1) Probability transfer from class a to class b
  (M2) Emergence of a rare class (probability spike)
  (M3) Gradual drift toward a shifted PMF
  (M4) Adjacent-class confusion (blur)

Output .npz structure (all designs):
  - phaseI / streams: object array, each element is (n_points, d)
  - map_params: object array of per-time parameter dicts
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------
# Utilities
# ---------------------------

def _as_col(x: np.ndarray) -> np.ndarray:
    """Ensure (n_points, 1) column shape for d=1."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _save_phaseI(outdir: Path, fname: str, *, d: int, n0: int, n_points: int,
                 phaseI: List[np.ndarray], map_params: List[Dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / fname,
        d=int(d),
        n0=int(n0),
        n_points=int(n_points),
        phaseI=np.array(phaseI, dtype=object),
        map_params=np.array(map_params, dtype=object),
    )


def _save_phaseII(outdir: Path, fname: str, *, d: int, nT: int, n_points: int,
                  streams: List[np.ndarray], map_params: List[Dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / fname,
        d=int(d),
        nT=int(nT),
        n_points=int(n_points),
        streams=np.array(streams, dtype=object),
        map_params=np.array(map_params, dtype=object),
    )


# ---------------------------
# Design 1: Counting process (1D)
# ---------------------------

def _sample_poisson(rng: np.random.Generator, lam: float, n_points: int) -> np.ndarray:
    return rng.poisson(lam=lam, size=n_points)


def _sample_count_spike(
    rng: np.random.Generator,
    lam0: float,
    n_points: int,
    *,
    alpha: float,
    k_star: int,
) -> np.ndarray:
    """
    Mixture: (1-alpha) Poisson(lam0) + alpha * delta_{k_star}.
    """
    base = _sample_poisson(rng, lam0, n_points)
    mask = rng.random(n_points) < float(alpha)
    base[mask] = int(k_star)
    return base


def _sample_count_zeroinfl(
    rng: np.random.Generator,
    lam0: float,
    n_points: int,
    *,
    pi0: float,
) -> np.ndarray:
    """
    Zero-inflated Poisson:
      X=0 w.p. pi0, else Poisson(lam0).
    """
    x = _sample_poisson(rng, lam0, n_points)
    mask = rng.random(n_points) < float(pi0)
    x[mask] = 0
    return x


def _sample_poisson_mixture(
    rng: np.random.Generator,
    n: int,
    *,
    pi_hi: float,
    lam_lo: float,
    lam_hi: float,
) -> np.ndarray:
    """Two-component mixture: prob pi_hi draws Poisson(lam_hi), else Poisson(lam_lo)."""
    z = rng.random(int(n)) < float(pi_hi)
    x = np.empty(int(n), dtype=int)
    n_hi = int(z.sum())
    n_lo = int(n) - n_hi
    if n_lo > 0:
        x[~z] = rng.poisson(lam=float(lam_lo), size=n_lo)
    if n_hi > 0:
        x[z] = rng.poisson(lam=float(lam_hi), size=n_hi)
    return x


def generate_phaseI_count(
    rng: np.random.Generator,
    n0: int,
    n_points: int,
    *,
    lam0: float,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase I: IC Poisson(lam0) i.i.d. across time."""
    phaseI, params = [], []
    for t in range(n0):
        x = _sample_poisson(rng, lam0, n_points)
        phaseI.append(_as_col(x))
        params.append({"t": int(t), "dist": "poisson", "lam0": float(lam0)})
    return phaseI, params


def generate_phaseII_count_spike(
    rng: np.random.Generator,
    nT: int,
    n_points: int,
    *,
    lam0: float,
    alpha: float,
    k_star: int,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II: fully OC spike mixture."""
    streams, params = [], []
    for t in range(nT):
        x = _sample_count_spike(rng, lam0, n_points, alpha=alpha, k_star=k_star)
        streams.append(_as_col(x))
        params.append(
            {"t": int(t), "scenario": "count_spike", "lam0": float(lam0),
             "alpha": float(alpha), "k_star": int(k_star)}
        )
    return streams, params


def generate_phaseII_count_zeroinfl(
    rng: np.random.Generator,
    nT: int,
    n_points: int,
    *,
    lam0: float,
    pi0: float,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II: fully OC zero-inflated Poisson."""
    streams, params = [], []
    for t in range(nT):
        x = _sample_count_zeroinfl(rng, lam0, n_points, pi0=pi0)
        streams.append(_as_col(x))
        params.append(
            {"t": int(t), "scenario": "count_zeroinfl", "lam0": float(lam0),
             "pi0": float(pi0)}
        )
    return streams, params


# ---------------------------
# Design 2: Dust-on-screens (Poisson-based, Fig. 2a)
# ---------------------------

def generate_phaseI_dust_poisson(
    rng: np.random.Generator,
    n0: int,
    n_screens: int,
    *,
    lam0: float,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase I (IC): per-screen counts iid Poisson(lam0)."""
    phaseI, params = [], []
    for t in range(int(n0)):
        x = rng.poisson(lam=float(lam0), size=int(n_screens))
        phaseI.append(_as_col(x))
        params.append({"t": int(t), "dist": "poisson", "lam0": float(lam0)})
    return phaseI, params


def generate_phaseII_dust_heavytail_mixture(
    rng: np.random.Generator,
    nT: int,
    n_screens: int,
    *,
    lam0: float,
    pi_hi: float = 0.10,
    lam_lo: Optional[float] = None,
    lam_hi: Optional[float] = None,
    match_mean: bool = True,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II (OC): heavy tail via across-screen Poisson mixture.

    With match_mean=True (default): lam_lo = 0.6*lam0, and lam_hi is solved so
    that the mixture mean equals lam0 (mean-preserving shape change).
    """
    lam0 = float(lam0)
    pi_hi = float(pi_hi)
    if not (0.0 < pi_hi < 1.0):
        raise ValueError("pi_hi must be in (0,1)")
    if lam_lo is None:
        lam_lo = 0.60 * lam0
    lam_lo = float(lam_lo)
    if match_mean:
        lam_hi_val = (lam0 - (1.0 - pi_hi) * lam_lo) / pi_hi
        if lam_hi_val <= 0:
            raise ValueError(
                f"Computed lam_hi={lam_hi_val:.4g} <= 0. "
                "Try smaller lam_lo or pi_hi, or set match_mean=False."
            )
        lam_hi = lam_hi_val
    else:
        lam_hi = float(lam_hi) if lam_hi is not None else 2.50 * lam0

    streams, params = [], []
    for t in range(int(nT)):
        x = _sample_poisson_mixture(rng, int(n_screens), pi_hi=pi_hi,
                                    lam_lo=lam_lo, lam_hi=float(lam_hi))
        streams.append(_as_col(x))
        params.append({
            "t": int(t), "scenario": "dust_heavytail_mixture",
            "lam0": lam0, "pi_hi": pi_hi, "lam_lo": lam_lo,
            "lam_hi": float(lam_hi), "match_mean": bool(match_mean),
        })
    return streams, params


def generate_phaseII_dust_spike(
    rng: np.random.Generator,
    nT: int,
    n_screens: int,
    *,
    lam0: float,
    alpha: float = 0.06,
    k_star: int = 12,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II (OC): spike/atom injection at k_star."""
    streams, params = [], []
    for t in range(int(nT)):
        x = _sample_count_spike(rng, float(lam0), int(n_screens),
                                alpha=float(alpha), k_star=int(k_star))
        streams.append(_as_col(x))
        params.append({
            "t": int(t), "scenario": "dust_spike_kstar",
            "lam0": float(lam0), "alpha": float(alpha), "k_star": int(k_star),
        })
    return streams, params


def run_dust_sims(
    outdir: Path,
    *,
    B: int = 10,
    n0: int = 300,
    nT: int = 300,
    n_screens: int = 300,
    base_seed: int = 42,
    lam0: float = 4.0,
    mix_pi_hi: float = 0.10,
    mix_lam_lo: Optional[float] = None,
    mix_match_mean: bool = True,
    spike_alpha: float = 0.06,
    spike_k_star: int = 12,
) -> None:
    """Generate dust-on-screens replications (Fig. 2a scenarios).

    Outputs (per replication r):
      phaseI_dust_poisson_d1_repXX.npz
      phaseII_dust_mix_d1_repXX.npz
      phaseII_dust_spike_d1_repXX.npz
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d = 1
    for r in range(int(B)):
        rng_I  = np.random.default_rng(int(base_seed) + 10_000 + r)
        rng_II = np.random.default_rng(int(base_seed) + 20_000 + r)

        phaseI, pI = generate_phaseI_dust_poisson(rng_I, int(n0), int(n_screens), lam0=float(lam0))
        _save_phaseI(outdir, f"phaseI_dust_poisson_d1_rep{r:02d}.npz",
                     d=d, n0=int(n0), n_points=int(n_screens),
                     phaseI=phaseI, map_params=pI)

        streams, pII = generate_phaseII_dust_heavytail_mixture(
            rng_II, int(nT), int(n_screens),
            lam0=float(lam0), pi_hi=float(mix_pi_hi),
            lam_lo=mix_lam_lo, match_mean=bool(mix_match_mean),
        )
        _save_phaseII(outdir, f"phaseII_dust_mix_d1_rep{r:02d}.npz",
                      d=d, nT=int(nT), n_points=int(n_screens),
                      streams=streams, map_params=pII)

        streams, pII = generate_phaseII_dust_spike(
            rng_II, int(nT), int(n_screens),
            lam0=float(lam0), alpha=float(spike_alpha), k_star=int(spike_k_star),
        )
        _save_phaseII(outdir, f"phaseII_dust_spike_d1_rep{r:02d}.npz",
                      d=d, nT=int(nT), n_points=int(n_screens),
                      streams=streams, map_params=pII)

        print(f"[dust OK] rep={r:02d} saved to {outdir}")


# ---------------------------
# Design 3: Multiclass categorical
# ---------------------------

def _validate_prob(p: np.ndarray, name: str = "p") -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    if np.any(p < 0):
        raise ValueError(f"{name} has negative entries.")
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"{name} sum is invalid: {s}")
    p = p / s
    return p


def _sample_categorical(rng: np.random.Generator, p: np.ndarray, n_points: int) -> np.ndarray:
    p = _validate_prob(p, "p")
    m = p.size
    return rng.choice(m, size=n_points, replace=True, p=p)


def _p_transfer(p0: np.ndarray, a: int, b: int, delta: float) -> np.ndarray:
    p0 = _validate_prob(p0, "p0")
    m = p0.size
    if not (0 <= a < m and 0 <= b < m and a != b):
        raise ValueError("Invalid (a,b) for transfer.")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0,1).")
    if p0[a] <= delta:
        raise ValueError(f"p0[a]={p0[a]:.4g} <= delta={delta:.4g}; choose smaller delta or different a.")
    p = p0.copy()
    p[a] -= delta
    p[b] += delta
    return _validate_prob(p, "p_oc")


def _p_rare_spike(p0: np.ndarray, rare_idx: int, delta: float) -> np.ndarray:
    p0 = _validate_prob(p0, "p0")
    m = p0.size
    if not (0 <= rare_idx < m):
        raise ValueError("Invalid rare_idx.")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0,1).")
    # (1-delta)p0 + delta e_rare
    p = (1.0 - delta) * p0
    p[rare_idx] += delta
    return _validate_prob(p, "p_oc")

# pmf builders for scenario c and d

def _p_shift_right(p0: np.ndarray, s: int = 1) -> np.ndarray:
    p0 = _validate_prob(p0, "p0")
    m = p0.size
    s = int(s)
    if s <= 0:
        return p0.copy()
    p = np.zeros_like(p0)
    # shift mass right; overflow accumulates at last bin
    if s < m:
        p[s:] = p0[:-s]
        p[-1] += p0[-s:].sum()
    else:
        p[-1] = 1.0
    return _validate_prob(p, "p_shift")

def _p_drift(p0: np.ndarray, t: int, nT: int, *, shift: int = 1, onset: float = 0.0) -> np.ndarray:
    """
    Drift from p0 toward p_end via gamma_t.
    onset in [0,1): fraction of Phase II before drift starts.
    """
    p0 = _validate_prob(p0, "p0")
    p_end = _p_shift_right(p0, s=shift)

    t = int(t)
    nT = int(nT)
    t0 = int(np.floor(onset * nT))
    if t <= t0:
        gamma = 0.0
    else:
        gamma = min(1.0, (t - t0) / max(nT - 1 - t0, 1))
    p = (1.0 - gamma) * p0 + gamma * p_end
    return _validate_prob(p, "p_drift")

def _blur_kernel(m: int, eta: float = 0.20) -> np.ndarray:
    """
    Simple tri-diagonal blur kernel:
      diag = 1-2eta, offdiag = eta (edges renormalized).
    """
    m = int(m)
    eta = float(eta)
    if not (0.0 <= eta < 0.5):
        raise ValueError("eta must be in [0, 0.5).")
    K = np.zeros((m, m), float)
    for i in range(m):
        K[i, i] = 1.0
        if i - 1 >= 0:
            K[i, i - 1] = eta
        if i + 1 < m:
            K[i, i + 1] = eta
    # renormalize rows to sum 1
    K = K / K.sum(axis=1, keepdims=True)
    return K

def _p_blur(p0: np.ndarray, *, eta: float = 0.20) -> np.ndarray:
    p0 = _validate_prob(p0, "p0")
    m = p0.size
    K = _blur_kernel(m, eta=eta)
    p = K @ p0
    return _validate_prob(p, "p_blur")






def generate_phaseI_categorical(
    rng: np.random.Generator,
    n0: int,
    n_points: int,
    *,
    p0: np.ndarray,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase I: IC categorical with p0."""
    p0 = _validate_prob(p0, "p0")
    phaseI, params = [], []
    for t in range(n0):
        y = _sample_categorical(rng, p0, n_points)
        phaseI.append(_as_col(y))  # stored as numeric labels in a column
        params.append({"t": int(t), "dist": "categorical", "p0": p0.copy()})
    return phaseI, params


def generate_phaseII_cat_transfer(
    rng: np.random.Generator,
    nT: int,
    n_points: int,
    *,
    p0: np.ndarray,
    a: int,
    b: int,
    delta: float,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II: fully OC probability transfer a -> b."""
    p0 = _validate_prob(p0, "p0")
    p_oc = _p_transfer(p0, a=a, b=b, delta=delta)
    streams, params = [], []
    for t in range(nT):
        y = _sample_categorical(rng, p_oc, n_points)
        streams.append(_as_col(y))
        params.append(
            {"t": int(t), "scenario": "cat_transfer", "p0": p0.copy(),
             "p_oc": p_oc.copy(), "a": int(a), "b": int(b), "delta": float(delta)}
        )
    return streams, params


def generate_phaseII_cat_rare_spike(
    rng: np.random.Generator,
    nT: int,
    n_points: int,
    *,
    p0: np.ndarray,
    rare_idx: int,
    delta: float,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II: fully OC rare-class spike."""
    p0 = _validate_prob(p0, "p0")
    p_oc = _p_rare_spike(p0, rare_idx=rare_idx, delta=delta)
    streams, params = [], []
    for t in range(nT):
        y = _sample_categorical(rng, p_oc, n_points)
        streams.append(_as_col(y))
        params.append(
            {"t": int(t), "scenario": "cat_rare_spike", "p0": p0.copy(),
             "p_oc": p_oc.copy(), "rare_idx": int(rare_idx), "delta": float(delta)}
        )
    return streams, params


def generate_phaseII_cat_drift(
    rng: np.random.Generator,
    nT: int,
    n_points: int,
    *,
    p0: np.ndarray,
    shift: int = 1,
    onset: float = 0.0,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II: OC gradual drift toward a shifted PMF."""
    p0 = _validate_prob(p0, "p0")
    streams, params = [], []
    for t in range(nT):
        p_t = _p_drift(p0, t=t, nT=nT, shift=shift, onset=onset)
        y = _sample_categorical(rng, p_t, n_points)
        streams.append(_as_col(y))
        params.append({"t": int(t), "scenario": "cat_drift", "p_t": p_t})
    return streams, params

def generate_phaseII_cat_blur(
    rng: np.random.Generator,
    nT: int,
    n_points: int,
    *,
    p0: np.ndarray,
    eta: float = 0.20,
    onset: float = 0.0,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Phase II: OC increasing adjacent-class confusion (blur)."""
    p0 = _validate_prob(p0, "p0")
    p_blur = _p_blur(p0, eta=eta)

    streams, params = [], []
    for t in range(nT):
        # same onset-style ramp
        t0 = int(np.floor(onset * nT))
        gamma = 0.0 if t <= t0 else min(1.0, (t - t0) / max(nT - 1 - t0, 1))
        p_t = _validate_prob((1.0 - gamma) * p0 + gamma * p_blur, "p_t")

        y = _sample_categorical(rng, p_t, n_points)
        streams.append(_as_col(y))
        params.append({"t": int(t), "scenario": "cat_blur", "p_t": p_t, "eta": float(eta)})
    return streams, params


# ---------------------------
# Driver: generate 10 replications per scenario
# ---------------------------

def run_discrete_sims(
    outdir: Path,
    *,
    B: int = 10,
    n0: int = 300,
    nT: int = 300,
    n_points: int = 1024,
    base_seed: int = 42,
    # Counting params
    lam0: float = 6.0,
    spike_alpha: float = 0.08,
    spike_k_star: int = 12,
    zeroinfl_pi0: float = 0.15,
    # Categorical params
    p0: Optional[np.ndarray] = None,
    a: int = 1,
    b: int = 4,
    transfer_delta: float = 0.08,
    rare_idx: Optional[int] = None,
    rare_delta: float = 0.06,
) -> None:
    """
    Generate:
      - Counting IC + (C1) + (C2)
      - Categorical IC + (M1) + (M2)
    Each with B replications.
    """
    d = 1
    if p0 is None:
        p0 = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03], dtype=float)
    p0 = _validate_prob(p0, "p0")
    m = p0.size
    if rare_idx is None:
        rare_idx = m - 1

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for r in range(B):
        # Separate RNGs for Phase I and Phase II for reproducibility
        rng_I  = np.random.default_rng(base_seed + 10_000 + r)
        rng_II = np.random.default_rng(base_seed + 20_000 + r)

        # ----- Counting -----
        phaseI, pI = generate_phaseI_count(rng_I, n0, n_points, lam0=lam0)
        _save_phaseI(outdir, f"phaseI_count_d1_rep{r:02d}.npz",
                     d=d, n0=n0, n_points=n_points, phaseI=phaseI, map_params=pI)

        streams, pII = generate_phaseII_count_spike(
            rng_II, nT, n_points, lam0=lam0, alpha=spike_alpha, k_star=spike_k_star
        )
        _save_phaseII(outdir, f"phaseII_count_spike_d1_rep{r:02d}.npz",
                      d=d, nT=nT, n_points=n_points, streams=streams, map_params=pII)

        streams, pII = generate_phaseII_count_zeroinfl(
            rng_II, nT, n_points, lam0=lam0, pi0=zeroinfl_pi0
        )
        _save_phaseII(outdir, f"phaseII_count_zeroinfl_d1_rep{r:02d}.npz",
                      d=d, nT=nT, n_points=n_points, streams=streams, map_params=pII)

        # ----- Categorical -----
        phaseI, pI = generate_phaseI_categorical(rng_I, n0, n_points, p0=p0)
        _save_phaseI(outdir, f"phaseI_cat_m{m}_rep{r:02d}.npz",
                     d=d, n0=n0, n_points=n_points, phaseI=phaseI, map_params=pI)

        streams, pII = generate_phaseII_cat_transfer(
            rng_II, nT, n_points, p0=p0, a=a, b=b, delta=transfer_delta
        )
        _save_phaseII(outdir, f"phaseII_cat_transfer_m{m}_rep{r:02d}.npz",
                      d=d, nT=nT, n_points=n_points, streams=streams, map_params=pII)

        streams, pII = generate_phaseII_cat_rare_spike(
            rng_II, nT, n_points, p0=p0, rare_idx=rare_idx, delta=rare_delta
        )
        _save_phaseII(outdir, f"phaseII_cat_rare_m{m}_rep{r:02d}.npz",
                      d=d, nT=nT, n_points=n_points, streams=streams, map_params=pII)

        streams, pII = generate_phaseII_cat_drift(
            rng_II, nT, n_points, p0=p0, shift=1, onset=0.2
        )
        _save_phaseII(outdir, f"phaseII_cat_drift_m{m}_rep{r:02d}.npz",
                      d=d, nT=nT, n_points=n_points, streams=streams, map_params=pII)

        streams, pII = generate_phaseII_cat_blur(
            rng_II, nT, n_points, p0=p0, eta=0.20, onset=0.2
        )
        _save_phaseII(outdir, f"phaseII_cat_blur_m{m}_rep{r:02d}.npz",
                      d=d, nT=nT, n_points=n_points, streams=streams, map_params=pII)

        print(f"[OK] rep={r:02d} saved to {outdir}")


if __name__ == "__main__":
    import os, argparse
    parser = argparse.ArgumentParser(description="Generate discrete stream simulation data.")
    parser.add_argument("--n_points", type=int, default=100,
                        help="Points per distribution / n_screens for dust (default: 100)")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--scenario", choices=["all", "count_cat", "dust"], default="all",
                        help="Which scenario group to generate (default: all)")
    args = parser.parse_args()

    n_points = args.n_points
    out_dir = Path(args.out) if args.out else Path(
        os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent.parent / "data")
    ) / "discrete" / f"n{n_points}"

    if args.scenario in ("all", "count_cat"):
        run_discrete_sims(
            out_dir,
            B=10, n0=300, nT=300, n_points=n_points, base_seed=42,
            lam0=4.0,
            spike_alpha=0.08, spike_k_star=10, zeroinfl_pi0=0.15,
            p0=np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03], dtype=float),
            a=1, b=4, transfer_delta=0.08, rare_idx=5, rare_delta=0.06,
        )

    if args.scenario in ("all", "dust"):
        run_dust_sims(
            out_dir,
            B=10, n0=300, nT=300, n_screens=n_points, base_seed=42,
            lam0=4.0,
            mix_pi_hi=0.10, mix_lam_lo=None, mix_match_mean=True,
            spike_alpha=0.04, spike_k_star=10,
        )
