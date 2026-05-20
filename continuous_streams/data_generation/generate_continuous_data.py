"""
Continuous stream simulation data generator.

Generates Phase-I (IC) and Phase-II (shift) clouds for all paper scenarios:
  - IC            : in-control (baseline shape variability)
  - mm_reweight   : multimodal reweight (perturbed mixture weights, marginals preserved)
  - copula_shift  : copula shift (changed dependence structure, marginals preserved)
  - barycenter    : barycenter change (new mixture-of-Betas base distribution)

Usage:
  python generate_continuous_data.py --out ../../data/continuous --n_points 1024

Configure via environment variable:
  DIDO_DATA_ROOT: root data directory (default: ../../../data/)
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

sys.path.insert(0, str(Path(__file__).parent))

from common_ot import (
    make_barycenter_mob,
    sample_from_mob,
    softplus_beta,
    sigmoid_beta,
    make_mean_preserving,
)


# ============================================================
# Utilities
# ============================================================

def _unit_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, float)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def renorm(w: np.ndarray, mode: str = "l1") -> np.ndarray:
    w = np.abs(np.asarray(w, float))
    if mode == "l1":
        s = w.sum()
        return w / (s if s > 0 else 1.0)
    if mode == "l2":
        s = np.linalg.norm(w)
        return w / (s if s > 0 else 1.0)
    return w


def _safe_cov(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    n, d = X.shape
    if d == 1:
        v = float(np.var(X[:, 0], ddof=1)) if n > 1 else 0.0
        return np.array([[v + eps]], dtype=float)
    return np.asarray(np.cov(X.T, ddof=1), float) + eps * np.eye(d)


def _moment_match_affine(
    X: np.ndarray,
    X_target: np.ndarray,
    match: str = "mean_cov",
    eps: float = 1e-8,
) -> np.ndarray:
    """Affine-transform X to match mean (and optionally covariance) of X_target."""
    X = np.asarray(X, float)
    X_target = np.asarray(X_target, float)
    if match == "none":
        return X
    mu_t = X_target.mean(axis=0)
    mu_x = X.mean(axis=0)
    if match == "mean":
        return X + (mu_t - mu_x)
    C_t = _safe_cov(X_target, eps=eps)
    C_x = _safe_cov(X, eps=eps)
    w_x, V_x = np.linalg.eigh(C_x)
    w_t, V_t = np.linalg.eigh(C_t)
    w_x = np.maximum(w_x, eps)
    w_t = np.maximum(w_t, eps)
    Cx_invsqrt = V_x @ np.diag(1.0 / np.sqrt(w_x)) @ V_x.T
    Ct_sqrt    = V_t @ np.diag(np.sqrt(w_t))        @ V_t.T
    return (X - mu_x) @ Cx_invsqrt.T @ Ct_sqrt.T + mu_t


def _perturb_simplex_weights(
    w: np.ndarray,
    rng: np.random.Generator,
    delta: float,
) -> np.ndarray:
    """Log-perturb and renormalize mixture weights."""
    w = np.maximum(np.asarray(w, float), 1e-12)
    w_new = np.exp(np.log(w) + delta * rng.normal(size=w.shape))
    return w_new / np.sum(w_new)


def _iman_conover_reorder(
    X: np.ndarray,
    rng: np.random.Generator,
    rho: float = 0.6,
) -> np.ndarray:
    """Rank-reorder X to impose approximate equicorrelation rho (preserves marginals)."""
    X = np.asarray(X, float)
    n, d = X.shape
    if d <= 1:
        return X.copy()
    rho = float(np.clip(rho, -0.99 / (d - 1) + 1e-6, 0.99))
    R = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))
    Z = rng.multivariate_normal(mean=np.zeros(d), cov=R, size=n)
    X_new = np.empty_like(X)
    for j in range(d):
        x_sorted = np.sort(X[:, j])
        ranks = np.empty(n, dtype=int)
        ranks[np.argsort(Z[:, j])] = np.arange(n)
        X_new[:, j] = x_sorted[ranks]
    return X_new


def _perturb_bar_beta(
    bar: Dict,
    delta: float,
    effect: str,
    rng: np.random.Generator,
    d: int,
) -> None:
    """In-place perturbation of a mixture-of-Betas barycenter's concentration parameters."""
    eps = 1e-6
    for k in range(len(bar["weights"])):
        ab      = bar["betas"][k]       # (d, 2)
        alpha_k = ab[:, 0]
        beta_k  = ab[:, 1]
        kappa   = alpha_k + beta_k
        m       = alpha_k / np.maximum(kappa, eps)
        m_new   = np.clip(m + delta * rng.normal(size=d), eps, 1.0 - eps)
        if effect == "mean_only":
            kappa_new = kappa
        elif effect == "mean_var":
            kappa_new = np.maximum(kappa * np.exp(delta * rng.normal(size=d)), 2 * eps)
        else:
            raise ValueError(f"bary_effect must be 'mean_only' or 'mean_var', got {effect!r}")
        bar["betas"][k] = np.stack([
            np.maximum(m_new * kappa_new, eps),
            np.maximum((1.0 - m_new) * kappa_new, eps),
        ], axis=-1)


# ============================================================
# Phase I generator
# ============================================================

def generate_phaseI(
    d: int,
    K_bary: int,
    n0: int,
    n_points: int,
    eps_ic: Tuple[float, float] = (0.10, 0.15),
    beta: float = 2.0,
    J: int = 6,
    seed: int = 43,
) -> Tuple[Dict, np.ndarray, List[np.ndarray], List, List[Dict]]:
    """
    Phase-I (IC) generator: shape-only, mean-preserving smooth OT maps.

    Returns
    -------
    bar      : MoG barycenter parameter dict
    X_base   : (n_points, d) reference particles
    phaseI   : list[n0] of (n_points, d) IC clouds
    mapsI    : list of callables T_t
    parasI   : list of per-observation parameter dicts
    """
    r      = default_rng(seed)
    bar    = make_barycenter_mob(d, K_bary, seed=43)
    X_base = sample_from_mob(n_points, bar, seed=43)
    A_ref  = _unit_rows(r.normal(size=(J, d)))

    phaseI: List[np.ndarray] = []
    mapsI:  List             = []
    parasI: List[Dict]       = []

    for _ in range(n0):
        eps_t = float(r.uniform(*eps_ic))
        w_t   = renorm(np.abs(r.normal(loc=1.0, scale=0.30, size=J)), mode="l1")
        c_t   = r.normal(loc=0.0, scale=0.6, size=J)

        def T_raw(x, A=A_ref, w=w_t, c=c_t, eps=eps_t, beta_=beta):
            X = np.atleast_2d(x)
            z = X @ A.T - c[None, :]
            g = softplus_beta(z, beta_) * sigmoid_beta(z, beta_)
            return X + (2.0 * eps) * (g @ (w[:, None] * A))

        T_t, drift = make_mean_preserving(T_raw, X_base)
        phaseI.append(np.asarray(T_t(X_base), float))
        mapsI.append(T_t)
        parasI.append(dict(
            A=A_ref, w=w_t, c=c_t, eps=eps_t, beta=float(beta),
            mean_drift=drift, kind="IC_shape_only",
        ))

    return bar, X_base, phaseI, mapsI, parasI


# ============================================================
# Phase II generator
# ============================================================

def generate_phaseII_stream_unified(
    d: int,
    nT: int,
    n_points: int,
    X_base_phaseI: np.ndarray,
    scenario: str,
    *,
    map_strength: float = 1.0,
    use_A_ref: Optional[np.ndarray] = None,
    keep_mean: bool = True,
    beta: float = 2.0,
    J: int = 6,
    rng: Optional[np.random.Generator] = None,
    # mm_reweight
    bar_phaseI: Optional[Dict] = None,
    mm_reweight_delta: float = 0.6,
    mm_match: str = "mean_cov",
    # copula_shift
    copula_rho: float = 0.6,
    # barycenter
    K_bary: int = 4,
    barycenter_delta: float = 0.0,
    bary_effect: str = "mean_var",
) -> Tuple[np.ndarray, List[np.ndarray], List[Dict], Optional[Dict]]:
    """
    Phase-II generator for all paper scenarios.

    Scenarios (case-insensitive):
      - "IC"           : same distribution as Phase I
      - "mm_reweight"  : perturb mixture weights (requires bar_phaseI)
      - "copula_shift" : change dependence structure via Iman-Conover reordering
      - "barycenter"   : sample from a new mixture-of-Betas barycenter

    Returns
    -------
    X_base_use : (n_points, d) Phase-II support
    streams    : list[nT] of (n_points, d) clouds
    params     : list of per-t parameter dicts
    bar2       : new/perturbed barycenter dict (mm_reweight / barycenter), else None
    """
    r  = np.random.default_rng() if rng is None else rng
    sc = scenario.lower()
    X_base_phaseI = np.asarray(X_base_phaseI, float)
    bar2 = None

    # ---- choose Phase-II base distribution ----
    if sc == "ic":
        X_base_use = X_base_phaseI.copy()

    elif sc == "mm_reweight":
        if bar_phaseI is None:
            raise ValueError("mm_reweight requires bar_phaseI (returned by generate_phaseI).")
        bar2 = copy.deepcopy(bar_phaseI)
        bar2["weights"] = _perturb_simplex_weights(
            np.asarray(bar2["weights"], float), r, mm_reweight_delta)
        X_base_use = sample_from_mob(n_points, bar2, seed=int(r.integers(1, 1_000_000)))
        X_base_use = _moment_match_affine(X_base_use, X_base_phaseI, match=mm_match)

    elif sc == "copula_shift":
        X_base_use = _iman_conover_reorder(X_base_phaseI, r, rho=copula_rho)

    elif sc == "barycenter":
        bar2 = make_barycenter_mob(d, K_bary, seed=int(r.integers(1, 1_000_000)))
        if barycenter_delta > 0:
            _perturb_bar_beta(bar2, barycenter_delta, bary_effect, r, d)
        X_base_use = sample_from_mob(n_points, bar2, seed=int(r.integers(1, 1_000_000)))

    else:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            "Choose from: 'IC', 'mm_reweight', 'copula_shift', 'barycenter'."
        )

    # ---- generate per-time streams with near-IC maps ----
    streams: List[np.ndarray] = []
    params:  List[Dict]       = []

    for _ in range(nT):
        eps   = float(r.uniform(0.10, 0.15)) * float(map_strength)
        A_use = use_A_ref if use_A_ref is not None else _unit_rows(r.normal(size=(J, d)))
        w_use = renorm(np.abs(r.normal(loc=1.0, scale=0.30, size=J)), mode="l1")
        c_use = r.normal(loc=0.0, scale=0.6, size=J)

        def T_raw(x, A=A_use, w=w_use, c=c_use, eps_=eps, beta_=beta):
            X = np.atleast_2d(x)
            z = X @ A.T - c[None, :]
            g = softplus_beta(z, beta_) * sigmoid_beta(z, beta_)
            return X + (2.0 * eps_) * (g @ (w[:, None] * A))

        if keep_mean:
            T_use, drift = make_mean_preserving(T_raw, X_base_use)
        else:
            T_use, drift = T_raw, None

        streams.append(np.asarray(T_use(X_base_use), float))
        params.append(dict(
            kind=sc,
            eps=float(eps), beta=float(beta),
            A=A_use, w=w_use, c=c_use,
            mean_preserved=bool(keep_mean),
            drift_centered=drift,
            mm_reweight_delta=(float(mm_reweight_delta) if sc == "mm_reweight" else None),
            mm_match=(mm_match if sc == "mm_reweight" else None),
            copula_rho=(float(copula_rho) if sc == "copula_shift" else None),
            barycenter_delta=(float(barycenter_delta) if sc == "barycenter" else None),
            note=("copula_shift has no effect in 1D" if sc == "copula_shift" and d <= 1 else None),
        ))

    return X_base_use, streams, params, bar2


# ============================================================
# Batch simulator
# ============================================================

def simulate_replications(
    outdir: Path,
    d_list: Tuple = (1, 5, 10, 50),
    K_list: Tuple = (4,),
    scenarios: Tuple = ("IC", "mm_reweight", "copula_shift", "barycenter"),
    B: int = 10,
    n0: int = 300,
    nT: int = 300,
    n_points: int = 1024,
    eps_ic: Tuple[float, float] = (0.10, 0.15),
    beta: float = 2.0,
    J: int = 6,
    seed0: int = 423,
    *,
    map_strength: float = 1.0,
    mm_reweight_delta: float = 0.6,
    mm_match: str = "mean_cov",
    copula_rho: float = 0.6,
    barycenter_delta: float = 0.0,
    bary_effect: str = "mean_var",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[simulate_replications] writing to: {outdir}")

    for d in d_list:
        for K in (K_list if isinstance(K_list, (list, tuple)) else (K_list,)):
            for rep in range(B):
                rI  = default_rng(seed0 + 10_000 * d + 1_000 * K + 10 * rep)
                rII = default_rng(seed0 + 20_000 * d + 2_000 * K + 20 * rep)

                bar, X_base, phaseI, _, parasI = generate_phaseI(
                    d=d, K_bary=K, n0=n0, n_points=n_points,
                    eps_ic=eps_ic, beta=beta, J=J,
                    seed=int(rI.integers(1, 1_000_000)),
                )
                A_ref = np.asarray(parasI[0]["A"], float)

                np.savez(
                    outdir / f"phaseI_d{d}_K{K}_rep{rep:02d}.npz",
                    d=d, K=K, n0=n0, n_points=n_points,
                    X_base=X_base,
                    phaseI=np.array(phaseI, dtype=object),
                    map_params=np.array(parasI, dtype=object),
                    barycenter_phaseI=bar,
                )

                for scen in scenarios:
                    Xb2, streams, parasII, bar2 = generate_phaseII_stream_unified(
                        d=d, nT=nT, n_points=n_points,
                        X_base_phaseI=X_base,
                        scenario=scen,
                        map_strength=map_strength,
                        use_A_ref=A_ref,
                        keep_mean=True,
                        beta=beta, J=J, rng=rII,
                        bar_phaseI=bar,
                        mm_reweight_delta=mm_reweight_delta,
                        mm_match=mm_match,
                        copula_rho=copula_rho,
                        K_bary=K,
                        barycenter_delta=barycenter_delta,
                        bary_effect=bary_effect,
                    )

                    np.savez(
                        outdir / f"phaseII_{scen}_d{d}_K{K}_rep{rep:02d}.npz",
                        d=d, K=K, nT=nT, n_points=n_points,
                        X_base=Xb2,
                        streams=np.array(streams, dtype=object),
                        map_params=np.array(parasII, dtype=object),
                        barycenter_phaseII=(bar2 if bar2 is not None else None),
                    )

                print(f"[OK] d={d}, K={K}, rep={rep:02d}")

    print("[simulate_replications] done.")


# ============================================================
# Debug visualization (1D only)
# ============================================================

def viz_1d_phaseI_phaseII(
    phaseI_file: Path,
    phaseII_file: Path,
    t_show: int = 0,
    bins: int = 40,
) -> None:
    zI = np.load(phaseI_file, allow_pickle=True)
    z2 = np.load(phaseII_file, allow_pickle=True)
    xI  = np.asarray(list(zI["phaseI"])[t_show]).reshape(-1)
    xII = np.asarray(list(z2["streams"])[t_show]).reshape(-1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), dpi=160)
    ax[0].hist(xI,  bins=bins, density=True, alpha=0.5, label="Phase I (IC)")
    ax[0].hist(xII, bins=bins, density=True, alpha=0.5, label="Phase II (OC)")
    ax[0].set_title("Histogram (density)")
    ax[0].legend()
    ax[1].plot(np.sort(xI),  np.linspace(0, 1, len(xI)),  label="Phase I (IC)")
    ax[1].plot(np.sort(xII), np.linspace(0, 1, len(xII)), label="Phase II (OC)")
    ax[1].set_title("ECDF")
    ax[1].legend()
    fig.suptitle(
        f"t={t_show}: mean(I)={xI.mean():.3f}, mean(II)={xII.mean():.3f} | "
        f"std(I)={xI.std(ddof=1):.3f}, std(II)={xII.std(ddof=1):.3f}"
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help=(
                            "Output directory. If the path does not end with n<N>, "
                            "data are written under <out>/n<N> so main_run_flow.py can find them."
                        ))
    parser.add_argument("--n_points", type=int, default=1024,
                        help="Points per distribution (default: 1024)")
    args = parser.parse_args()

    outdir = (
        Path(args.out)
        if args.out
        else Path(os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent.parent / "data"))
        / "continuous"
    )
    n_tag = f"n{args.n_points}"
    if outdir.name != n_tag:
        outdir = outdir / n_tag
    outdir.mkdir(parents=True, exist_ok=True)

    simulate_replications(
        outdir=outdir,
        d_list=(1, 5, 10, 50),
        K_list=(4,),
        scenarios=("IC", "mm_reweight", "copula_shift", "barycenter"),
        B=10,
        n0=300, nT=300, n_points=args.n_points,
        seed0=423,
        map_strength=1.0,
    )

    phaseI_file  = outdir / "phaseI_d1_K4_rep00.npz"
    phaseII_file = outdir / "phaseII_mm_reweight_d1_K4_rep00.npz"
    if phaseI_file.exists() and phaseII_file.exists():
        viz_1d_phaseI_phaseII(phaseI_file, phaseII_file, t_show=0,  bins=40)
        viz_1d_phaseI_phaseII(phaseI_file, phaseII_file, t_show=50, bins=40)
