# barycenter estimation
# optimal transport plan estimation
# barycentric projection

import ot
import numpy as np
from pathlib import Path
import torch
# ---------- POT helpers: barycenter & barycentric projection ----------

def _to_2d_clouds(clouds):
    lst = [np.asarray(x, dtype=float) for x in clouds]
    d = 1 if lst[0].ndim == 1 else lst[0].shape[1]
    return [x.reshape(-1, d) for x in lst], d


def _as_float2d(X):
    """Coerce to contiguous float64 array of shape (n, d)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.ascontiguousarray(X, dtype=np.float64)
    return X


def _normalize_weights(w, floor=1e-12):
    w = np.asarray(w, dtype=np.float64).ravel()
    w = np.maximum(w, 0.0)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:  # degenerate
        n = w.size
        return np.full(n, 1.0/n, dtype=np.float64)
    w = w / s
    # avoid exact zeros (can blow up dual updates)
    w[w < floor] = floor
    w /= w.sum()
    return w

def _use_sinkhorn_with_coreset(d: int, n_points: int) -> bool:
    """Policy: only use Sinkhorn + coreset when d=100 and n_points=300."""
    return (d >= 5) # and (n_points == 50)
    # return (d == 100) and (n_points == 300)

def _sinkhorn_stable(a, b, C,
                     reg=None,               # ε on the *scaled* cost
                     numItermax=5000,
                     stopThr=1e-5,
                     tau=1e-3,               # stabilization; smaller => more stable, slower
                     use_eps_scaling=True,
                     verbose=False):
    """
    Stable wrapper for POT's entropic OT solvers.

    a, b : 1D histograms (nonnegative, sum>0)
    C    : cost matrix (n_a x n_b)
    reg  : regularization parameter on the *scaled* cost; if None, defaults to 0.2.
    """
    # 1) rescale the cost to O(1)
    C = np.asarray(C, dtype=np.float64)
    nz = C[C > 0]
    med = float(np.median(nz)) if nz.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = float(np.mean(nz)) if nz.size else 1.0
        if med <= 0:
            med = 1.0
    Cscaled = C / med

    # 2) choose ε on the *scaled* cost
    eps = 0.2 if (reg is None) else float(reg)

    info = {"scaled_cost_median": med,
            "eps_scaled": eps,
            "method": None}

    try:
        if use_eps_scaling:
            # Most robust: log-stabilized + epsilon scaling
            info["method"] = "sinkhorn_epsilon_scaling"
            Pi = ot.bregman.sinkhorn_epsilon_scaling(
                a, b, Cscaled,
                reg=eps,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose
            )
        else:
            # Log-stabilized Sinkhorn
            info["method"] = "sinkhorn_stabilized"
            Pi = ot.bregman.sinkhorn_stabilized(
                a, b, Cscaled,
                reg=eps,
                tau=tau,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose
            )

        if not np.all(np.isfinite(Pi)):
            raise FloatingPointError("non-finite Pi")

        return Pi, info

    except Exception as e:
        # Fallback 1: plain Sinkhorn with slightly larger ε
        if verbose:
            print(f"[sinkhorn] stabilized failed ({e}); "
                  f"retrying with plain sinkhorn and larger eps")
        eps2 = eps * 2.0
        try:
            Pi = ot.sinkhorn(
                a, b, Cscaled,
                reg=eps2,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose
            )
            if not np.all(np.isfinite(Pi)):
                raise FloatingPointError("non-finite Pi (plain)")

            info.update({"method": "sinkhorn_plain",
                         "eps_scaled": eps2})
            return Pi, info

        except Exception as e2:
            # Fallback 2: exact EMD on the *unscaled* cost
            if verbose:
                print(f"[sinkhorn] plain failed ({e2}); falling back to EMD")

            Pi = ot.emd(a, b, C)
            info.update({"method": "emd_fallback",
                         "eps_scaled": None})
            return Pi, info


'''
Pytorch version sinkhorn!
'''

def sinkhorn_stable_torch(
    a, b, C,
    reg=None,               # ε on the *scaled* cost
    numItermax=5000,
    stopThr=1e-5,
    tau=1e-3,
    use_eps_scaling=True,
    verbose=False,
    device="cuda",
    dtype=torch.float64,
    return_torch=False,     # if True, return Pi as torch tensor on device
):
    """
    GPU version of your _sinkhorn_stable:
      - median scales cost
      - runs POT sinkhorn solvers with torch backend
      - fallbacks to plain sinkhorn, then EMD (EMD is CPU only)
    """
    # ---- move inputs to torch on device ----
    if not torch.is_tensor(a):
        a_t = torch.tensor(np.asarray(a, dtype=np.float64), device=device, dtype=dtype)
    else:
        a_t = a.to(device=device, dtype=dtype)

    if not torch.is_tensor(b):
        b_t = torch.tensor(np.asarray(b, dtype=np.float64), device=device, dtype=dtype)
    else:
        b_t = b.to(device=device, dtype=dtype)

    if not torch.is_tensor(C):
        C_t = torch.tensor(np.asarray(C, dtype=np.float64), device=device, dtype=dtype)
    else:
        C_t = C.to(device=device, dtype=dtype)

    assert torch.is_tensor(C_t), f"C_t is not torch, got {type(C_t)}"
    assert torch.is_tensor(a_t), f"a_t is not torch, got {type(a_t)}"
    assert torch.is_tensor(b_t), f"b_t is not torch, got {type(b_t)}"

    # print('Sinkhorn implemented on CUDA!')
    # ---- 1) median cost scaling to O(1) ----
    nz = C_t[C_t > 0]
    if nz.numel() > 0:
        med = torch.median(nz)
        if not torch.isfinite(med) or med <= 0:
            med = torch.mean(nz)
    else:
        med = torch.tensor(1.0, device=device, dtype=dtype)

    if not torch.isfinite(med) or med <= 0:
        med = torch.tensor(1.0, device=device, dtype=dtype)

    Cscaled = C_t / med.clamp(min=1e-12)

    # ---- 2) choose eps on scaled cost ----
    eps = 0.2 if (reg is None) else float(reg)

    info = {
        "scaled_cost_median": float(med.detach().cpu()),
        "eps_scaled": eps,
        "method": None,
        "backend": "torch",
        "device": str(device),
    }

    def _finite(Pi_t: torch.Tensor) -> bool:
        return bool(torch.isfinite(Pi_t).all().item())

    # ---- try stabilized variants on GPU ----
    try:
        if use_eps_scaling:
            info["method"] = "sinkhorn_epsilon_scaling"
            Pi_t = ot.bregman.sinkhorn_epsilon_scaling(
                a_t, b_t, Cscaled,
                reg=eps,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose
            )
        else:
            info["method"] = "sinkhorn_stabilized"
            Pi_t = ot.bregman.sinkhorn_stabilized(
                a_t, b_t, Cscaled,
                reg=eps,
                tau=tau,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose
            )

        if not _finite(Pi_t):
            raise FloatingPointError("non-finite Pi (torch stabilized)")

        if return_torch:
            return Pi_t, info

        return Pi_t.detach().cpu().numpy(), info

    except Exception as e:
        # ---- fallback 1: plain sinkhorn with larger eps ----
        if verbose:
            print(f"[sinkhorn_torch] stabilized failed ({e}); retrying plain sinkhorn with larger eps")
        eps2 = eps * 2.0
        try:
            Pi_t = ot.sinkhorn(
                a_t, b_t, Cscaled,
                reg=eps2,
                numItermax=numItermax,
                stopThr=stopThr,
                verbose=verbose
            )
            if not _finite(Pi_t):
                raise FloatingPointError("non-finite Pi (torch plain)")

            info.update({"method": "sinkhorn_plain", "eps_scaled": eps2})
            if return_torch:
                return Pi_t, info
            return Pi_t.detach().cpu().numpy(), info

        except Exception as e2:
            # ---- fallback 2: EMD (CPU only) on unscaled cost ----
            if verbose:
                print(f"[sinkhorn_torch] plain failed ({e2}); falling back to EMD (CPU)")
            a_np = a_t.detach().cpu().numpy()
            b_np = b_t.detach().cpu().numpy()
            C_np = C_t.detach().cpu().numpy()
            Pi = ot.emd(a_np, b_np, C_np)
            info.update({"method": "emd_fallback", "eps_scaled": None})
            return Pi, info



def barycenter_estimation(particle_sets,
                          n_bary: int = 500,
                          seed: int = 0,
                          ):
    """
    Free-support Wasserstein barycenter via POT's LP solver.

    Parameters
    ----------
    particle_sets : sequence of arrays
        Each element Xi has shape (n_i, d) = point cloud for one distribution.
    n_bary : int
        Desired number of support points for the barycenter.
    seed : int
        RNG seed for selecting initial barycenter support.

    Returns
    -------
    Xb : (n_bary_eff, d)
        Barycenter support locations.
    wb : (n_bary_eff,)
        Uniform barycenter weights.
    """
    rng = np.random.default_rng(seed)

    # 1) numeric clouds
    Xs, d = _to_2d_clouds(particle_sets)

    # 2) per-measure weights (uniform within each cloud)
    as_ = [np.ones(len(Xi), dtype=float) / float(len(Xi)) for Xi in Xs]

    # 3) pooled Phase-I particles
    pooled = np.vstack(Xs)
    N = pooled.shape[0]

    # 4) choose barycenter support size and initialization
    if n_bary >= N:
        # use all pooled points (your previous behaviour)
        X_init = pooled
        n_bary_eff = N
    else:
        # subsample pooled points as initial support
        idx = rng.choice(N, size=n_bary, replace=False)
        X_init = pooled[idx]
        n_bary_eff = n_bary

    # 5) uniform barycenter weights
    wb = np.ones(n_bary_eff, dtype=float) / float(n_bary_eff)

    # 6) free-support barycenter
    Xb = ot.lp.free_support_barycenter(
        measures_locations=Xs,
        measures_weights=as_,
        X_init=X_init,
        b=wb,
        numItermax=1000,
        stopThr=1e-5,
        verbose=True,
    )

    return Xb, wb


# TODO: should we use sinkhorn
def barycentric_projection_map(X_src: np.ndarray, a_src: np.ndarray,
                               X_tgt: np.ndarray, a_tgt: np.ndarray,
                               method: str = "emd",      # "sinkhorn" or "emd"
                               reg: float = 0.05, numItermax=2000, stopThr=1e-5) -> np.ndarray:
    """
    Compute entropic OT coupling pi between (X_src, a_src) and (X_tgt, a_tgt),
    then compute barycentric projection of X_src onto X_tgt: T(x_i) = sum_j pi_ij * y_j / a_src_i.
    Returns array T(X_src) with shape (n_src, d).
    """
    X_src = _as_float2d(X_src)     # (n_s, d), float64
    X_tgt = _as_float2d(X_tgt)     # (n_t, d), float64

    a = _normalize_weights(a_src)
    b = _normalize_weights(a_tgt)

    C = ot.dist(X_src, X_tgt, metric='euclidean')**2 # X_src, X_tgt should be the locations?

    # Plan solver
    if method.lower() == "emd":
        # Exact unregularized OT: network simplex
        # (fast for small/medium problems; can be heavy for 1000x2000)
        # Pi = ot.emd(a_src, a_tgt, C)
        Pi = ot.emd(a, b, C)
        # TODO: check the error estimation - may use sinkhorn in high-dimension

    elif method.lower() == "sinkhorn":
        # stabilized Sinkhorn on scaled cost
        # Pi, med = _sinkhorn_fast(a_src, a_tgt, C, reg, numItermax=numItermax, stopThr=stopThr)
        Pi, med = _sinkhorn_stable(a, b, C, reg, numItermax=numItermax, stopThr=stopThr,use_eps_scaling=True, verbose=False)

        # Torch version is not faster
        # Pi_t, med = sinkhorn_stable_torch(a, b, C, reg=reg, numItermax=numItermax, stopThr=stopThr,
        #                                   use_eps_scaling=True,
        #                                   device="cuda",
        #                                   verbose=False)
        # Pi = Pi_t

    else:
        raise ValueError("method must be 'sinkhorn' or 'emd'")
    # compute barycentric map; scaling of C does not affect Pi @ X_tgt
    denom = Pi.sum(axis=1, keepdims=True) + 1e-16 #a[:, None] + 1e-16

    Txs = (Pi @ X_tgt) / denom  # (n_s, d)

    return Txs


#%% The functions for Phase I and Phase II optimal transport map estimations

def process_phaseI(file_phaseI: Path, reg=0.05, method='emd', n_bary=512):
    """
    NOTE: 'method' arg is ignored. We enforce:
      - Sinkhorn + coreset ONLY if (d==100 and n_points==300)
      - Otherwise EMD with raw points (no coreset).
    """
    z = np.load(file_phaseI, allow_pickle=True)
    phaseI_raw = list(z["phaseI"])
    phaseI, d  = _to_2d_clouds(phaseI_raw)

    n_points = int(z["n_points"]) if "n_points" in z.files else phaseI[0].shape[0]

    # keep your current choice (free-support LP barycenter on pooled particles)
    Xb, wb = barycenter_estimation(phaseI)

    # decide OT path once (applies to every Phase-I cloud)
    use_sinkhorn = _use_sinkhorn_with_coreset(d, n_points)
    print('use sinkhorn? ', use_sinkhorn)
    if not use_sinkhorn:
        # EMD path: raw targets, uniform weights; reg not used
        n_co_default = None   # not used
        reg_eff = reg         # irrelevant for EMD

    tangents, T_hat_list, tan_norms = [], [], []
    for Y in phaseI:
        if use_sinkhorn:
            Yc, wc = Y, np.ones(len(Y)) / len(Y)
            Txs = barycentric_projection_map(
                Xb, wb, Yc, wc, method="sinkhorn",
                reg=reg, numItermax=5000, stopThr=1e-5
            )
        else:
            # EMD path on raw data
            Yc, wc = Y, np.ones(len(Y)) / len(Y)
            Txs = barycentric_projection_map(
                Xb, wb, Yc, wc, method="emd",
                reg=reg, numItermax=5000, stopThr=1e-5
            )

        t = Txs - Xb
        tangents.append(t)
        T_hat_list.append(Txs)
        tan_norms.append(float(np.mean(np.sum(t * t, axis=1))))

    paramsI = list(z["map_params"]) if "map_params" in z.files else [None] * len(phaseI)
    return Xb, wb, tangents, T_hat_list, paramsI, phaseI, tan_norms



def process_phaseII(file_phaseII: Path,
                    Xb: np.ndarray,
                    wb: np.ndarray,
                    reg=0.05,
                    numItermax=5000,
                    stopThr=1e-5,
                    verbose=False,
                    method='sinkhorn',       # ignored; path decided internally
                    ):
    """
    NOTE: 'method' and 'n_co' args are ignored for policy. We enforce:
      - Sinkhorn + coreset ONLY if (d==100 and n_points==300)
      - Otherwise EMD with raw points (no coreset).
    """
    z = np.load(file_phaseII, allow_pickle=True)
    streams_raw = list(z["streams"])
    streams, d  = _to_2d_clouds(streams_raw)

    tangents, T_hat_list, tan_norms = [], [], []
    for Y in streams:
        n_points = Y.shape[0]
        use_sinkhorn = _use_sinkhorn_with_coreset(d, n_points)

        if use_sinkhorn:
            Yc, wc = Y, np.ones(len(Y)) / len(Y)
            Txs = barycentric_projection_map(
                Xb, wb, Yc, wc, method="sinkhorn",
                reg=reg, numItermax=numItermax, stopThr=stopThr
            )
        else:
            # EMD on raw points
            Yc, wc = Y, np.ones(len(Y)) / len(Y)
            Txs = barycentric_projection_map(
                Xb, wb, Yc, wc, method="emd",
                reg=reg, numItermax=numItermax, stopThr=stopThr
            )

        t = Txs - Xb
        tangents.append(t)
        T_hat_list.append(Txs)
        tan_norms.append(float(np.mean(np.sum(t * t, axis=1))))

    paramsII = list(z["map_params"]) if "map_params" in z.files else [None] * len(streams)
    Txs_true = streams_raw
    return tangents, T_hat_list, paramsII, streams, tan_norms, Txs_true







