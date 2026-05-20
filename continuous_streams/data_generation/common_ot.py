# common_ot.py
import numpy as np
from numpy.random import default_rng
from typing import List, Tuple, Dict

rng = default_rng

# TODO: check the hyperparatemers setup and meaning

# ---------- Random reference (barycenter) MoG and sampling ----------
def rand_spd(d, r):
    U, _ = np.linalg.qr(r.normal(size=(d, d)))
    vals = np.exp(r.normal(0.0, 0.4, size=d))
    return U @ np.diag(vals) @ U.T

def sample_dirichlet(K, alpha=3.0, r=None):
    r = rng() if r is None else r
    a = r.gamma(shape=alpha, scale=1.0, size=K)
    return a / a.sum()

# barycenter constrained to (0,1) --> mixture of beta
# def make_barycenter_mog(d: int, K: int, seed: int = 0) -> Dict:
#     r = rng(seed)
#     means = [r.normal(0.0, 1.0, size=d) for _ in range(K)] # change to beta
#     covs  = [rand_spd(d, r) for _ in range(K)]
#     w     = sample_dirichlet(K, alpha=3.0, r=r)
#     return dict(weights=w, means=means, covs=covs)
#
# # sample from mixture of Beta
# def sample_from_mog(n: int, mog: Dict, seed: int = None) -> np.ndarray:
#     r = rng(seed)
#     w, means, covs = mog["weights"], mog["means"], mog["covs"]
#     K, d = len(w), means[0].shape[0]
#     comps = r.choice(K, size=n, p=w)
#     X = np.zeros((n, d))
#     for k in range(K):
#         idx = (comps == k)
#         nk = idx.sum()
#         if nk:
#             X[idx] = r.multivariate_normal(means[k], covs[k], size=nk)
#     return X

def _rand_beta_params(
    d: int,
    r: np.random.Generator,
    mean_range: tuple[float, float] = (0.1, 0.9),   # where the Beta means live
    kappa_range: tuple[float, float] = (3.0, 30.0)  # concentration range (larger => tighter)
) -> np.ndarray:
    """
    Returns shape params with shape (d, 2), columns are (alpha, beta) per coordinate.
    """
    m = r.uniform(*mean_range, size=d)             # target mean in (0,1)
    kappa = r.uniform(*kappa_range, size=d)        # total concentration
    alpha = m * kappa
    beta  = (1.0 - m) * kappa
    return np.stack([alpha, beta], axis=-1)        # (d, 2)

# ===== Mixture of Betas (independent across dimensions) =====
# barycenter constrained to (0,1)^d --> mixture of Beta
def make_barycenter_mob(
    d: int,
    K: int,
    seed: int = 0,
    mean_range: tuple[float,float] = (0.1, 0.9),
    kappa_range: tuple[float,float] = (3.0, 30.0),
    alpha_mix: float = 3.0,
) -> Dict:
    """
    Build a mixture-of-Betas (product across dims) with K components in d dimensions.
    Returns:
      dict(weights=w, betas=betas)
      - weights: (K,) mixture weights
      - betas: list of length K; each item is an array (d,2) with (alpha_j, beta_j)
    """
    r = rng(seed)
    w = sample_dirichlet(K, alpha=alpha_mix, r=r)
    betas = [_rand_beta_params(d, r, mean_range, kappa_range) for _ in range(K)]
    return dict(weights=w, betas=betas)

def sample_from_mob(n: int, mob: Dict, seed: int | None = None) -> np.ndarray:
    """
    Sample n points from a mixture-of-Betas on (0,1)^d.
    mob = dict(weights=(K,), betas=[(d,2)]*K)
    Returns X with shape (n, d).
    """
    r = rng(seed)
    w, betas = mob["weights"], mob["betas"]
    K = len(w)
    d = betas[0].shape[0]
    comps = r.choice(K, size=n, p=w)
    X = np.empty((n, d))
    for k in range(K):
        idx = (comps == k)
        nk = int(idx.sum())
        if nk:
            # Vectorized: r.beta supports broadcasting over (d,) alpha/beta
            alpha_k = betas[k][:, 0]
            beta_k  = betas[k][:, 1]
            X[idx, :] = r.beta(alpha_k, beta_k, size=(nk, d))
    return X


def as_2d(x, d):
    """Ensure x is (n,d) float64."""
    X = np.asarray(x, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, d)
    return np.ascontiguousarray(X, dtype=np.float64)

# ---------- OT-valid random convex map: T = grad psi ----------
# def sample_convex_ot_map(d: int, J: int = 6, eps: float = 0.25, seed: int = None):
#     """
#     # TODO: Change to sigmoid
#     E [] = 0 + noise
#
#     T(x)=x + 2 eps sum w_j (a_j^T x - c_j)_+ a_j
#     returned as a function that maps (n,d)->(n,d) with NO squeeze.
#     Also returns the parameter dict for validation.
#     """
#     r = default_rng(seed)
#     A = []
#     for _ in range(J):
#         a = r.normal(size=d)
#         a /= np.linalg.norm(a)
#         A.append(a)
#     A = np.stack(A, axis=0)  # (J, d)
#     # TODO: check W is abs?
#     w = np.abs(r.normal(loc=1.0, scale=0.3, size=J))
#     c = r.normal(loc=0.0, scale=0.6, size=J)
#
#     def T(x):
#         X = as_2d(x, d)  # (n, d)
#         s = X @ A.T - c[None, :]  # (n, J)
#         s_pos = np.maximum(s, 0.0)  # (n, J)
#         # weights 'w' scale each bump direction
#
#         # TODO: make it Gaussian
#         Y = X + (2.0 * eps) * (s_pos @ (w[:, None] * A))  # (n, d)
#         return Y  # (n, d), no squeeze
#
#     params = dict(A=A, w=w, c=c, eps=float(eps))
#     return T, params

'''
improve to use sigmoid to generate the optimal transport map
'''
def softplus_beta(z, beta=4.0):
    # stable softplus
    z = np.asarray(z, dtype=float)
    return np.where(z > 0, z + np.log1p(np.exp(-z*beta))/beta, np.log1p(np.exp(z*beta))/beta)

def sigmoid_beta(z, beta=4.0):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-beta * z))


def make_mean_preserving(T_raw, X_base):
    U = T_raw(X_base) - X_base  # (n,d)
    b_hat = U.mean(axis=0)      # (d,)
    def T_centered(x):
        x = np.atleast_2d(x)
        return (T_raw(x) - b_hat).squeeze()
    return T_centered, b_hat



def sample_convex_ot_map_sigmoid(
    d, J=6, eps=0.25, beta=4.0, mean_preserve_with=None, seed=None,
    # new shape-variation knobs:
    A_base=None,               # reuse directions (aligned) or None to resample
    w_mode="normalize_l1",     # "none" | "normalize_l1" | "normalize_l2"
    c_jitter_std=0.0,          # small std to move bump locations (shape)
):
    r = default_rng(seed)

    # directions
    if A_base is not None:
        A = np.array(A_base, float)
    else:
        A = r.normal(size=(J, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)

    # weights (positive) then OPTIONAL renormalize to keep amplitu de comparable
    w = np.abs(r.normal(loc=1.0, scale=0.3, size=J))
    if w_mode == "normalize_l1":
        s = w.sum();  w = w / (s if s > 0 else 1.0)
    elif w_mode == "normalize_l2":
        s = np.linalg.norm(w); w = w / (s if s > 0 else 1.0)
    # else: leave as-is

    # thresholds with optional jitter (shape change by moving bump location)
    c = r.normal(loc=0.0, scale=0.6, size=J)
    if c_jitter_std > 0:
        c = c + r.normal(0.0, c_jitter_std, size=J)

    def T_raw(x):
        X = as_2d(x, d)
        z = X @ A.T - c[None, :]
        g = softplus_beta(z, beta) * sigmoid_beta(z, beta)
        Y = X + (2.0 * eps) * (g @ (w[:, None] * A))
        return Y

    T = T_raw
    drift = None
    if mean_preserve_with is not None:
        T, drift = make_mean_preserving(T_raw, np.asarray(mean_preserve_with, float))

    return T, dict(A=A, w=w, c=c, eps=float(eps), beta=float(beta), mean_drift=drift)




# ---------- POT helpers: barycenter & barycentric projection ----------
# Requires: pip install POT
import ot
import ot.bregman

# def _uniform_coreset(X, n_co=None, seed=0):
#     """
#     Return (X_co, w_co). If n_co is None or X is already small, return X unchanged.
#     """
#     X = _as_float2d(X)
#     n = X.shape[0]
#     if (n_co is None) or (n <= int(n_co)):
#         w = np.ones(n, dtype=np.float64) / n
#         return X, w
#     rng = np.random.default_rng(seed)
#     idx = rng.choice(n, size=int(n_co), replace=False)
#     Xc = X[idx]
#     w  = np.ones(Xc.shape[0], dtype=np.float64) / Xc.shape[0]
#     return Xc, w

def _as_2d_clouds(particle_sets):
    """
    Coerce every cloud to a numeric float64 array of shape (n_i, d).
    Works even if inputs came from an npz object array.
    """
    # infer d from the first item
    X0 = np.asarray(particle_sets[0], dtype=float)
    d = 1 if X0.ndim == 1 else X0.shape[1]
    Xs2d = [np.asarray(Xi, dtype=float).reshape(-1, d) for Xi in particle_sets]
    return Xs2d, d

def _as_float2d(X):
    """Coerce to contiguous float64 array of shape (n, d)."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.ascontiguousarray(X, dtype=np.float64)
    return X



# def _sinkhorn_fast(a, b, C, reg,
#                    numItermax=2000, stopThr=1e-9,
#                    screen_tau=1e-3,  # 1e-2..1e-4 are common
#                    scaling_steps=5,   # epsilon-scaling levels
#                    verbose=True):
#     """
#     Robust, fast entropic OT solver:
#       - scale C to median(C)=1
#       - try Screenkhorn (if present), else epsilon-scaling, else stabilized, else classic
#     Returns:
#       Pi  : transport plan
#       med : cost scale factor (so caller can undo scaling in objective if needed)
#     """
#     # --- scale cost to stabilize numerics (so reg has O(1) scale) ---
#     C = np.asarray(C, dtype=np.float64)
#     med = float(np.median(C))
#     if not np.isfinite(med) or med <= 0:
#         med = 1.0
#     C_ = C / med
#     reg_ = reg / med
#
#     # 1) Screenkhorn (fast on big histograms)
#     try:
#         Pi = ot.bregman.screenkhorn(
#             a, b, C_, reg_,
#             tau=screen_tau,
#             numItermax=numItermax,
#             stopThr=stopThr,
#             verbose=verbose
#         )
#         return Pi, med
#     except Exception:
#         pass
#
#     # 2) Epsilon-scaling (coarse-to-fine regularization)
#     try:
#         # schedule from larger to target reg
#         reg0 = max(reg_, 1.0)          # start at ~1 (or higher) for very coarse coupling
#         regs = np.geomspace(reg0, reg_, num=scaling_steps)
#         u = v = None
#         for r in regs:
#             Pi = ot.bregman.sinkhorn(
#                 a, b, C_, r,
#                 numItermax=numItermax//2,
#                 stopThr=max(stopThr, 1e-8),
#                 warmstart=(u, v) if (u is not None and v is not None) else None
#             )
#             # extract duals for warm start if available in your POT version
#             # (POT doesn't always return u,v; if not, just keep Pi)
#             u = v = None
#         return Pi, med
#     except Exception:
#         pass
#
#     # 3) Stabilized Sinkhorn (log-domain)
#     try:
#         Pi = ot.bregman.sinkhorn_stabilized(
#             a, b, C_, reg_,
#             numItermax=numItermax,
#             stopThr=stopThr,
#             tau=1e-3,            # stronger stabilization
#             verbose=verbose
#         )
#         return Pi, med
#     except Exception:
#         pass
#
#     # 4) Classic Sinkhorn (last resort)
#     Pi = ot.bregman.sinkhorn(
#         a, b, C_, reg_,
#         numItermax=max(numItermax, 5000),
#         stopThr=stopThr
#     )
#     return Pi, med


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

def _sinkhorn_stable(a, b, C,
                     reg=None,                 # ε on the *scaled* cost
                     numItermax=5000,
                     stopThr=1e-5,
                     tau=1e-3,                 # stabilization; smaller => more stable, slower
                     use_eps_scaling=True,
                     verbose=False):
    """
    Returns: (Pi, info) with cost scaled by median; ε also on that scale.
    """
    # 1) rescale the cost to O(1)
    C = np.asarray(C, dtype=np.float64)
    nz = C[C > 0]
    med = float(np.median(nz)) if nz.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = float(np.mean(nz)) if nz.size else 1.0
        if med <= 0: med = 1.0
    Cscaled = C / med

    # 2) set a sensible ε on the *scaled* cost
    eps = 0.2 if (reg is None) else float(reg)  # try 0.05–0.5 range
    # 3) choose stabilized variant
    info = {"scaled_cost_median": med, "eps_scaled": eps, "method": None}

    try:
        if use_eps_scaling:
            # Most robust
            info["method"] = "sinkhorn_epsilon_scaling"
            Pi = ot.bregman.sinkhorn_epsilon_scaling(
                a, b, Cscaled, reg=eps,
                numItermax=numItermax, stopThr=stopThr,
                verbose=verbose
            )
        else:
            info["method"] = "sinkhorn_stabilized"
            Pi = ot.bregman.sinkhorn_stabilized(
                a, b, Cscaled, reg=eps,
                tau=tau, numItermax=numItermax, stopThr=stopThr,
                verbose=verbose
            )
        # NaN/inf guard
        if not np.all(np.isfinite(Pi)):
            raise FloatingPointError("non-finite Pi")
        return Pi, info
    except Exception as e:
        # Soften ε and try plain sinkhorn
        if verbose:
            print(f"[sinkhorn] stabilized failed ({e}); retrying with plain sinkhorn and larger eps")
        eps2 = eps * 2.0
        try:
            Pi = ot.sinkhorn(a, b, Cscaled, reg=eps2,
                             numItermax=numItermax, stopThr=stopThr, verbose=verbose)
            if not np.all(np.isfinite(Pi)):
                raise FloatingPointError("non-finite Pi (plain)")
            info.update({"method": "sinkhorn_plain", "eps_scaled": eps2})
            return Pi, info
        except Exception as e2:
            # Last resort: exact EMD (may be slow)
            if verbose:
                print(f"[sinkhorn] plain failed ({e2}); falling back to EMD")
            Pi = ot.emd(a, b, C)  # use *unscaled* cost for EMD
            info.update({"method": "emd_fallback", "eps_scaled": None})
            return Pi, info


def entropic_barycenter_particles(
    particle_sets,
    reg: float = 5e-2,          # kept for signature compatibility; not used here
    n_bary: int = 512,
    seed: int = 0,
    allow_replace_if_needed: bool = True,
):
    """
    Free-support Wasserstein barycenter via POT's LP solver.
    Returns:
      Xb: (n_bary, d) barycenter support
      wb: (n_bary,) uniform weights
    """
    rng = np.random.default_rng(seed)

    # 1) numeric 2-D clouds
    Xs, d = _as_2d_clouds(particle_sets)

    # 2) per-measure weights (uniform)
    as_ = [np.ones(len(Xi), dtype=float) / float(len(Xi)) for Xi in Xs]

    # 3) init support by subsampling pooled points
    pooled = np.vstack(Xs)
    N = pooled.shape[0]
    if n_bary > N:
        if allow_replace_if_needed:
            idx = rng.choice(N, size=n_bary, replace=True)
        else:
            n_bary = N
            idx = np.arange(N)
    else:
        idx = rng.choice(N, size=n_bary, replace=False)
    X_init = pooled[idx]

    # 4) uniform barycenter weights
    wb = np.ones(n_bary, dtype=float) / float(n_bary)

    # 5) free-support barycenter (unregularized W2)
    Xb = ot.lp.free_support_barycenter(
        measures_locations=Xs,
        measures_weights=as_,
        X_init=X_init,
        b=wb,
        numItermax=500,
        stopThr=1e-7,
        verbose=True,
    )
    # TODO: Why return wb?
    return Xb, wb


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

     #a = _normalize_weights(a_src)
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

    else:
        raise ValueError("method must be 'sinkhorn' or 'emd'")
    # compute barycentric map; scaling of C does not affect Pi @ X_tgt
    denom = Pi.sum(axis=1, keepdims=True) + 1e-16 #a[:, None] + 1e-16

    Txs = (Pi @ X_tgt) / denom  # (n_s, d)

    return Txs

def tangent_from_barycentric(X_src: np.ndarray, Txs: np.ndarray) -> np.ndarray:
    return Txs - X_src



# === Validation helpers (paste near top of 02_fit_barycenter_and_ot.py) ===
import numpy as np
import ot

# def reconstruct_T_from_params(params):
#     """
#     Rebuilds a callable T(x) from a param dict produced in (A).
#     Supports:
#       - base convex map: {A:(J,d), w:(J,), c:(J,), eps:float}
#       - base plus mean shift: {kind:"base_plus_shift", base:<base_dict>, b:(d,)}
#       - base plus one extra bump: {kind:"base_plus_bump", base:<base_dict>, a:(d,), c:float, amp:float}
#       - composite: combine both patterns as needed
#     """
#     if "kind" not in params:
#         # base convex map
#         A = np.asarray(params["A"]); w = np.asarray(params["w"])
#         c = np.asarray(params["c"]); eps = float(params["eps"])
#         def T(x):
#             # x = np.atleast_2d(x)
#             x = _as_float2d(x)  # (n,d)
#             out = x.copy()
#             s = x @ A.T - c[None,:]
#             s_pos = np.maximum(s, 0.0)
#             out += (2.0*eps) * (s_pos @ (w[:,None]*A))
#             return out.squeeze()
#         return T
#
#     kind = params["kind"]
#     if kind == "base_plus_shift":
#         T0 = reconstruct_T_from_params(params["base"])
#         b  = np.asarray(params["b"])
#         def T(x):
#             y = T0(x); y = np.atleast_2d(y); y += b; return y.squeeze()
#         return T
#
#     if kind == "base_plus_bump":
#         T0 = reconstruct_T_from_params(params["base"])
#         a  = np.asarray(params["a"]); a = a/np.linalg.norm(a)
#         c  = float(params["c"]); amp = float(params["amp"])
#         def T(x):
#             x = np.atleast_2d(x)
#             y = T0(x)
#             s = np.maximum((x @ a) - c, 0.0)[:,None]
#             y += 2.0*amp*(s @ a[None,:])
#             return y.squeeze()
#         return T
#
#     if kind == "composite":
#         # expects keys: base, b (optional), a,c,amp (optional)
#         T0 = reconstruct_T_from_params(params["base"])
#         b  = np.asarray(params.get("b", None)) if ("b" in params) else None
#         add_bump = ("a" in params) and ("c" in params) and ("amp" in params)
#         if add_bump:
#             a  = np.asarray(params["a"]); a = a/np.linalg.norm(a)
#             c0 = float(params["c"]); amp = float(params["amp"])
#         def T(x):
#             x = np.atleast_2d(x)
#             y = T0(x)
#             if b is not None: y += b
#             if add_bump:
#                 s = np.maximum((x @ a) - c0, 0.0)[:,None]
#                 y += 2.0*amp*(s @ a[None,:])
#             return y.squeeze()
#         return T
#
#     raise ValueError(f"Unknown param kind: {params['kind']}")

def l2_map_error(Xb, wb, T_hat_Xb, T_true_fn, return_rel=True):
    T_true_Xb = T_true_fn(Xb)
    err2 = np.sum(wb * np.sum((T_hat_Xb - T_true_Xb)**2, axis=1))
    if return_rel:
        disp2 = np.sum(wb * np.sum((T_true_Xb - Xb)**2, axis=1)) + 1e-16
        return float(err2), float(err2/disp2)
    return float(err2)

def w2_pushforward_discrepancy(T_hat_Xb, wb, Y_true, reg=0.1):
    X1 = _as_float2d(T_hat_Xb)
    X2 = _as_float2d(Y_true)
    a = wb
    b = np.ones(Y_true.shape[0]) / Y_true.shape[0]
    C = ot.dist(X1, X2, metric='euclidean')**2
    pi = ot.bregman.sinkhorn(a, b, C, reg)
    return float(np.sum(pi * C))

# def map_cost_vs_sinkhorn(Xb, wb, T_hat_Xb, Y_true, reg=0.1):
#     X1 = _as_float2d(T_hat_Xb)
#     X2 = _as_float2d(Y_true)
#
#     a = wb
#     b = np.ones(X2.shape[0]) / Y_true.shape[0]
#     C = ot.dist(Xb, X2, metric='euclidean')**2
#     pi_star = ot.bregman.sinkhorn(a, b, C, reg)
#     sink_cost = float(np.sum(pi_star * C))
#     map_cost  = float(np.sum(wb * np.sum((X1 - Xb)**2, axis=1)))
#     return map_cost, sink_cost, map_cost - sink_cost
#
# def map_cost_vs_ot(
#     Xb, wb, T_hat_Xb, Y_true, *,
#     method: str = "emd",         # "emd" (exact) or "sinkhorn"
#     reg: float = 0.01,           # only used if method="sinkhorn"
#     n_co: int | None = None,     # optional coreset size for Y_true (e.g., 600–1000)
#     seed: int = 0
# ):
#     """
#     Compare:
#       - map_cost:  ∑_i w_i ||T_hat_Xb[i] - Xb[i]||^2     (energy of the estimated map)
#       - ot_cost:   OT cost between (Xb, w) and Y_true using either EMD or Sinkhorn
#     Returns: (map_cost, ot_cost, map_cost - ot_cost)
#     """
#     # coerce shapes/dtypes
#     Xb        = _as_float2d(Xb)
#     T_hat_Xb  = _as_float2d(T_hat_Xb)
#
#     # optional coreset for speed
#     Y2, b = _uniform_coreset(Y_true, n_co=n_co, seed=seed)
#
#     # ground cost between barycenter support and target cloud
#     C = ot.dist(Xb, Y2, metric="euclidean")**2  # (n_bary, n_tgt)
#
#     # unregularized OT plan (or Sinkhorn if requested)
#     if method.lower() == "emd":
#         # exact network simplex
#         pi_star = ot.emd(wb, b, C)
#     elif method.lower() == "sinkhorn":
#         pi_star = ot.bregman.sinkhorn(wb, b, C, reg)
#     else:
#         raise ValueError("method must be 'emd' or 'sinkhorn'")
#
#     # OT cost using the plan
#     ot_cost = float(np.sum(pi_star * C))
#
#     # energy of the estimated map (wb-weighted L2^2 displacement on Xb)
#     diff = T_hat_Xb - Xb
#     map_cost = float(np.sum((diff**2) * wb[:, None]))
#
#     return map_cost, ot_cost, map_cost - ot_cost


def monotonicity_check_sampled(T, X, n_pairs=4000, seed=0, tol=1e-12, batch=2000):
    """
    Check (ε-)cyclical monotonicity of a map T on random pairs from X:
        (T(x1)-T(x2)) · (x1-x2) >= -tol
    Returns:
        mean_dot   : average (T(x1)-T(x2))·(x1-x2) over sampled pairs
        frac_viol  : fraction of pairs violating the inequality by more than tol
    Notes:
        - Enforces 2-D float64 shapes (so d=1 is handled as (n,1)).
        - Evaluates T in batches to keep memory bounded.
    """
    rng = np.random.default_rng(seed)
    X = _as_float2d(X)                 # (n, d)
    n, d = X.shape

    # sample pairs (with replacement is fine)
    i1 = rng.integers(0, n, size=n_pairs)
    i2 = rng.integers(0, n, size=n_pairs)

    # gather pairs in batches
    dots = []
    for s in range(0, n_pairs, batch):
        e = min(s + batch, n_pairs)
        X1 = X[i1[s:e], :]            # (m, d)
        X2 = X[i2[s:e], :]            # (m, d)

        # make sure T returns (m, d)
        TX1 = _as_float2d(T(X1))      # (m, d)
        TX2 = _as_float2d(T(X2))      # (m, d)

        # dot products rowwise
        D  = (TX1 - TX2) * (X1 - X2)  # (m, d)
        dots.append(np.sum(D, axis=1))# (m,)

    dots = np.concatenate(dots, axis=0)  # (n_pairs,)
    mean_dot  = float(np.mean(dots))
    frac_viol = float(np.mean(dots < -tol))
    return mean_dot, frac_viol


import numpy as np


# def _as_float2d(x):
#     x = np.asarray(x, dtype=np.float64)
#     if x.ndim == 1:
#         x = x[None, :]
#     return x

def reconstruct_T_from_params(params):
    """
    Rebuild T(x) matching your Phase-II generator.
    Supported fields in `params` (some optional depending on scenario):
      - 'A': (J,d)   directions (unit rows)
      - 'w': (J,)    nonnegative weights (often L1-normalized)
      - 'c': (J,)    thresholds
      - 'eps': float strength
      - 'beta': float smoothness for softplus/sigmoid
      - 'mean_preserved': bool   (True if generator centered the drift)
      - 'drift_centered': (d,) or None  (vector subtracted to make mean-preserving)
      - 'add_shift_b': (d,) or None     (extra constant shift b)
      - 'kind': one of {'ic','aligned','reweight','shape_reweight',
                        'relocate','shape_relocate','novel','mean',
                        'composite','corr_shift'}  # informational
    Returns a vectorized callable T(x) for x in R^d.
    """
    A   = np.asarray(params["A"],   dtype=np.float64)   # (J,d)
    w   = np.asarray(params["w"],   dtype=np.float64)   # (J,)
    c   = np.asarray(params["c"],   dtype=np.float64)   # (J,)
    eps = float(params.get("eps", 0.15))
    beta= float(params.get("beta", 2.0))

    # Optional bits
    mean_preserved = bool(params.get("mean_preserved", False))
    drift_centered = params.get("drift_centered", None)
    if drift_centered is not None:
        drift_centered = np.asarray(drift_centered, dtype=np.float64)
    bshift = params.get("add_shift_b", None)
    if bshift is not None:
        bshift = np.asarray(bshift, dtype=np.float64)

    # Base raw map
    def T_raw(x):
        X = _as_float2d(x)                       # (n,d)
        z = X @ A.T - c[None, :]                 # (n,J)
        g = softplus_beta(z, beta) * sigmoid_beta(z, beta)  # (n,J)
        # (w[:,None]*A) is (J,d); g @ (...) is (n,d)
        return X + (2.0 * eps) * (g @ (w[:, None] * A))

    # Wrap with mean-preserving and optional shift
    def T_use(x):
        Y = T_raw(x)
        if mean_preserved and (drift_centered is not None):
            Y = Y - drift_centered[None, :]
        if bshift is not None:
            Y = Y + bshift[None, :]
        return Y

    return T_use



def kmeans_coreset(Y: np.ndarray, n_co: int, seed: int = 0):
    Y = np.asarray(Y, float)
    n = len(Y)
    n_co_eff = min(max(2, n_co), n)
    if n_co_eff == n:
        # no compression
        return Y, np.ones(n) / n

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_co_eff, n_init=5, random_state=seed).fit(Y)
    centers = km.cluster_centers_.astype(float)
    labels  = km.labels_
    counts  = np.bincount(labels, minlength=n_co_eff).astype(float)
    w = counts / counts.sum()  # == counts / n
    return centers, w
