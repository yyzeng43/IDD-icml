import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl

# =========================
# CONFIG YOU EDIT
# =========================
D_LIST = [1, 5, 10, 50]
K_LIST = [4]
N_POINTS_LIST = [50, 100, 300]
SCENARIOS = ["barycenter", "mm_reweight", "copula_shift"]

# ALPHAS = [0.50, 0.30, 0.20, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]
ALPHAS = np.unique(np.concatenate([
    np.linspace(0.50, 0.10, 20),
    np.linspace(0.10, 0.02, 20),
    np.linspace(0.02, 0.005, 20),
    np.linspace(0.005, 0.001, 20),
])).tolist()

import os
DATA_ROOT = Path(os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent / "data" / "continuous"))
OUT_ROOT = DATA_ROOT / "tradeoff_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# Folder names under each cfg folder
FLOW_METHOD_SUBDIR = {
    "F-CPD":  "FRECHET_rff",
    "Scan-B": "ScanB_rff",
    "NEWMA":  "NEWMA_rff",
    "Ours":     "OT",              # OT is multi-stat (T2/SPE) in *_OT_*_scores.*
}

NONFLOW_METHOD_SUBDIR = {
    "Log-KDE": "KDE_full",        # multi-stat (T2/SPE) in *_series.*
    # Shewhart handled by d rule
}


# =========================
# FILE PATTERNS + STAT COLS
# =========================
# All single-stat methods use column "stat" in *_series.*
# Multi-stat methods (OT, Log-KDE) use T2/SPE with OR rule.
METHOD_SPECS = {
    "F-CPD": {
        "ic_glob": "*_IC_series.*",
        "oc_glob": "*_{scenario}_series.*",
        "stat_cols": ["stat"],
        "combine": "single",   # single-stat
    },
    "Scan-B": {
        "ic_glob": "*_IC_series.*",
        "oc_glob": "*_{scenario}_series.*",
        "stat_cols": ["stat"],
        "combine": "single",
    },
    "NEWMA": {
        "ic_glob": "*_IC_series.*",
        "oc_glob": "*_{scenario}_series.*",
        "stat_cols": ["stat"],
        "combine": "single",
    },
    "Shewhart": {
        "ic_glob": "*_IC_series.*",
        "oc_glob": "*_{scenario}_series.*",
        "stat_cols": ["stat"],
        "combine": "single",
    },
    "Log-KDE": {
        "ic_glob": "*_IC_scores.*",
        "oc_glob": "*_{scenario}*_KDE_full_OC_scores.*",
        "stat_cols": ["T2", "SPE"],
        "combine": "OR",       # alarm if any exceeds its own UCL
    },
    "Ours": {
        "ic_glob": "*_OT_IC_scores.*",
        # prefer scenario-specific match; fallback handled in code if none found
        "oc_glob": "*{scenario}*_OT_OC_scores.*",
        "oc_glob_fallback": "*_OT_OC_scores.*",
        "stat_cols": ["T2", "SPE"],
        "combine": "OR",
    },
}


REP_RE = re.compile(r"rep(\d+)", re.IGNORECASE)


# =========================
# IO + SERIES HELPERS
# =========================
def read_table(fp: Path) -> pd.DataFrame:
    if fp.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(fp)
    else:
        df = pd.read_csv(fp)

    # remove fully blank rows (common cause of "blank series")
    df = df.dropna(how="all").copy()

    # idx
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1)

    # phase
    if "phase" not in df.columns:
        # OT score exports often omit phase; treat as Phase II
        df["phase"] = "II"
    df["phase"] = df["phase"].astype(str).str.strip()

    return df


def index_by_rep(files: List[Path]) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for fp in files:
        m = REP_RE.search(fp.name)
        if m:
            out[int(m.group(1))] = fp
    return out


def collect_files(method_dir: Path, scenario: str, spec: Dict) -> Tuple[List[Path], List[Path]]:
    method_dir = Path(method_dir)
    ic_files = sorted(method_dir.glob(spec["ic_glob"]))

    oc_pat = spec["oc_glob"].format(scenario=scenario)
    oc_files = sorted(method_dir.glob(oc_pat))

    # OT may not include scenario token in filename; allow fallback
    if len(oc_files) == 0 and "oc_glob_fallback" in spec:
        oc_files = sorted(method_dir.glob(spec["oc_glob_fallback"]))

    if len(ic_files) == 0:
        raise FileNotFoundError(f"No IC files in {method_dir} with pattern '{spec['ic_glob']}'")
    if len(oc_files) == 0:
        raise FileNotFoundError(f"No OC files in {method_dir} with pattern '{oc_pat}' (and fallback if provided)")

    return ic_files, oc_files


def phase_df(df: pd.DataFrame, phase: str, stat_cols: List[str]) -> pd.DataFrame:
    """
    Return Phase-I or Phase-II sub-DF with chosen stat columns coerced to numeric.
    Drops rows where ALL chosen stat columns are NaN (blank/warmup).
    """
    sub = df.loc[df["phase"].astype(str) == str(phase), ["idx"] + stat_cols].copy()

    # numeric coercion fixes blank entries in CSV
    for c in stat_cols:
        if c not in sub.columns:
            raise ValueError(f"Missing column '{c}' in DF. Available: {list(df.columns)}")
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # drop rows where all stats are NaN (blank warmup)
    mask_all_nan = sub[stat_cols].isna().all(axis=1)
    sub = sub.loc[~mask_all_nan].copy()

    return sub


def compute_ucls_from_ic_pool(
    ic_files_by_rep: Dict[int, Path],
    reps: List[int],
    *,
    ucl_phase: str,
    stat_cols: List[str],
    alpha: float,
) -> Dict[str, float]:
    """
    Compute per-stat UCL as quantile(1-alpha) over pooled IC (Phase ucl_phase).
    """
    ucls: Dict[str, float] = {}
    for c in stat_cols:
        pool = []
        for rep in reps:
            df = read_table(ic_files_by_rep[rep])
            sub = phase_df(df, ucl_phase, [c])
            pool.append(sub[c].to_numpy(dtype=float))
        pool = np.concatenate(pool, axis=0) if pool else np.zeros((0,), float)
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            ucls[c] = float("inf")
        else:
            ucls[c] = float(np.quantile(pool, 1.0 - float(alpha)))
    return ucls


def run_length_alarm(
    df_phase: pd.DataFrame,
    *,
    stat_cols: List[str],
    ucls: Dict[str, float],
    combine: str,
) -> Tuple[int, int]:
    """
    Returns (RL, H) where:
      RL = first time index (1-based within this phase) where alarm triggers, else H+1
      H = horizon length (#rows in df_phase)
    combine:
      - "single": use stat_cols[0] > UCL
      - "OR": alarm if any stat_col > its UCL at the same time index
    """
    H = int(len(df_phase))
    if H == 0:
        return 1, 0

    if combine == "single":
        c = stat_cols[0]
        x = df_phase[c].to_numpy(dtype=float)
        exc = np.isfinite(x) & (x > float(ucls[c]))
    elif combine == "OR":
        exc = np.zeros(H, dtype=bool)
        for c in stat_cols:
            x = df_phase[c].to_numpy(dtype=float)
            exc = exc | (np.isfinite(x) & (x > float(ucls[c])))
    else:
        raise ValueError(f"Unknown combine rule: {combine}")

    idx = np.where(exc)[0]
    RL = int(idx[0] + 1) if idx.size else int(H + 1)
    return RL, H


def trigger_fraction(stat_dfII: pd.DataFrame, ucls: dict, stat_cols: list[str], *, combine: str = "single") -> float:
    """
    stat_dfII: Phase-II dataframe with stat columns
    ucls: dict of UCL per stat col (for single-stat, only one entry)
    combine: "single" or "OR" (for multi-stat OT/Log-KDE)
    Returns mean trigger rate over Phase-II rows (NaNs never trigger).
    """
    H = len(stat_dfII)
    if H == 0:
        return float("nan")

    if len(stat_cols) == 1 and combine != "OR":
        c = stat_cols[0]
        x = pd.to_numeric(stat_dfII[c], errors="coerce").to_numpy(float)
        U = float(ucls[c])
        trig = np.isfinite(x) & (x > U)
        return float(np.mean(trig))

    # multi-stat OR
    trig_any = np.zeros(H, dtype=bool)
    for c in stat_cols:
        x = pd.to_numeric(stat_dfII[c], errors="coerce").to_numpy(float)
        U = float(ucls[c])
        trig_any |= (np.isfinite(x) & (x > U))
    return float(np.mean(trig_any))


# =========================
# TRADEOFF COMPUTATION + PLOT
# =========================
def compute_tradeoff_table_for_scenario(
    method_dirs: Dict[str, Path],
    *,
    scenario: str,
    alphas: List[float],
    ucl_phase: str = "II",
    eval_phase: str = "II",
    delay_mode: str = "censored",  # only "censored" implemented here (consistent with your earlier runs)
) -> pd.DataFrame:
    rows = []

    for method, mdir in method_dirs.items():
        if method not in METHOD_SPECS:
            print(f"[WARN] No METHOD_SPECS for {method}; skip.")
            continue

        spec = METHOD_SPECS[method]
        stat_cols = spec["stat_cols"]
        combine = spec["combine"]

        # Collect files
        try:
            ic_files, oc_files = collect_files(mdir, scenario, spec)
        except Exception as e:
            print(f"[WARN] {method}: cannot collect files for scenario={scenario}: {e}")
            continue

        ic_by_rep = index_by_rep(ic_files)
        oc_by_rep = index_by_rep(oc_files)
        reps = sorted(set(ic_by_rep.keys()).intersection(set(oc_by_rep.keys())))
        if len(reps) == 0:
            print(f"[WARN] {method}: no overlapping reps between IC and {scenario}; skip.")
            continue

        for alpha in alphas:
            alpha = float(alpha)

            # per-stat UCLs
            ucls = compute_ucls_from_ic_pool(ic_by_rep, reps, ucl_phase=ucl_phase, stat_cols=stat_cols, alpha=alpha)

            # IC RLs -> ARL0
            RL0_list = []
            H0_list = []
            for rep in reps:
                df_ic = read_table(ic_by_rep[rep])
                df0 = phase_df(df_ic, eval_phase, stat_cols)
                rl0, H0 = run_length_alarm(df0, stat_cols=stat_cols, ucls=ucls, combine=combine)
                RL0_list.append(rl0)
                H0_list.append(H0)

            # RL0 = np.array(RL0_list, float)
            # ARL0 = float(np.mean(RL0)) if RL0.size else float("nan")
            # FAR = float(1.0 / ARL0) if np.isfinite(ARL0) and ARL0 > 0 else float("nan")
            #
            # OC RLs -> ARL1 / delay + detection rate
            RL1_list = []
            det_list = []
            H1_list = []
            trig_frac_list = []
            for rep in reps:
                df_oc = read_table(oc_by_rep[rep])
                df1 = phase_df(df_oc, eval_phase, stat_cols)
                rl1, H1 = run_length_alarm(df1, stat_cols=stat_cols, ucls=ucls, combine=combine)
                RL1_list.append(rl1)
                H1_list.append(H1)

                # For OC: compute trigger fraction over ALL Phase-II samples
                trig_frac = trigger_fraction(df_oc, ucls, stat_cols, combine=combine)
                trig_frac_list.append(trig_frac)

                # det_list.append(1.0 if (H1 > 0 and rl1 <= H1) else 0.0)

            RL1 = np.array(RL1_list, float)
            mean_delay = float(np.mean(RL1)) if RL1.size else float("nan")
            det_rate = float(np.mean(det_list)) if len(det_list) else float("nan")
            #
            # row = dict(
            #     method=method,
            #     scenario=scenario,
            #     alpha=alpha,
            #     ARL0=ARL0,
            #     FAR=FAR,
            #     mean_delay=mean_delay,
            #     detection_rate=det_rate,
            #     n_rep=int(len(reps)),
            #     combine=combine,
            # )

            # ---- IC RLs -> ARL0 + FAR
            RL0 = np.asarray(RL0_list, float)
            ARL0_mean = float(np.mean(RL0)) if RL0.size else float("nan")
            ARL0_std = float(np.std(RL0, ddof=1)) if RL0.size > 1 else 0.0

            FAR_i = 1.0 / RL0
            FAR_mean = float(np.mean(FAR_i)) if FAR_i.size else float("nan")
            FAR_std = float(np.std(FAR_i, ddof=1)) if FAR_i.size > 1 else 0.0

            # ---- OC RLs -> delay stats + detection stats
            RL1 = np.asarray(RL1_list, float)
            delay_mean = float(np.mean(RL1)) if RL1.size else float("nan")
            delay_std = float(np.std(RL1, ddof=1)) if RL1.size > 1 else 0.0

            det = np.asarray(det_list, float)
            # det_rate = float(np.mean(det)) if det.size else float("nan")
            det_rate = float(np.mean(trig_frac_list)) if trig_frac_list else float("nan")

            # Binomial standard error is often more meaningful than std for detection rate
            det_se = float(np.sqrt(det_rate * (1.0 - det_rate) / det.size)) if det.size else float("nan")

            row = dict(
                method=method,
                scenario=scenario,
                alpha=float(alpha),

                # x-axis metrics
                ARL0_mean=ARL0_mean,
                ARL0_std=ARL0_std,
                FAR_mean=FAR_mean,
                FAR_std=FAR_std,

                # y-axis metrics
                detection_rate_mean=det_rate,
                detection_rate_se=det_se,
                delay_mean=delay_mean,
                delay_std=delay_std,

                n_rep=int(len(reps)),
                combine=combine,
            )

            # store UCLs (per stat)
            for c in stat_cols:
                row[f"UCL_{c}"] = float(ucls.get(c, float("nan")))

            rows.append(row)

            # store UCLs (per stat)
            for c in stat_cols:
                row[f"UCL_{c}"] = float(ucls.get(c, float("nan")))

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["scenario", "method", "alpha"]).reset_index(drop=True)
    return df


def plot_tradeoff_curves(df: pd.DataFrame, *, scenario: str, out_png: Path, title_prefix: str = ""):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[WARN] empty df for scenario={scenario}; skip plot {out_png}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=220)

    # Panel 1: detection rate vs FAR
    ax = axes[0]
    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("FAR")
        ax.plot(g["FAR"].to_numpy(), g["detection_rate"].to_numpy(),
                marker="o", linewidth=1.6, label=method)
    ax.set_xlabel("False alarm rate (1/ARL0)")
    ax.set_ylabel("Detection rate")
    ax.set_title(f"{title_prefix}{scenario}: Detection vs FAR")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8)

    # Panel 2: mean delay vs ARL0
    ax = axes[1]
    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("ARL0")
        ax.plot(g["ARL0"].to_numpy(), g["mean_delay"].to_numpy(),
                marker="o", linewidth=1.6, label=method)
    ax.set_xlabel("ARL0")
    ax.set_ylabel("Mean detection delay")
    # ax.set_title(f"{title_prefix}{scenario}: Delay vs ARL0")
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_tradeoff_curves_with_std(df: pd.DataFrame, *, scenario: str, out_png: Path, title_prefix: str = ""):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[WARN] empty df for scenario={scenario}; skip plot {out_png}")
        return

    # Sort per method for nice curves
    df = df.copy()
    df = df.sort_values(["method", "FAR_mean", "alpha"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=240, constrained_layout=True)

    # ---- shared style tweaks (no explicit colors)
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # =======================
    # Panel 1: Detection vs FAR
    # =======================
    ax = axes[0]
    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("FAR_mean")
        x = g["FAR_mean"].to_numpy(float)
        y = g["detection_rate_mean"].to_numpy(float)

        xerr = g["FAR_std"].to_numpy(float)
        yerr = g["detection_rate_se"].to_numpy(float)  # SE (recommended for rates)

        # Mean curve
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=5.5, label=method)

        # Errorbars (both x and y)
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt="none",
            capsize=2.8,
            elinewidth=1.0,
            alpha=0.65,
        )

        # Optional: light band for y uncertainty only (SE band)
        ylo = np.clip(y - yerr, 0.0, 1.0)
        yhi = np.clip(y + yerr, 0.0, 1.0)
        ax.fill_between(x, ylo, yhi, alpha=0.10)

    ax.set_xlabel("False alarm rate (mean of 1/RL0)")
    ax.set_ylabel("Detection rate (mean ± SE)")
    ax.set_title(f"{title_prefix}{scenario}: Detection vs FAR")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # =======================
    # Panel 2: Delay vs ARL0
    # =======================
    ax = axes[1]
    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("ARL0_mean")
        x = g["ARL0_mean"].to_numpy(float)
        y = g["delay_mean"].to_numpy(float)

        xerr = g["ARL0_std"].to_numpy(float)
        yerr = g["delay_std"].to_numpy(float)

        ax.plot(x, y, marker="o", linewidth=1.8, markersize=5.5, label=method)
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt="none",
            capsize=2.8,
            elinewidth=1.0,
            alpha=0.65,
        )
        ylo = np.maximum(y - yerr, 0.0)
        yhi = y + yerr
        ax.fill_between(x, ylo, yhi, alpha=0.10)

    ax.set_xlabel("ARL0 (mean RL0)")
    ax.set_ylabel("Mean detection delay ± SD")
    ax.set_title(f"{title_prefix}{scenario}: Delay vs ARL0")
    ax.set_xlim(left=0.0)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    fig.savefig(out_png)
    plt.close(fig)


def plot_tradeoff_curves_meanonly(
    df: pd.DataFrame, *,
    scenario: str,
    out_png: Path,
    title_prefix: str = "",
    arl0_max: float = 100.0,
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[WARN] empty df for scenario={scenario}; skip plot {out_png}")
        return

    df = df.copy()
    df = df[np.isfinite(df["ARL0_mean"]) & (df["ARL0_mean"] <= arl0_max)].copy()
    if df.empty:
        print(f"[WARN] all points filtered out by arl0_max={arl0_max} for {scenario}")
        return

    # --- Consistent marker assignment per method (print-friendly)

    methods = list(df["method"].dropna().unique())
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", ">", "<", "h", "*"])
    marker_map = {m: next(marker_cycle) for m in methods}

    # --- put this near the top of your script (before any plotting) ---
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=240, constrained_layout=True)

    for ax in axes:
        # Nature-ish: subtle grid (or turn off if you prefer)
        ax.grid(True, alpha=0.12, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3.2, width=0.8)

    # Helper: choose sparse marker placement (avoid clutter)
    def _markevery(npts: int) -> int:
        # roughly 6–10 markers per curve depending on length
        return max(1, int(np.ceil(npts / 8)))

    # -------- Panel 1: Detection vs FAR --------
    ax = axes[0]
    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("FAR_mean")
        x = g["FAR_mean"].to_numpy(float)
        y = g["detection_rate_mean"].to_numpy(float)

        keep = np.r_[True, np.diff(x) != 0]
        x, y = x[keep], y[keep]

        me = _markevery(len(x))
        ax.plot(
            x, y,
            linewidth=2.0,
            marker=marker_map.get(method, "o"),
            markersize=4.2,
            markevery=me,
            markerfacecolor="white",     # open markers (publication-friendly)
            markeredgewidth=1.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=method,
        )

    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Detection rate")
    # ax.set_title(f"{title_prefix}{scenario}: Detection vs FAR")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right", handlelength=2.2)

    # -------- Panel 2: Delay vs ARL0 --------
    ax = axes[1]
    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("ARL0_mean")
        x = g["ARL0_mean"].to_numpy(float)
        y = g["delay_mean"].to_numpy(float)

        keep = np.r_[True, np.diff(x) != 0]
        x, y = x[keep], y[keep]

        me = _markevery(len(x))
        ax.plot(
            x, y,
            linewidth=2.0,
            marker=marker_map.get(method, "o"),
            markersize=4.2,
            markevery=me,
            markerfacecolor="white",
            markeredgewidth=1.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=method,
        )

    ax.set_xlabel("ARL0")
    ax.set_ylabel("ARL1")
    # ax.set_title(f"{title_prefix}{scenario}: Delay vs ARL0")
    ax.set_xlim(0.0, arl0_max)
    ax.legend(frameon=False, fontsize=9, loc="upper left", handlelength=2.2)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



# def plot_tradeoff_curves_meanonly_twofigs(
#     df: pd.DataFrame, *,
#     scenario: str,
#     out_png: Path,              # treated as a "base name"
#     title_prefix: str = "",
#     arl0_max: float = 100.0,
# ):
#     out_png = Path(out_png)
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#
#     if df.empty:
#         print(f"[WARN] empty df for scenario={scenario}; skip plot {out_png}")
#         return
#
#     df = df.copy()
#     df = df[np.isfinite(df["ARL0_mean"]) & (df["ARL0_mean"] <= arl0_max)].copy()
#     if df.empty:
#         print(f"[WARN] all points filtered out by arl0_max={arl0_max} for {scenario}")
#         return
#
#     # --- outputs
#     out_det = out_png.with_name(f"{out_png.stem}_{scenario}_det_vs_far{out_png.suffix}")
#     out_del = out_png.with_name(f"{out_png.stem}_{scenario}_delay_vs_arl0{out_png.suffix}")
#
#     # --- consistent markers per method
#     methods = list(df["method"].dropna().unique())
#     marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", ">", "<", "h", "*"])
#     marker_map = {m: next(marker_cycle) for m in methods}
#
#     def _markevery(npts: int) -> int:
#         return max(1, int(np.ceil(npts / 8)))  # ~8 markers per curve
#
#     def _style_axes(ax):
#         ax.grid(True, alpha=0.12, linewidth=0.6)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         ax.tick_params(direction="out", length=3.2, width=0.8)
#
#     # --- put this near the top of your script (before any plotting) ---
#     mpl.rcParams.update({
#         "font.family": "Times New Roman",
#         "font.size": 18,
#         "legend.fontsize": 18,
#         "legend.title_fontsize": 18,
#     })
#     # =========================
#     # Figure 1: Detection vs FAR
#     # =========================
#     fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), dpi=240, constrained_layout=True)
#     _style_axes(ax)
#
#     for method, g in df.groupby("method", sort=False):
#         g = g.sort_values("FAR_mean")
#         x = g["FAR_mean"].to_numpy(float)
#         y = g["detection_rate_mean"].to_numpy(float)
#
#         keep = np.r_[True, np.diff(x) != 0]
#         x, y = x[keep], y[keep]
#         me = _markevery(len(x))
#
#         ax.plot(
#             x, y,
#             linewidth=2.0,
#             marker=marker_map.get(method, "o"),
#             markersize=4.2,
#             markevery=me,
#             markerfacecolor="white",
#             markeredgewidth=1.0,
#             solid_capstyle="round",
#             solid_joinstyle="round",
#             label=method,
#         )
#
#     ax.set_xlabel("False alarm rate")
#     ax.set_ylabel("Detection rate")
#     # ax.set_title(f"{title_prefix}{scenario}: Detection vs FAR")
#     ax.set_xlim(left=0.0)
#     ax.set_ylim(0.0, 1.02)
#     ax.legend(frameon=False, fontsize=18, loc="lower right", handlelength=2.2)
#
#     fig.savefig(out_det, bbox_inches="tight")
#     plt.close(fig)
#
#     # ======================
#     # Figure 2: Delay vs ARL0
#     # ======================
#     fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), dpi=240, constrained_layout=True)
#     _style_axes(ax)
#
#     for method, g in df.groupby("method", sort=False):
#         g = g.sort_values("ARL0_mean")
#         x = g["ARL0_mean"].to_numpy(float)
#         y = g["delay_mean"].to_numpy(float)
#
#         keep = np.r_[True, np.diff(x) != 0]
#         x, y = x[keep], y[keep]
#         me = _markevery(len(x))
#
#         ax.plot(
#             x, y,
#             linewidth=2.0,
#             marker=marker_map.get(method, "o"),
#             markersize=4.2,
#             markevery=me,
#             markerfacecolor="white",
#             markeredgewidth=1.0,
#             solid_capstyle="round",
#             solid_joinstyle="round",
#             label=method,
#         )
#
#     ax.set_xlabel("ARL0")
#     ax.set_ylabel("ARL1")
#     # ax.set_title(f"{title_prefix}{scenario}: Delay vs ARL0")
#     ax.set_xlim(0.0, arl0_max)
#     ax.legend(frameon=False, fontsize=18, loc="upper left", handlelength=2.2)
#
#     fig.savefig(out_del, bbox_inches="tight")
#     plt.close(fig)
#
#     print(f"[OK] saved:\n  {out_det}\n  {out_del}")



PUB = dict(
    base_font=12,      # 10–11 for Nature-like single column
    label_font=13.0,
    tick_font=11,
    legend_font=13.0,
    title_font=11.0,     # you said remove title; keep if needed later
    lw=2.2,
    ms=6.0,              # marker size
    mew=1.1,             # marker edge width
    grid_lw=0.7,
    spine_lw=0.9,
)

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": PUB["base_font"],
    "axes.labelsize": PUB["label_font"],
    "xtick.labelsize": PUB["tick_font"],
    "ytick.labelsize": PUB["tick_font"],
    "legend.fontsize": PUB["legend_font"],
    "axes.linewidth": PUB["spine_lw"],
})


def plot_tradeoff_curves_meanonly_twofigs(
    df: pd.DataFrame, *,
    scenario: str,
    out_png: Path,
    title_prefix: str = "",
    arl0_max: float = 100.0,
    fig_w_in: float = 3.45,     # single-column width (~88 mm)
    fig_h_in: float = 3.0,      # height per panel
    dpi: int = 300,
):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[WARN] empty df for scenario={scenario}; skip plot {out_png}")
        return

    df = df.copy()
    df = df[np.isfinite(df["ARL0_mean"]) & (df["ARL0_mean"] <= arl0_max)].copy()
    if df.empty:
        print(f"[WARN] all points filtered out by arl0_max={arl0_max} for {scenario}")
        return

    # outputs
    out_det = out_png.with_name(f"{out_png.stem}_{scenario}_det_vs_far{out_png.suffix}")
    out_del = out_png.with_name(f"{out_png.stem}_{scenario}_delay_vs_arl0{out_png.suffix}")

    methods = list(df["method"].dropna().unique())
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", ">", "<", "h", "*"])
    marker_map = {m: next(marker_cycle) for m in methods}

    def _markevery(npts: int) -> int:
        return max(1, int(np.ceil(npts / 7)))  # slightly denser markers

    def _style_axes(ax):
        ax.grid(True, alpha=0.18, linewidth=PUB["grid_lw"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(PUB["spine_lw"])
        ax.spines["bottom"].set_linewidth(PUB["spine_lw"])
        ax.tick_params(direction="out", length=4.0, width=0.9, pad=2.5)

    # ===== Figure 1 =====
    fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in), dpi=dpi, constrained_layout=True)
    _style_axes(ax)

    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("FAR_mean")
        x = g["FAR_mean"].to_numpy(float)
        y = g["detection_rate_mean"].to_numpy(float)

        keep = np.r_[True, np.diff(x) != 0]
        x, y = x[keep], y[keep]
        me = _markevery(len(x))

        ax.plot(
            x, y,
            linewidth=PUB["lw"],
            marker=marker_map.get(method, "o"),
            markersize=PUB["ms"],
            markevery=me,
            markerfacecolor="white",
            markeredgewidth=PUB["mew"],
            solid_capstyle="round",
            solid_joinstyle="round",
            label=method,
        )

    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Detection rate")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)

    ax.legend(
        frameon=False,
        loc="lower right",
        handlelength=2.6,
        markerscale=1.2,
        labelspacing=0.45,
        handletextpad=0.6,
        borderaxespad=0.35,
    )

    fig.savefig(out_det, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 2 =====
    fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in), dpi=dpi, constrained_layout=True)
    _style_axes(ax)

    for method, g in df.groupby("method", sort=False):
        g = g.sort_values("ARL0_mean")
        x = g["ARL0_mean"].to_numpy(float)
        y = g["delay_mean"].to_numpy(float)

        keep = np.r_[True, np.diff(x) != 0]
        x, y = x[keep], y[keep]
        me = _markevery(len(x))

        ax.plot(
            x, y,
            linewidth=PUB["lw"],
            marker=marker_map.get(method, "o"),
            markersize=PUB["ms"],
            markevery=me,
            markerfacecolor="white",
            markeredgewidth=PUB["mew"],
            solid_capstyle="round",
            solid_joinstyle="round",
            label=method,
        )

    ax.set_xlabel("ARL0")
    ax.set_ylabel("ARL1")
    ax.set_xlim(0.0, arl0_max)

    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=2.6,
        markerscale=1.2,
        labelspacing=0.45,
        handletextpad=0.6,
        borderaxespad=0.35,
    )

    fig.savefig(out_del, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved:\n  {out_det}\n  {out_del}")

# =========================
# BUILD METHOD DIRS PER CFG
# =========================
def build_method_dirs_for_cfg(*, cfg_flow: Path, cfg_nf: Path, d: int) -> Dict[str, Path]:
    method_dirs: Dict[str, Path] = {}

    # flow methods
    if cfg_flow.exists():
        for name, sub in FLOW_METHOD_SUBDIR.items():
            p = cfg_flow / sub
            if p.exists():
                method_dirs[name] = p

    # non-flow methods
    if cfg_nf.exists():
        for name, sub in NONFLOW_METHOD_SUBDIR.items():
            p = cfg_nf / sub
            if p.exists():
                method_dirs[name] = p

        # shewhart naming depends on d
        shewhart_sub = "Xbar" if int(d) == 1 else "HotellingT2"
        p_sh = cfg_nf / shewhart_sub
        if p_sh.exists():
            method_dirs["Shewhart"] = p_sh

    return method_dirs


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for n_point in N_POINTS_LIST:
        datadir = DATA_ROOT / f"n{n_point}"
        resdir_flow = datadir / f"results_flow_n{n_point}"
        resdir_nf = datadir / f"results_n{n_point}"

        if (not resdir_flow.exists()) and (not resdir_nf.exists()):
            print(f"[WARN] Neither results folder exists for n={n_point}:")
            print(f"       {resdir_flow}")
            print(f"       {resdir_nf}")
            continue

        for d in D_LIST:
            for K in K_LIST:
                cfg_name = f"phase_d{d}_K{K}"
                cfg_flow = resdir_flow / cfg_name
                cfg_nf = resdir_nf / cfg_name

                print(f"\n=== CONFIG {cfg_name} (n={n_point}) ===")
                method_dirs = build_method_dirs_for_cfg(cfg_flow=cfg_flow, cfg_nf=cfg_nf, d=d)

                if len(method_dirs) == 0:
                    print("  [WARN] no method dirs found; skip cfg.")
                    continue

                for scenario in SCENARIOS:
                    out_dir = OUT_ROOT / f"n{n_point}" / cfg_name / scenario
                    out_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        df = compute_tradeoff_table_for_scenario(
                            method_dirs,
                            scenario=scenario,
                            alphas=ALPHAS,
                            ucl_phase="II",
                            eval_phase="II",
                            delay_mode="censored",
                        )

                        # Save df with cfg/scenario in filename
                        df_csv = out_dir / f"tradeoff_n{n_point}_{cfg_name}_{scenario}.csv"
                        df_xlsx = out_dir / f"tradeoff_n{n_point}_{cfg_name}_{scenario}.xlsx"
                        df.to_csv(df_csv, index=False)
                        df.to_excel(df_xlsx, index=False)

                        # Plot
                        out_png = out_dir / f"tradeoff_n{n_point}_{cfg_name}_{scenario}.png"
                        # plot_tradeoff_curves_with_std(
                        #     df,
                        #     scenario=scenario,
                        #     out_png=out_png,
                        #     title_prefix=f"n={n_point}, {cfg_name} — ",
                        # )

                        plot_tradeoff_curves_meanonly_twofigs(
                            df,
                            scenario=scenario,
                            out_png=out_png,
                            title_prefix=f"n={n_point}, {cfg_name} — ",
                            arl0_max=100.0,
                        )



                        print(f"  [OK] {scenario}: saved df + plot to {out_dir}")

                    except Exception as e:
                        print(f"  [ERROR] failed scenario={scenario} cfg={cfg_name} n={n_point}: {type(e).__name__}: {e}")
