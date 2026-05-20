#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade-off curves (ARL0 vs ARL1) for *discrete* simulation results in:

Focus scenarios:
  - cat_m6          / M3_drift
  - dust_poisson_1d / P1_mix, P2_spike

IMPORTANT: This script supports TWO storage layouts:

(A) ATTR_CHART / C_CHART
    Each scenario/rep file contains BOTH Phase I and Phase II:
      .../<METHOD>/<SCEN>/scores_<METHOD>_<SCEN>_<rep>.csv
    => Use Phase I rows to build UCLs and estimate ARL0; use Phase II rows for ARL1.

(B) OT / KDE_FULL
    Phase I is pooled in a single training file:
      .../<METHOD>/train_<METHOD>_scores_phaseI.csv
    Phase II is stored per scenario/rep somewhere under .../<METHOD>/<SCEN>/...
    => Use train file for UCLs. For ARL0:
       - if IC Phase-II files exist, compute ARL0 from them;
       - else estimate ARL0 geometrically from Phase-I exceedance probability.

Multi-stat methods (OT/KDE_FULL): alarm if (T2 > UCL_T2) OR (SPE > UCL_SPE).

Outputs:
  .../results_tradeoff_discreteV1/...
    tradeoff_*.csv / .xlsx
    *_det_vs_far.png
    *_arl1_vs_arl0.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl


# =========================
# CONFIG (EDIT THESE)
# =========================
import os
DATA_ROOT = Path(os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent / "data" / "discrete"))

N_LIST = [50, 100, 300]

TARGETS = {
    "cat_m6":          ["M3_drift"],   # Ordered categorical drift (Fig. 2b)
    "dust_poisson_1d": ["P2_spike"],   # Poisson spike injection (Fig. 2a)
}

# Alpha grid
ALPHAS = np.unique(np.concatenate([
    np.linspace(0.30, 0.10, 12),
    np.linspace(0.10, 0.02, 12),
    np.linspace(0.02, 0.005, 12),
    np.linspace(0.005, 0.001, 10),
])).tolist()

# If you *do* have an explicit IC scenario folder for OT/KDE, set here (try "IC", "IC0", etc.)
IC_SCENARIO_CANDIDATES = ["IC", "IC0", "IC_0", "in_control", "incontrol"]

OUT_ROOT = DATA_ROOT / "tradeoff_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# METHODS
# =========================
METHOD_SPECS = {
    # Embedded Phase I+II per rep file
    "ATTR_CHART": dict(stat_cols=["max_abs_z"], combine="single", ic_source="embedded"),
    "C_CHART":    dict(stat_cols=["c"],         combine="single", ic_source="embedded"),

    # Train Phase I pooled file + scenario Phase II per rep file
    "KDE_FULL":   dict(stat_cols=["T2", "SPE"], combine="OR",     ic_source="train"),
    "OT":         dict(stat_cols=["T2", "SPE"], combine="OR",     ic_source="train"),
}

METHOD_SUBDIR = {
    "ATTR_CHART": "ATTR_CHART",
    "C_CHART": "C_CHART",
    "KDE_FULL": "KDE_FULL",
    "OT": "OT",
}

METHOD_DISPLAY = {
    "KDE_FULL": "Log-KDE",
    "OT": "Ours",          # if you want
}

# name patterns
REP_RE = re.compile(r"rep(\d+)", re.IGNORECASE)
TAIL_REP_RE = re.compile(r"_(\d+)(?:\.[^.]+)?$")  # ..._0.csv / ..._9.xlsx
PHASEI_TRAIN_RE = re.compile(r"train_.*phasei", re.IGNORECASE)


# =========================
# IO HELPERS
# =========================
def read_table(fp: Path) -> pd.DataFrame:
    if fp.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(fp)
    else:
        df = pd.read_csv(fp)

    df = df.dropna(how="all").copy()

    # idx
    if "idx" not in df.columns:
        df["idx"] = np.arange(1, len(df) + 1)

    # phase
    if "phase" not in df.columns:
        df["phase"] = "II"
    df["phase"] = df["phase"].astype(str).str.strip()

    return df


def rep_from_name(name: str) -> Optional[int]:
    m = REP_RE.search(name)
    if m:
        return int(m.group(1))
    m = TAIL_REP_RE.search(name)
    if m:
        return int(m.group(1))
    return None


def _candidate_files(method_dir: Path) -> List[Path]:
    exts = {".csv", ".xlsx", ".xls"}
    out = []
    for fp in method_dir.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in exts:
            continue
        nl = fp.name.lower()
        if "summary" in nl or "tradeoff" in nl:
            continue
        if nl.startswith("~$"):  # excel temp
            continue
        out.append(fp)
    return out


def _pick_file(files: List[Path], want_cols: List[str], *, prefer_phaseII: bool = True) -> Optional[Path]:
    """
    Pick the best file among candidates for one rep.
    Preference:
      1) contains all want_cols
      2) contains any want_cols
      3) contains 'stat'
    Additionally: avoid train phaseI when searching for Phase II.
    """
    if not files:
        return None

    def _cols(fp: Path) -> List[str]:
        try:
            if fp.suffix.lower() in (".xlsx", ".xls"):
                d = pd.read_excel(fp, nrows=3)
            else:
                d = pd.read_csv(fp, nrows=3)
            return list(d.columns)
        except Exception:
            return []

    # exclude train phaseI for Phase II picking
    files2 = []
    for fp in files:
        nl = fp.name.lower()
        if prefer_phaseII and PHASEI_TRAIN_RE.search(nl):
            continue
        files2.append(fp)
    if files2:
        files = files2

    for fp in files:
        cols = _cols(fp)
        if all(c in cols for c in want_cols):
            return fp
    for fp in files:
        cols = _cols(fp)
        if any(c in cols for c in want_cols):
            return fp
    for fp in files:
        cols = _cols(fp)
        if "stat" in cols:
            return fp
    return files[0]


def phase_df(df: pd.DataFrame, phase: str, want_cols: List[str]) -> pd.DataFrame:
    """
    Return Phase sub-DF with idx + stat columns coerced to numeric.
    If phase not found, returns whole df with usable columns.
    """
    phase = str(phase).upper()
    ph = df["phase"].astype(str).str.upper()
    sub = df.loc[ph == phase].copy()
    if sub.empty:
        sub = df.copy()

    # choose columns present
    cols = ["idx"] + [c for c in want_cols if c in sub.columns]
    if len(cols) == 1:
        if "stat" in sub.columns:
            cols = ["idx", "stat"]
        else:
            # pick first numeric column besides idx/phase
            cand = [c for c in sub.columns if c not in ("idx", "phase")]
            num_cols = []
            for c in cand:
                x = pd.to_numeric(sub[c], errors="coerce")
                if np.isfinite(x).any():
                    num_cols.append(c)
            if not num_cols:
                raise ValueError(f"No usable stat columns in DF. Columns={list(sub.columns)}")
            cols = ["idx", num_cols[0]]

    out = sub[cols].copy()

    for c in out.columns:
        if c == "idx":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    stat_cols = [c for c in out.columns if c != "idx"]
    out = out.loc[~out[stat_cols].isna().all(axis=1)].copy()
    return out


# =========================
# SCENARIO FILE DISCOVERY
# =========================
def collect_rep_files_for_scenario(method_dir: Path, scenario: str, want_cols: List[str]) -> Dict[int, Path]:
    """
    Find per-rep score files for a scenario.
    Priority:
      1) method_dir/scenario/*.(csv|xlsx)
      2) any file under method_dir containing scenario token in filename
    Returns dict rep->file
    """
    method_dir = Path(method_dir)
    scen_dir = method_dir / scenario
    cand = []

    if scen_dir.exists():
        cand = [fp for fp in scen_dir.rglob("*") if fp.is_file() and fp.suffix.lower() in (".csv", ".xlsx", ".xls")]
    else:
        # fallback: name contains scenario
        scen_l = scenario.lower()
        for fp in _candidate_files(method_dir):
            if scen_l in fp.name.lower():
                cand.append(fp)

    # group by rep
    by_rep: Dict[int, List[Path]] = {}
    for fp in cand:
        rep = rep_from_name(fp.stem)
        if rep is None:
            continue
        by_rep.setdefault(rep, []).append(fp)

    out: Dict[int, Path] = {}
    for rep, fps in by_rep.items():
        pick = _pick_file(fps, want_cols, prefer_phaseII=True)
        if pick is not None:
            out[rep] = pick
    return out


def find_train_phaseI_file(method_dir: Path, method: str) -> Path:
    """
    Find pooled Phase-I training score file for OT/KDE.
    Expected like: train_OT_scores_phaseI.csv
    """
    method_dir = Path(method_dir)
    # direct expected
    expected = method_dir / f"train_{method}_scores_phaseI.csv"
    if expected.exists():
        return expected

    # scan for something that contains "train" and "phaseI"
    cand = []
    for fp in method_dir.glob("*.csv"):
        if PHASEI_TRAIN_RE.search(fp.name) and method.lower() in fp.name.lower():
            cand.append(fp)
    if cand:
        return sorted(cand)[0]

    # broader scan
    for fp in method_dir.rglob("*.csv"):
        if PHASEI_TRAIN_RE.search(fp.name) and method.lower() in fp.name.lower():
            return fp

    raise FileNotFoundError(f"Cannot find train Phase-I file under {method_dir} for method={method}")


def find_ic_rep_files_if_exist(method_dir: Path, want_cols: List[str]) -> Dict[int, Path]:
    """
    Try to find explicit IC scenario files for OT/KDE (optional).
    """
    method_dir = Path(method_dir)
    for cand in IC_SCENARIO_CANDIDATES:
        d = method_dir / cand
        if d.exists():
            files = collect_rep_files_for_scenario(method_dir, cand, want_cols)
            if files:
                return files

    # fallback: any file whose name contains "_IC_" (but not train)
    ic_files = []
    for fp in _candidate_files(method_dir):
        nl = fp.name.lower()
        if "ic" in nl and not PHASEI_TRAIN_RE.search(nl):
            ic_files.append(fp)

    by_rep: Dict[int, List[Path]] = {}
    for fp in ic_files:
        rep = rep_from_name(fp.stem)
        if rep is None:
            continue
        by_rep.setdefault(rep, []).append(fp)

    out: Dict[int, Path] = {}
    for rep, fps in by_rep.items():
        pick = _pick_file(fps, want_cols, prefer_phaseII=True)
        if pick is not None:
            out[rep] = pick
    return out


# =========================
# CONTROL LIMITS + RL
# =========================
def compute_ucls_from_pool(pool_by_col: Dict[str, np.ndarray], alpha: float) -> Dict[str, float]:
    ucls: Dict[str, float] = {}
    for c, arr in pool_by_col.items():
        x = np.asarray(arr, float)
        x = x[np.isfinite(x)]
        ucls[c] = float(np.quantile(x, 1.0 - float(alpha))) if x.size else float("inf")
    return ucls


def run_length_alarm(df_phase: pd.DataFrame, *, stat_cols: List[str], ucls: Dict[str, float], combine: str) -> Tuple[int, int]:
    H = int(len(df_phase))
    if H == 0:
        return 1, 0

    if combine == "single":
        c = stat_cols[0]
        if c not in df_phase.columns:
            c = "stat" if "stat" in df_phase.columns else df_phase.columns[1]
        x = df_phase[c].to_numpy(float)
        exc = np.isfinite(x) & (x > float(ucls.get(stat_cols[0], np.inf)))
    elif combine == "OR":
        exc = np.zeros(H, dtype=bool)
        for c in stat_cols:
            c_eff = c if c in df_phase.columns else ("stat" if "stat" in df_phase.columns else None)
            if c_eff is None:
                continue
            x = df_phase[c_eff].to_numpy(float)
            exc |= (np.isfinite(x) & (x > float(ucls.get(c, np.inf))))
    else:
        raise ValueError(f"Unknown combine rule: {combine}")

    idx = np.where(exc)[0]
    RL = int(idx[0] + 1) if idx.size else int(H + 1)
    return RL, H


def estimate_far_from_pool(df_pool: pd.DataFrame, *, stat_cols: List[str], ucls: Dict[str, float], combine: str) -> float:
    """
    Estimate per-time false alarm probability from pooled IC points (Phase I pool).
    This is used when we have no explicit IC Phase-II sequences.
    """
    if df_pool.empty:
        return float("nan")

    if combine == "single":
        c = stat_cols[0]
        c_eff = c if c in df_pool.columns else ("stat" if "stat" in df_pool.columns else None)
        if c_eff is None:
            return float("nan")
        x = pd.to_numeric(df_pool[c_eff], errors="coerce").to_numpy(float)
        p = float(np.nanmean(x > float(ucls.get(stat_cols[0], np.inf))))
        return p
    elif combine == "OR":
        exc = np.zeros(len(df_pool), dtype=bool)
        for c in stat_cols:
            c_eff = c if c in df_pool.columns else ("stat" if "stat" in df_pool.columns else None)
            if c_eff is None:
                continue
            x = pd.to_numeric(df_pool[c_eff], errors="coerce").to_numpy(float)
            exc |= np.isfinite(x) & (x > float(ucls.get(c, np.inf)))
        p = float(np.mean(exc)) if len(exc) else float("nan")
        return p
    else:
        return float("nan")


# =========================
# TRADEOFF CORE
# =========================
def compute_tradeoff_for_method(method: str, mdir: Path, scenario: str, alphas: List[float]) -> pd.DataFrame:
    spec = METHOD_SPECS[method]
    stat_cols = spec["stat_cols"]
    combine = spec["combine"]
    ic_source = spec["ic_source"]

    rows = []

    if ic_source == "embedded":
        # scenario files contain Phase I + Phase II
        rep_files = collect_rep_files_for_scenario(mdir, scenario, stat_cols)
        if not rep_files:
            raise FileNotFoundError(f"{method}: no rep score files found for scenario={scenario} under {mdir}")

        reps = sorted(rep_files.keys())

        # preload per-rep phase series + build pooled Phase-I points for UCLs
        pool_frames = []
        I_by_rep: Dict[int, pd.DataFrame] = {}
        II_by_rep: Dict[int, pd.DataFrame] = {}
        for rep in reps:
            df = read_table(rep_files[rep])
            dfI = phase_df(df, "I", stat_cols)
            dfII = phase_df(df, "II", stat_cols)
            I_by_rep[rep] = dfI
            II_by_rep[rep] = dfII
            pool_frames.append(dfI.drop(columns=["idx"], errors="ignore"))

        dfI_pool = pd.concat(pool_frames, axis=0, ignore_index=True) if pool_frames else pd.DataFrame()

        for alpha in alphas:
            alpha = float(alpha)
            pool_by_col = {}
            for c in stat_cols:
                c_eff = c if c in dfI_pool.columns else ("stat" if "stat" in dfI_pool.columns else None)
                if c_eff is None:
                    continue
                pool_by_col[c] = pd.to_numeric(dfI_pool[c_eff], errors="coerce").to_numpy(float)

            ucls = compute_ucls_from_pool(pool_by_col, alpha)

            # --- ARL0 from Phase-I sequences
            RL0 = []
            for rep in reps:
                rl0, _ = run_length_alarm(I_by_rep[rep], stat_cols=stat_cols, ucls=ucls, combine=combine)
                RL0.append(rl0)
            RL0 = np.asarray(RL0, float)
            ARL0_mean = float(np.mean(RL0))
            ARL0_std = float(np.std(RL0, ddof=1)) if RL0.size > 1 else 0.0
            FAR_mean = float(np.mean(1.0 / RL0))

            # --- ARL1 from Phase-II sequences
            RL1 = []
            det = []
            for rep in reps:
                rl1, H1 = run_length_alarm(II_by_rep[rep], stat_cols=stat_cols, ucls=ucls, combine=combine)
                RL1.append(rl1)
                det.append(1.0 if (H1 > 0 and rl1 <= H1) else 0.0)
            RL1 = np.asarray(RL1, float)
            det = np.asarray(det, float)

            ARL1_mean = float(np.mean(RL1))
            ARL1_std = float(np.std(RL1, ddof=1)) if RL1.size > 1 else 0.0
            det_rate = float(np.mean(det))
            RL1_det = RL1[det.astype(bool)]
            ARL1_cond = float(np.mean(RL1_det)) if RL1_det.size else float("nan")

            row = dict(
                method=method,
                scenario=scenario,
                alpha=alpha,
                n_rep=int(len(reps)),
                combine=combine,
                ARL0_mean=ARL0_mean,
                ARL0_std=ARL0_std,
                FAR_mean=FAR_mean,
                ARL1_mean=ARL1_mean,
                ARL1_std=ARL1_std,
                ARL1_cond_mean=ARL1_cond,
                detection_prob=det_rate,
                ARL0_source="phaseI_sequence",
            )
            for c in stat_cols:
                row[f"UCL_{c}"] = float(ucls.get(c, float("nan")))
            rows.append(row)

    elif ic_source == "train":
        # UCL from pooled Phase-I train file
        train_fp = find_train_phaseI_file(mdir, method)
        df_train = read_table(train_fp)
        dfI_pool = phase_df(df_train, "I", stat_cols)  # should be phase I
        dfI_pool_nidx = dfI_pool.drop(columns=["idx"], errors="ignore")

        # Phase-II OC per scenario/rep
        oc_files = collect_rep_files_for_scenario(mdir, scenario, stat_cols)
        if not oc_files:
            raise FileNotFoundError(f"{method}: no OC rep files found for scenario={scenario} under {mdir}")
        reps_oc = sorted(oc_files.keys())

        # Try IC Phase-II per rep (optional)
        ic_files = find_ic_rep_files_if_exist(mdir, stat_cols)
        reps_ic = sorted(set(ic_files.keys())) if ic_files else []

        II_oc_by_rep: Dict[int, pd.DataFrame] = {}
        for rep in reps_oc:
            df = read_table(oc_files[rep])
            dfII = phase_df(df, "II", stat_cols)
            II_oc_by_rep[rep] = dfII

        II_ic_by_rep: Dict[int, pd.DataFrame] = {}
        if reps_ic:
            for rep in reps_ic:
                df = read_table(ic_files[rep])
                dfII = phase_df(df, "II", stat_cols)
                II_ic_by_rep[rep] = dfII

        for alpha in alphas:
            alpha = float(alpha)

            pool_by_col = {}
            for c in stat_cols:
                c_eff = c if c in dfI_pool_nidx.columns else ("stat" if "stat" in dfI_pool_nidx.columns else None)
                if c_eff is None:
                    continue
                pool_by_col[c] = pd.to_numeric(dfI_pool_nidx[c_eff], errors="coerce").to_numpy(float)

            ucls = compute_ucls_from_pool(pool_by_col, alpha)

            # --- ARL0: prefer explicit IC Phase-II sequences; else geometric estimate from Phase-I pool
            if reps_ic:
                RL0 = []
                for rep in reps_ic:
                    rl0, _ = run_length_alarm(II_ic_by_rep[rep], stat_cols=stat_cols, ucls=ucls, combine=combine)
                    RL0.append(rl0)
                RL0 = np.asarray(RL0, float)
                ARL0_mean = float(np.mean(RL0))
                ARL0_std = float(np.std(RL0, ddof=1)) if RL0.size > 1 else 0.0
                FAR_mean = float(np.mean(1.0 / RL0))
                arl0_source = "IC_phaseII_sequence"
            else:
                # estimate per-time false alarm probability from Phase-I pool
                p_hat = estimate_far_from_pool(dfI_pool_nidx, stat_cols=stat_cols, ucls=ucls, combine=combine)
                if not np.isfinite(p_hat) or p_hat <= 0:
                    ARL0_mean = float("inf")
                    FAR_mean = float("nan")
                else:
                    ARL0_mean = float(1.0 / p_hat)
                    FAR_mean = float(p_hat)
                ARL0_std = 0.0
                arl0_source = "geom_from_phaseI_pool"

            # --- ARL1 from OC Phase-II sequences
            RL1 = []
            det = []
            for rep in reps_oc:
                rl1, H1 = run_length_alarm(II_oc_by_rep[rep], stat_cols=stat_cols, ucls=ucls, combine=combine)
                RL1.append(rl1)
                det.append(1.0 if (H1 > 0 and rl1 <= H1) else 0.0)
            RL1 = np.asarray(RL1, float)
            det = np.asarray(det, float)

            ARL1_mean = float(np.mean(RL1))
            ARL1_std = float(np.std(RL1, ddof=1)) if RL1.size > 1 else 0.0
            det_rate = float(np.mean(det))
            RL1_det = RL1[det.astype(bool)]
            ARL1_cond = float(np.mean(RL1_det)) if RL1_det.size else float("nan")

            row = dict(
                method=method,
                scenario=scenario,
                alpha=alpha,
                n_rep=int(len(reps_oc)),
                combine=combine,
                ARL0_mean=ARL0_mean,
                ARL0_std=ARL0_std,
                FAR_mean=FAR_mean,
                ARL1_mean=ARL1_mean,
                ARL1_std=ARL1_std,
                ARL1_cond_mean=ARL1_cond,
                detection_prob=det_rate,
                ARL0_source=arl0_source,
                train_phaseI_file=str(train_fp),
            )
            for c in stat_cols:
                row[f"UCL_{c}"] = float(ucls.get(c, float("nan")))
            rows.append(row)
    else:
        raise ValueError(f"Unknown ic_source for method={method}: {ic_source}")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["method", "alpha"]).reset_index(drop=True)
    return out


def compute_tradeoff(method_dirs: Dict[str, Path], *, scenario: str, alphas: List[float]) -> pd.DataFrame:
    all_rows = []
    for method, mdir in method_dirs.items():
        df_m = compute_tradeoff_for_method(method, mdir, scenario, alphas)
        all_rows.append(df_m)
    out = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    return out


# =========================
# PLOTTING
# =========================
# PUB = dict(base_font=11,
#            label_font=12,
#            tick_font=10,
#            legend_font=10,
#            lw=2.2,
#            ms=6.0,
#            mew=1.1,
#            grid_lw=0.7,
#            spine_lw=0.9)

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


def plot_tradeoff_twofigs(df: pd.DataFrame, *, out_base: Path, arl0_max: float = 120.0, dpi: int = 320):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print(f"[WARN] empty df; skip {out_base}")
        return

    df["method_display"] = df["method"].map(lambda m: METHOD_DISPLAY.get(m, m))
    # then use method_display for legends/labels


    methods = list(df["method_display"].dropna().unique())
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", ">", "<", "h", "*"])
    marker_map = {m: next(marker_cycle) for m in methods}
    # ---- style overrides for "Ours" (OT)
    marker_map["OT"] = "D"  # diamond marker
    color_map = {"OT": "red"}  # red line
    label_map = {"OT": "Ours"}  # legend label

    def _markevery(npts: int) -> int:
        return max(1, int(np.ceil(npts / 7)))

    def _style_axes(ax):
        ax.grid(True, alpha=0.18, linewidth=PUB["grid_lw"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4.0, width=0.9, pad=2.5)

    # Fig 1: detection vs FAR
    fig, ax = plt.subplots(1, 1, figsize=(3.55, 3.05), dpi=dpi, constrained_layout=True)
    _style_axes(ax)

    for method, g in df.groupby("method_display", sort=False):
        label = "Ours" if method == "OT" else method
        g = g.sort_values("FAR_mean")
        x = g["FAR_mean"].to_numpy(float)
        y = g["detection_prob"].to_numpy(float)
        keep = np.r_[True, np.diff(x) != 0]
        x, y = x[keep], y[keep]
        marker_map["OT"] = "D"  # diamond marker
        color_map = {"Ours": "red"}  # red line
        label_map = {"OT": "Ours"}  # legend label
        ax.plot(
            x, y,
            linewidth=PUB["lw"],
            marker=marker_map.get(method, "o"),
            markersize=PUB["ms"],
            markevery=_markevery(len(x)),
            markerfacecolor="white",
            markeredgewidth=PUB["mew"],
            solid_capstyle="round",
            solid_joinstyle="round",
            color=color_map.get(method, None),
            label=label_map.get(method, method),
            # markeredgecolor=color_map.get(method, "black"),
        )

    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("Detection probability")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, loc="lower right", handlelength=2.6, markerscale=1.1)

    out_det = out_base.with_name(out_base.stem + "_det_vs_far.png")
    fig.savefig(out_det, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: ARL1 vs ARL0
    fig, ax = plt.subplots(1, 1, figsize=(3.55, 3.05), dpi=dpi, constrained_layout=True)
    _style_axes(ax)

    dff = df.copy()
    dff = dff[np.isfinite(dff["ARL0_mean"]) & (dff["ARL0_mean"] <= arl0_max)].copy()

    for method, g in dff.groupby("method_display", sort=False):
        label = "Ours" if method == "OT" else method
        g = g.sort_values("ARL0_mean")
        x = g["ARL0_mean"].to_numpy(float)
        y = g["ARL1_mean"].to_numpy(float)
        keep = np.r_[True, np.diff(x) != 0]
        x, y = x[keep], y[keep]
        marker_map["OT"] = "D"  # diamond marker
        color_map = {"Ours": "red"}  # red line
        label_map = {"OT": "Ours"}  # legend label
        ax.plot(
            x, y,
            linewidth=PUB["lw"],
            marker=marker_map.get(method, "o"),
            markersize=PUB["ms"],
            markevery=_markevery(len(x)),
            markerfacecolor="white",
            markeredgewidth=PUB["mew"],
            solid_capstyle="round",
            solid_joinstyle="round",
            color=color_map.get(method, None),
            label=label_map.get(method, method),
            # markeredgecolor=color_map.get(method, "black"),
        )

    ax.set_xlabel("ARL0")
    ax.set_ylabel("ARL1")
    ax.set_xlim(0.0, arl0_max)
    ax.legend(frameon=False, loc="upper left", handlelength=2.6, markerscale=1.1)

    out_del = out_base.with_name(out_base.stem + "_arl1_vs_arl0.png")
    fig.savefig(out_del, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved:\n  {out_det}\n  {out_del}")


# =========================
# MAIN
# =========================
def build_method_dirs(mode_dir: Path) -> Dict[str, Path]:
    method_dirs: Dict[str, Path] = {}
    for m, sub in METHOD_SUBDIR.items():
        p = mode_dir / sub
        if p.exists():
            method_dirs[m] = p
    return method_dirs


def main():
    for n in N_LIST:
        base = DATA_ROOT / f"n{n}" / f"results_discrete_all_methods_n{n}"
        if not base.exists():
            print(f"[WARN] missing: {base}")
            continue

        for cfg_tag, scenarios in TARGETS.items():
            mode_dir = base / cfg_tag
            if not mode_dir.exists():
                print(f"[WARN] missing mode dir: {mode_dir}")
                continue

            method_dirs = build_method_dirs(mode_dir)
            if not method_dirs:
                print(f"[WARN] no method dirs under: {mode_dir}")
                continue

            for scenario in scenarios:
                out_dir = OUT_ROOT / f"n{n}" / cfg_tag / scenario
                out_dir.mkdir(parents=True, exist_ok=True)

                try:
                    df_t = compute_tradeoff(method_dirs, scenario=scenario, alphas=ALPHAS)

                    df_t.to_csv(out_dir / f"tradeoff_n{n}_{cfg_tag}_{scenario}.csv", index=False)
                    df_t.to_excel(out_dir / f"tradeoff_n{n}_{cfg_tag}_{scenario}.xlsx", index=False)

                    plot_tradeoff_twofigs(
                        df_t,
                        out_base=out_dir / f"tradeoff_n{n}_{cfg_tag}_{scenario}",
                        arl0_max=120.0,
                    )

                    print(f"[OK] n={n} {cfg_tag} {scenario} -> {out_dir}")

                except Exception as e:
                    print(f"[ERROR] n={n} {cfg_tag} {scenario}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
