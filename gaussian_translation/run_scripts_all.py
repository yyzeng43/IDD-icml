# Gaussian translation comparison: IDD (OT-MFPCA) vs. log-KDE-MFPCA.
# Reproduces Table 1 / Theorem F.1 experiments from the paper.
import subprocess
import time, math, json, csv, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

r_script_proposed = Path(__file__).parent / "ot_mfpca_once.R"


# ======================
# Utilities
# ======================
def save_tangents_for_R(Xb, wb, tangI, tangII, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    np.save(outbase.with_suffix(".Xb.npy"), Xb)
    np.save(outbase.with_suffix(".wb.npy"), wb)
    # Save Phase I tangents as a 3D array [n0, n_bary*d]
    # n0 = len(tangI)
    n_bary, d = tangI[0].shape
    TI = np.stack([t.reshape(-1) for t in tangI], axis=0)
    np.save(outbase.with_suffix(".PhaseI.npy"), TI)
    # Save Phase II as [nT, n_bary*d]
    TII = np.stack([t.reshape(-1) for t in tangII], axis=0)
    np.save(outbase.with_suffix(".PhaseII.npy"), TII)

    # Metadata
    with open(outbase.with_suffix(".meta.txt"), "w") as f:
        f.write(f"n_bary={n_bary}, d={d}, p={n_bary*d}\n")


def run_proposed_once(
    base_prefix: Path,
    K=None,
    alpha: float = 0.05,
    limits: str = "standard",
    out_dir: Path | str | None = None,
    export_stem: str | None = None,
    scenario: str | None = None,
):
    """
    Call the R pipeline ONCE (no replicate logic).

    Expects tangent files (saved by save_tangents_for_R) with a common base:
        base_prefix.Xb.npy
        base_prefix.PhaseI.npy
        base_prefix.PhaseII.npy

    Parameters
    ----------
    base_prefix : Path
        Common prefix for the .npy files used by the R script.
    K : int or None
        Number of PCs to keep in mFPCA (None => let R choose via tot_var).
    alpha : float
        Overall alpha level for control limits.
    limits : {"standard", "cv"}
        Passed to funcharts::control_charts_pca.
    out_dir : Path or str or None
        Root output directory. If None, use base_prefix.parent.
    export_stem : str or None
        Filename stem (without suffix). If None, use base_prefix.name.
    scenario : str or None
        Optional tag (e.g., "IC", "OC"). If provided, results are written to
        out_dir / scenario and the stem is suffixed with "_<scenario>".

    Returns
    -------
    csv_path : Path
        Path to "<stem>_cc_all.csv".
    summary : dict
        Parsed JSON summary from "<stem>_summary.json".
    t_mfpca : float or None
        mFPCA fit time (seconds) from "<stem>_timing.json", if available.
    plot_csv_path : Path
        Path to "<stem>_plot_series.csv".
    """
    base_prefix = Path(base_prefix)

    # Decide output folder
    if out_dir is None:
        out_dir = base_prefix.parent
    out_dir = Path(out_dir)

    # If scenario tag is given, put results in a subfolder
    if scenario is not None:
        out_dir = out_dir / str(scenario)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide stem
    if export_stem is None:
        export_stem = base_prefix.name
    if scenario is not None:
        export_stem = f"{export_stem}_{scenario}"

    # Build R command
    cmd = [
        "Rscript",
        str(r_script_proposed),
        str(base_prefix),                   # arg1: base (prefix for .Xb/.PhaseI/.PhaseII)
        "" if K is None else str(int(K)),   # arg2: K or empty
        str(float(alpha)),                  # arg3: alpha
        str(limits),                        # arg4: limits
        str(out_dir),                       # arg5: out_dir
        export_stem,                        # arg6: export_stem
    ]

    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Rscript failed.")
        print("Command:", " ".join(cmd))
        print("R stdout:\n", e.stdout)
        print("R stderr:\n", e.stderr)
        raise

    # Collect outputs
    csv_path      = out_dir / f"{export_stem}_cc_all.csv"
    json_path     = out_dir / f"{export_stem}_summary.json"
    tim_path      = out_dir / f"{export_stem}_timing.json"
    plot_csv_path = out_dir / f"{export_stem}_plot_series.csv"

    summary = json.loads(json_path.read_text()) if json_path.exists() else {}
    t_mfpca = None
    if tim_path.exists():
        j = json.loads(tim_path.read_text())
        t_mfpca = float(j.get("mfpca_fit_sec", 0.0))

    return csv_path, summary, t_mfpca, plot_csv_path
# ======================
# MAIN DRIVER: OT vs log-KDE
# ======================




# ---------- small utilities ----------
def mean_se(x):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), 0.0
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(n)
    return float(m), float(se)

def parse_cfg_name(cfg_name: str):
    """
    Parse names like:
        'gauss_shift_d2_sig1p00_del0p50'
    into (d, sigma, delta1).
    """
    parts = cfg_name.split("_")

    # find the part that starts with 'd' followed by digits, e.g. 'd2', 'd10'
    d_part = None
    sig_part = None
    del_part = None

    for p in parts:
        if p.startswith("d") and p[1:].isdigit():
            d_part = p
        elif p.startswith("sig"):
            sig_part = p
        elif p.startswith("del"):
            del_part = p

    if d_part is None or sig_part is None or del_part is None:
        raise ValueError(f"Cannot parse cfg_name='{cfg_name}' into d, sigma, delta1")

    d = int(d_part[1:])  # drop leading 'd'

    # replace 'p' back to '.', e.g. '1p00' -> '1.00'
    sigma = float(sig_part.replace("sig", "").replace("p", "."))
    delta1 = float(del_part.replace("del", "").replace("p", "."))

    return d, sigma, delta1


def first_alarm(stat, UCL):
    """Run length: first index (1-based) where stat > UCL, or H+1 if none."""
    stat = np.asarray(stat, float)
    H = len(stat)
    idx = np.argmax(stat > UCL)
    if (stat > UCL).any():
        return int(idx + 1)
    else:
        return int(H + 1)

# ---------- OT: extract RL and #alarms from R plot_series CSV ----------
def extract_ot_metrics_from_plot_csv(plot_csv_path: Path):
    from log_kde import _read_plot_series_csv  # you already have this
    data = _read_plot_series_csv(plot_csv_path)
    idx = data["idx"]
    T2 = data["T2"]
    SPE = data["SPE"]
    nI = data["nI"]
    UCL_T2 = data["UCL_T2"]
    UCL_SPE = data["UCL_SPE"]

    # Phase II part only
    T2_II = T2[nI:]
    SPE_II = SPE[nI:]

    RL_T2 = first_alarm(T2_II, UCL_T2)
    RL_SPE = first_alarm(SPE_II, UCL_SPE)

    n_alarm_T2 = int((T2_II > UCL_T2).sum())
    n_alarm_SPE = int((SPE_II > UCL_SPE).sum())

    return dict(
        UCL_T2=UCL_T2,
        UCL_SPE=UCL_SPE,
        RL_T2=RL_T2,
        RL_SPE=RL_SPE,
        n_alarm_T2=n_alarm_T2,
        n_alarm_SPE=n_alarm_SPE,
        H=len(T2_II),
    )

# ======================================
# 1) LOOP over configs & replications
# ======================================
if __name__ == "__main__":

    from ot_mfpca import process_phaseI, process_phaseII
    from log_kde import (
        logkde_mfpca_full_fit_phaseI,
        logkde_mfpca_full_score_phaseII,
        plot_kde_control_charts
    )

    # ---------- basic paths ----------
    DATA_ROOT = Path(os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent / "data"))
    datadir = DATA_ROOT / "gaussian_translation"
    ot_export_root = datadir / "ot_exports"
    ot_export_root.mkdir(parents=True, exist_ok=True)

    resdir = datadir / "results"
    resdir.mkdir(parents=True, exist_ok=True)

    # ---------- OT settings ----------
    reg = 0.05
    METHOD = "emd"  # switch to "sinkhorn" for high-dimensional settings

    # ---------- KDE bandwidths ----------
    # None = automatic (your logkde code), plus some fixed h
    bandwidth_grid = [None, 0.5, 1.0, 1.5]

    rows = []  # each row = one (method, cfg, rep, bandwidth, scenario, stat) observation

    # find all config folders created earlier, e.g. gauss_shift_d2_sig1p00_del0p50
    cfg_dirs = sorted([p for p in datadir.iterdir()
                       if p.is_dir() and p.name.startswith("gauss_shift_d")])

    for cfg_dir in cfg_dirs:
        cfg_name = cfg_dir.name
        d, sigma, delta1 = parse_cfg_name(cfg_name)
        print(f"\n=== CONFIG {cfg_name} (d={d}, sigma={sigma}, delta1={delta1}) ===")

        if d < 5:
            continue
        # all rep phase-I files for this config
        # adjust pattern here if your filenames differ, e.g. "*_rep??.PhaseI.npz"
        phaseI_files = sorted(cfg_dir.glob("*_rep??_phaseI.npz"))
        if not phaseI_files:
            print(f"  [WARN] no *_rep??_phaseI.npz in {cfg_dir}, skip.")
            continue

        for phaseI_file in phaseI_files:
            # derive rep ID and the two Phase-II files
            stem = phaseI_file.stem  # e.g. gauss_shift_d2_sig1p00_del0p50_rep00_phaseI
            rep_tag = stem.split("_rep")[-1].split("_")[0]  # '00'
            rep_id = int(rep_tag)

            base_prefix = stem.rsplit("_phaseI", 1)[0]  # remove '_phaseI'
            phaseII_ic_file = cfg_dir / f"{base_prefix}_phaseII_ic.npz"
            phaseII_oc_file = cfg_dir / f"{base_prefix}_phaseII_oc.npz"

            if not (phaseII_ic_file.exists() and phaseII_oc_file.exists()):
                print(f"  [WARN] missing Phase-II files for {phaseI_file.name}, skip.")
                continue

            print(f"  - rep {rep_id:02d}: running OT & KDE")

            # -------------------------------------------------
            # 1A. Proposed OT–MFPCA
            # -------------------------------------------------
            t0 = time.perf_counter()
            Xb, wb, tangI, ThatI, paramsI, cloudsI, tan_norms_I = process_phaseI(
                phaseI_file, reg=reg, n_bary=None, method=METHOD
            )
            t_phaseI = time.perf_counter() - t0

            t1 = time.perf_counter()
            tangII_ic, ThatII_ic, paramsII_ic, streamsII_ic, tan_norms_ic, Txs_true_ic = process_phaseII(
                phaseII_ic_file, Xb, wb, reg=reg, method=METHOD
            )
            t_phaseII_ic = time.perf_counter() - t1

            t2 = time.perf_counter()
            tangII_oc, ThatII_oc, paramsII_oc, streamsII_oc, tan_norms_oc, Txs_true_oc = process_phaseII(
                phaseII_oc_file, Xb, wb, reg=reg, method=METHOD
            )
            t_phaseII_oc = time.perf_counter() - t2

            # save tangents for R (one base for IC, one for OC)
            cfg_ot_dir = ot_export_root / cfg_name
            cfg_ot_dir.mkdir(parents=True, exist_ok=True)

            base_ic = cfg_ot_dir / f"{base_prefix}_IC"
            base_oc = cfg_ot_dir / f"{base_prefix}_OC"
            save_tangents_for_R(Xb, wb, tangI, ThatII_ic, base_ic)
            save_tangents_for_R(Xb, wb, tangI, ThatII_oc, base_oc)

            print(base_ic, base_prefix)

            # run R once per scenario
            csv_ic, summary_ic, t_mfpca_ic, plot_csv_ic = run_proposed_once(
                base_prefix=base_ic,
                K=None,
                alpha=0.05,
                limits="standard",
                out_dir=resdir / cfg_name,
                export_stem=f"{base_prefix}_IC",
                scenario=None,
            )

            csv_oc, summary_oc, t_mfpca_oc, plot_csv_oc = run_proposed_once(
                base_prefix=base_oc,
                K=None,
                alpha=0.05,
                limits="standard",
                out_dir=resdir / cfg_name,
                export_stem=f"{base_prefix}_OC",
                scenario=None,
            )

            # extract RL, #alarms from plot CSVs
            met_ic = extract_ot_metrics_from_plot_csv(plot_csv_ic)
            met_oc = extract_ot_metrics_from_plot_csv(plot_csv_oc)

            runtime_ic_ot = t_phaseI + t_phaseII_ic + (t_mfpca_ic or 0.0)
            runtime_oc_ot = t_phaseI + t_phaseII_oc + (t_mfpca_oc or 0.0)

            # store two rows per stat (T2, SPE) × scenario (IC/OC)
            for scenario, met, rt in [("IC", met_ic, runtime_ic_ot),
                                      ("OC", met_oc, runtime_oc_ot)]:
                rows.append(dict(
                    method="OT",
                    bandwidth=np.nan,
                    cfg=cfg_name,
                    d=d, sigma=sigma, delta1=delta1,
                    rep=rep_id,
                    scenario=scenario,
                    stat="T2",
                    runtime=rt,
                    ARL=met["RL_T2"],
                    n_alarm=met["n_alarm_T2"],
                    H=met["H"],
                ))
                rows.append(dict(
                    method="OT",
                    bandwidth=np.nan,
                    cfg=cfg_name,
                    d=d, sigma=sigma, delta1=delta1,
                    rep=rep_id,
                    scenario=scenario,
                    stat="SPE",
                    runtime=rt,
                    ARL=met["RL_SPE"],
                    n_alarm=met["n_alarm_SPE"],
                    H=met["H"],
                ))

            # -------------------------------------------------
            # 1B. log-KDE–MFPCA for multiple bandwidths
            # -------------------------------------------------
            # TODO: SHOULD CHANGE! Use OT barycenter points as evaluation grid
            X_eval_grid = Xb.copy()

            for h in bandwidth_grid:
                t0 = time.perf_counter()
                logkde_model, phaseI_stats = logkde_mfpca_full_fit_phaseI(
                    phaseI=cloudsI,
                    X_eval=X_eval_grid,
                    bandwidth=h,        # <== THIS is where h enters
                    var_explained=0.95,
                    alpha=0.05,
                    eps_log=1e-8,
                )

                T2_I = np.asarray(phaseI_stats["T2"], float)
                SPE_I = np.asarray(phaseI_stats["SPE"], float)

                t_fit = time.perf_counter() - t0

                # IC
                t1 = time.perf_counter()
                T2_ic, SPE_ic, UCL_T2_ic, UCL_SPE_ic, H_ic, RL_T2_ic, RL_SPE_ic = \
                    logkde_mfpca_full_score_phaseII(logkde_model, streamsII_ic)
                t_score_ic = time.perf_counter() - t1

                # OC
                t2 = time.perf_counter()
                T2_oc, SPE_oc, UCL_T2_oc, UCL_SPE_oc, H_oc, RL_T2_oc, RL_SPE_oc = \
                    logkde_mfpca_full_score_phaseII(logkde_model, streamsII_oc)
                t_score_oc = time.perf_counter() - t2

                runtime_ic_kde = t_fit + t_score_ic
                runtime_oc_kde = t_fit + t_score_oc

                # ========= NEW: PLOTTING & SAVING =========
                kde_fig_dir = resdir / cfg_name / "logkde"
                # IC chart
                out_png_ic = kde_fig_dir / f"logkde_h{h}_IC_rep{rep_id:02d}_control_charts.png"
                plot_kde_control_charts(
                    T2_I=T2_I,
                    SPE_I=SPE_I,
                    T2_II=T2_ic,
                    SPE_II=SPE_ic,
                    UCL_T2=UCL_T2_ic,
                    UCL_SPE=UCL_SPE_ic,
                    out_png=out_png_ic,
                    scenario="IC",
                    bandwidth_label=str(h),
                )
                # OC chart
                out_png_oc = kde_fig_dir / f"logkde_h{h}_OC_rep{rep_id:02d}_control_charts.png"
                plot_kde_control_charts(
                    T2_I=T2_I,
                    SPE_I=SPE_I,
                    T2_II=T2_oc,
                    SPE_II=SPE_oc,
                    UCL_T2=UCL_T2_oc,
                    UCL_SPE=UCL_SPE_oc,
                    out_png=out_png_oc,
                    scenario="OC",
                    bandwidth_label=str(h),
                )
                # ========= END NEW PLOTTING =========

                rows.append(dict(
                    method="logKDE",
                    bandwidth=(h if h is not None else -1.0),  # encode None as -1 for CSV
                    cfg=cfg_name,
                    d=d, sigma=sigma, delta1=delta1,
                    rep=rep_id,
                    scenario="IC",
                    stat="T2",
                    runtime=runtime_ic_kde,
                    ARL=RL_T2_ic,
                    n_alarm=int((T2_ic > UCL_T2_ic).sum()),
                    H=H_ic,
                ))
                rows.append(dict(
                    method="logKDE",
                    bandwidth=(h if h is not None else -1.0),
                    cfg=cfg_name,
                    d=d, sigma=sigma, delta1=delta1,
                    rep=rep_id,
                    scenario="IC",
                    stat="SPE",
                    runtime=runtime_ic_kde,
                    ARL=RL_SPE_ic,
                    n_alarm=int((SPE_ic > UCL_SPE_ic).sum()),
                    H=H_ic,
                ))
                rows.append(dict(
                    method="logKDE",
                    bandwidth=(h if h is not None else -1.0),
                    cfg=cfg_name,
                    d=d, sigma=sigma, delta1=delta1,
                    rep=rep_id,
                    scenario="OC",
                    stat="T2",
                    runtime=runtime_oc_kde,
                    ARL=RL_T2_oc,
                    n_alarm=int((T2_oc > UCL_T2_oc).sum()),
                    H=H_oc,
                ))
                rows.append(dict(
                    method="logKDE",
                    bandwidth=(h if h is not None else -1.0),
                    cfg=cfg_name,
                    d=d, sigma=sigma, delta1=delta1,
                    rep=rep_id,
                    scenario="OC",
                    stat="SPE",
                    runtime=runtime_oc_kde,
                    ARL=RL_SPE_oc,
                    n_alarm=int((SPE_oc > UCL_SPE_oc).sum()),
                    H=H_oc,
                ))

    # =====================
    # 2) Aggregate across replications: mean + SE
    # =====================

    df = pd.DataFrame(rows)

    # true bandwidth: replace -1 with NaN or label "auto"
    df["bandwidth_label"] = df["bandwidth"].apply(
        lambda x: "auto" if (isinstance(x, float) and x < 0) else x
    )

    group_cols = ["method", "bandwidth_label", "cfg", "d", "sigma", "delta1", "scenario", "stat"]

    summary = (
        df
        .groupby(group_cols, dropna=False)
        .agg(
            runtime_mean=("runtime", "mean"),
            runtime_se=("runtime", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0),
            ARL_mean=("ARL", "mean"),
            ARL_se=("ARL", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0),
            n_alarm_mean=("n_alarm", "mean"),
            n_alarm_se=("n_alarm", lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0.0),
            H_mean=("H", "mean"),
        )
        .reset_index()
    )

    summary_csv = resdir / "summary_OT_vs_logKDE_gauss_shift_d5sh.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\n[OK] Summary saved to {summary_csv}")
