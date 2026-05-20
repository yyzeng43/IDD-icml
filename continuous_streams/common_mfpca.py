# used to facillicate the calling of all the mfpca in R


# common_mfpca.py
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_SCRIPT_MFPCA = Path(__file__).parent / "run_mfpca.R"

# ---------- R call wrapper ----------

def call_r_mfpca_scores(base_prefix: Path,
                        out_dir: Path,
                        export_stem: str,
                        K: int | None = None):
    """
    Call R run_mfpca_scores.R on base_prefix.{Xb,PhaseI,PhaseII}.npy.

    Returns
    -------
    df_scores : DataFrame with columns [idx, phase, T2, SPE]
    t_mfpca   : float or None, elapsed time in seconds
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "Rscript",
        str(R_SCRIPT_MFPCA),
        str(base_prefix),
        str(out_dir),
        export_stem,
    ]
    if K is not None:
        cmd.append(str(int(K)))

    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Rscript failed.")
        print("Command:", " ".join(cmd))
        print("R stdout:\n", e.stdout)
        print("R stderr:\n", e.stderr)
        raise

    scores_path = out_dir / f"{export_stem}_scores.csv"
    timing_path = out_dir / f"{export_stem}_timing.json"

    df_scores = pd.read_csv(scores_path)

    t_mfpca = None
    if timing_path.exists():
        with open(timing_path, "r") as f:
            try:
                t_mfpca = float(json.load(f).get("mfpca_fit_sec", 0.0))
            except Exception:
                t_mfpca = None

    return df_scores, t_mfpca


# ---------- empirical ARL utilities ----------

def compute_RL_and_alarms(values, UCL: float):
    """
    Given a 1D array of statistics and a fixed UCL,
    return (run_length, n_alarms).

    RL = index of first point > UCL (1-based).
    If no alarm, RL = H + 1.
    """
    v = np.asarray(values, float)
    H = len(v)
    above = v > UCL
    idx = np.flatnonzero(above)
    RL = int(idx[0]) + 1 if idx.size > 0 else H + 1
    n_alarm = int(above.sum())
    return RL, n_alarm


# ---------- plotting helpers ----------

def _plot_series(ax, x, y, nI, UCL, LCL, ylabel: str, title: str):
    """
    Plot Phase-I (grey) and Phase-II (black), mark y > UCL in red.
    """
    if len(x) == 0:
        return

    x = np.asarray(x)
    y = np.asarray(y)

    xI, yI = x[:nI], y[:nI]
    xII, yII = x[nI:], y[nI:]

    # Phase I: grey
    ax.plot(xI, yI, "-", color="#bdbdbd", linewidth=1.2)
    ax.scatter(xI, yI, s=14, color="#bdbdbd", zorder=3)

    # Phase II: black
    ax.plot(xII, yII, "-", color="black", linewidth=1.2)
    ax.scatter(xII, yII, s=14, color="black", zorder=3)

    # Out-of-control points (any phase) in red
    if UCL is not None:
        mask_red = y > UCL
        if np.any(mask_red):
            ax.scatter(
                x[mask_red],
                y[mask_red],
                s=30,
                color="red",
                edgecolor="none",
                zorder=5,
            )

    # Control limits
    if UCL is not None:
        ax.axhline(UCL, color="black", linewidth=1.4)
    if LCL is not None:
        ax.axhline(LCL, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    # Phase-I / Phase-II separator
    ax.axvline(nI + 0.5, color="black", linestyle="--", linewidth=1.0)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x.min(), x.max())


def plot_T2_SPE_OC(df_scores: pd.DataFrame,
                   UCL_T2: float,
                   UCL_SPE: float,
                   out_prefix: Path,
                   method_label: str,
                   cfg_name: str,
                   rep_id: int,
                   LCL_T2: float | None = None,
                   LCL_SPE: float | None = None):
    """
    Make separate T2 / SPE OC plots (Phase I + II) using the style you requested.
    """
    out_prefix = Path(out_prefix)
    nI = int((df_scores["phase"] == "I").sum())
    x  = df_scores["idx"].to_numpy()
    T2 = df_scores["T2"].to_numpy()
    SPE = df_scores["SPE"].to_numpy()

    # T2
    fig, ax = plt.subplots(figsize=(8, 3))
    _plot_series(
        ax,
        x=x,
        y=T2,
        nI=nI,
        UCL=UCL_T2,
        LCL=LCL_T2,
        ylabel="T²",
        title=f"{method_label} T² – {cfg_name}, rep {rep_id:02d}",
    )
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + f"_T2_rep{rep_id:02d}.png"),
                dpi=200)
    plt.close(fig)

    # SPE
    fig, ax = plt.subplots(figsize=(8, 3))
    _plot_series(
        ax,
        x=x,
        y=SPE,
        nI=nI,
        UCL=UCL_SPE,
        LCL=LCL_SPE,
        ylabel="SPE",
        title=f"{method_label} SPE – {cfg_name}, rep {rep_id:02d}",
    )
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + f"_SPE_rep{rep_id:02d}.png"),
                dpi=200)
    plt.close(fig)

