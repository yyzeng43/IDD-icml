from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent / "idd_core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "continuous_streams"))


def _plot_series(ax, x, y, nI, UCL, LCL, ylabel: str, title: str):
    if len(x) == 0:
        return

    x = np.asarray(x)
    y = np.asarray(y)

    xI, yI = x[:nI], y[:nI]
    xII, yII = x[nI:], y[nI:]

    ax.plot(xI, yI, "-", color="#bdbdbd", linewidth=1.2)
    ax.scatter(xI, yI, s=14, color="#bdbdbd", zorder=3)
    ax.plot(xII, yII, "-", color="black", linewidth=1.2)
    ax.scatter(xII, yII, s=14, color="black", zorder=3)

    if UCL is not None:
        mask_red = y > UCL
        if np.any(mask_red):
            ax.scatter(x[mask_red], y[mask_red], s=30, color="red", edgecolor="none", zorder=5)

    if UCL is not None:
        ax.axhline(UCL, color="black", linewidth=1.4)
    if LCL is not None:
        ax.axhline(LCL, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.axvline(nI + 0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x.min(), x.max())


def save_scores_csv(*, out_dir, method, scen, rep, stat_name, sI, sII):
    out_dir = safe_mkdir(out_dir)
    sI = np.asarray(sI, float).reshape(-1)
    sII = np.asarray(sII, float).reshape(-1)
    df_scores = pd.DataFrame({
        "idx": np.arange(1, len(sI) + len(sII) + 1, dtype=int),
        "phase": (["I"] * len(sI)) + (["II"] * len(sII)),
        stat_name: np.concatenate([sI, sII], axis=0),
    })
    fp = out_dir / f"scores_{method}_{scen}_{rep}.csv"
    df_scores.to_csv(fp, index=False)
    return fp


def plot_single_stat_OC(statI: np.ndarray,
                        statII: np.ndarray,
                        UCL: float,
                        out_prefix: Path,
                        method_label: str,
                        cfg_name: str,
                        rep_id: int,
                        ylabel: str,
                        LCL: float | None = None):
    statI = np.asarray(statI, float)
    statII = np.asarray(statII, float)
    nI = statI.size
    y = np.concatenate([statI, statII], axis=0)
    x = np.arange(1, y.size + 1)

    fig, ax = plt.subplots(figsize=(8, 3))
    _plot_series(ax, x=x, y=y, nI=nI, UCL=UCL, LCL=LCL,
                 ylabel=ylabel, title=f"{method_label} {ylabel} – {cfg_name}, rep {rep_id:02d}")
    fig.tight_layout()
    safe_tag = str(ylabel).replace("|", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    safe_tag = safe_tag.replace("<", "_").replace(">", "_").replace('"', "_").replace("?", "_").replace("*", "_")
    out_file = Path(out_prefix).with_name(Path(out_prefix).name + f"_{safe_tag}_rep{rep_id:02d}.png")
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def load_clouds(npz_path: Path, key: str) -> list[np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    raw = list(z[key])
    out = []
    for r in raw:
        a = np.asarray(r)
        if a.ndim == 1:
            a = a[:, None]
        out.append(a)
    return out


def compute_RL_and_alarms_one_sided(stat: np.ndarray, UCL: float) -> tuple[int, int]:
    stat = np.asarray(stat, float)
    idx = np.where(stat > float(UCL))[0]
    if idx.size == 0:
        return int(len(stat)), 0
    return int(idx[0] + 1), int(idx.size)


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def c_chart_limits(cbar: float) -> tuple[float, float]:
    cbar = float(max(cbar, 0.0))
    ucl = cbar + 3.0 * np.sqrt(cbar)
    lcl = max(0.0, cbar - 3.0 * np.sqrt(cbar))
    return lcl, ucl


def max_abs_z_multinomial(counts: np.ndarray, p0: np.ndarray) -> float:
    counts = np.asarray(counts, float)
    p0 = np.asarray(p0, float)
    n = float(np.sum(counts))
    if n <= 0:
        return 0.0
    phat = counts / n
    denom = np.sqrt(np.maximum(p0 * (1 - p0) / n, 1e-12))
    z = (phat - p0) / denom
    return float(np.max(np.abs(z)))


def sidak_ucl_for_max_abs_z(alpha: float, m: int) -> float:
    m = int(m)
    p = (1.0 + (1.0 - float(alpha))**(1.0 / m)) / 2.0
    return float(norm.ppf(p))


def run_one_config_all_methods(
    datadir: Path,
    resdir: Path,
    *,
    cfg_tag: str,
    mode: str,
    rep_ids: list[int],
    alpha0: float,
    m: int = 6,
) -> None:

    mode = str(mode).lower()
    out_cfg = safe_mkdir(Path(resdir) / cfg_tag)

    if mode == "count_1d":
        phaseI_files = [Path(datadir) / f"phaseI_count_d1_rep{r:02d}.npz" for r in rep_ids]
        scen_files = {
            "C1_spike":    [Path(datadir) / f"phaseII_count_spike_d1_rep{r:02d}.npz" for r in rep_ids],
            "C2_zeroinfl": [Path(datadir) / f"phaseII_count_zeroinfl_d1_rep{r:02d}.npz" for r in rep_ids],
        }
    elif mode == "categorical":
        phaseI_files = [Path(datadir) / f"phaseI_cat_m{m}_rep{r:02d}.npz" for r in rep_ids]
        scen_files = {
            "M1_transfer": [Path(datadir) / f"phaseII_cat_transfer_m{m}_rep{r:02d}.npz" for r in rep_ids],
            "M2_rare":     [Path(datadir) / f"phaseII_cat_rare_m{m}_rep{r:02d}.npz" for r in rep_ids],
            "M3_drift":    [Path(datadir) / f"phaseII_cat_drift_m{m}_rep{r:02d}.npz" for r in rep_ids],
            "M4_blur":     [Path(datadir) / f"phaseII_cat_blur_m{m}_rep{r:02d}.npz" for r in rep_ids],
        }
    elif mode == "dust_poisson_1d":
        phaseI_files = [Path(datadir) / f"phaseI_dust_poisson_d1_rep{r:02d}.npz" for r in rep_ids]
        scen_files = {
            "P1_mix":   [Path(datadir) / f"phaseII_dust_mix_d1_rep{r:02d}.npz" for r in rep_ids],
            "P2_spike": [Path(datadir) / f"phaseII_dust_spike_d1_rep{r:02d}.npz" for r in rep_ids],
        }
    else:
        raise ValueError("mode must be 'count_1d', 'categorical', or 'dust_poisson_1d'")

    for f in phaseI_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f}")
    for scen, lst in scen_files.items():
        for f in lst:
            if not f.exists():
                raise FileNotFoundError(f"Missing: {f}")

    rows = []

    # ==========================================================
    # ATTRIBUTE CHART baseline
    #   - count_1d / dust_poisson_1d: c-chart on total counts
    #   - categorical: max-|z| chart over category proportions
    # ==========================================================
    method = "ATTR_CHART"
    out_m = safe_mkdir(out_cfg / method)

    statI_by_rep = {}
    all_statI = []

    if mode in ("count_1d", "dust_poisson_1d"):
        for rep, f in zip(rep_ids, phaseI_files):
            cloudsI = load_clouds(f, "phaseI")
            s = np.array([float(np.sum(c[:, 0])) for c in cloudsI], float)
            statI_by_rep[rep] = s
            all_statI.append(s)

        all_statI = np.concatenate(all_statI, axis=0)
        cbar = float(np.mean(all_statI))
        LCL, UCL = c_chart_limits(cbar)
        lim = {"CL": cbar, "LCL": float(LCL), "UCL": float(UCL), "alpha": float(alpha0), "chart": "c"}

        with open(out_m / "limits_C.json", "w", encoding="utf-8") as f:
            json.dump(lim, f, indent=2)

        for scen, files in scen_files.items():
            out_s = safe_mkdir(out_m / scen)
            for rep, f in zip(rep_ids, files):
                cloudsII = load_clouds(f, "streams")
                sII = np.array([float(np.sum(c[:, 0])) for c in cloudsII], float)
                sI = statI_by_rep[rep]

                save_scores_csv(
                    out_dir=out_s, method=method, scen=scen, rep=rep,
                    stat_name="c", sI=sI, sII=sII,
                )

                RL, nA = compute_RL_and_alarms_one_sided(sII, lim["UCL"])
                rows.append({
                    "cfg_tag": cfg_tag, "mode": mode, "method": method,
                    "scenario": scen, "rep": rep, "stat": "c",
                    "ARL": RL, "n_alarm": nA, "H": int(len(sII)),
                })

                plot_single_stat_OC(
                    statI=sI, statII=sII,
                    UCL=lim["UCL"], LCL=lim["LCL"],
                    out_prefix=out_s / f"chart_{method}_{scen}",
                    method_label=method, cfg_name=f"{cfg_tag}/{scen}",
                    rep_id=rep, ylabel="C_t",
                )

    else:
        # categorical: max |z_j| across categories
        counts_total = np.zeros(m, float)
        n_total = 0.0
        cloudsI_all_by_rep = {}

        for rep, f in zip(rep_ids, phaseI_files):
            cloudsI = load_clouds(f, "phaseI")
            cloudsI_all_by_rep[rep] = cloudsI
            for c in cloudsI:
                x = c[:, 0].astype(int)
                h = np.bincount(x, minlength=m).astype(float)
                counts_total += h
                n_total += float(h.sum())

        p0 = counts_total / max(n_total, 1.0)
        p0 = np.maximum(p0, 1e-12)
        p0 = p0 / p0.sum()

        for rep in rep_ids:
            cloudsI = cloudsI_all_by_rep[rep]
            s = []
            for c in cloudsI:
                x = c[:, 0].astype(int)
                h = np.bincount(x, minlength=m).astype(float)
                s.append(max_abs_z_multinomial(h, p0))
            s = np.asarray(s, float)
            statI_by_rep[rep] = s
            all_statI.append(s)

        all_statI = np.concatenate(all_statI, axis=0)
        UCL = sidak_ucl_for_max_abs_z(alpha0, m)
        lim = {"UCL": float(UCL), "alpha": float(alpha0), "chart": "max_abs_z", "m": int(m), "p0": p0.tolist()}

        with open(out_m / "limits_ATTR.json", "w", encoding="utf-8") as f:
            json.dump(lim, f, indent=2)

        for scen, files in scen_files.items():
            out_s = safe_mkdir(out_m / scen)
            for rep, f in zip(rep_ids, files):
                cloudsII = load_clouds(f, "streams")
                sII = []
                for c in cloudsII:
                    x = c[:, 0].astype(int)
                    h = np.bincount(x, minlength=m).astype(float)
                    sII.append(max_abs_z_multinomial(h, p0))
                sII = np.asarray(sII, float)
                sI = statI_by_rep[rep]

                save_scores_csv(
                    out_dir=out_s, method=method, scen=scen, rep=rep,
                    stat_name="max_abs_z", sI=sI, sII=sII,
                )

                RL, nA = compute_RL_and_alarms_one_sided(sII, lim["UCL"])
                rows.append({
                    "cfg_tag": cfg_tag, "mode": mode, "method": method,
                    "scenario": scen, "rep": rep, "stat": "max_abs_z",
                    "ARL": RL, "n_alarm": nA, "H": int(len(sII)),
                })

                plot_single_stat_OC(
                    statI=sI, statII=sII,
                    UCL=lim["UCL"], LCL=None,
                    out_prefix=out_s / f"chart_{method}_{scen}",
                    method_label=method, cfg_name=f"{cfg_tag}/{scen}",
                    rep_id=rep, ylabel="max|z|",
                )

    # ==========================================================
    # Write results + summary
    # ==========================================================
    df = pd.DataFrame(rows)
    df.to_csv(out_cfg / "results_rows.csv", index=False)

    if not df.empty:
        summary = (
            df.groupby(["cfg_tag", "mode", "method", "scenario", "stat"], dropna=False)
              .agg(ARL_mean=("ARL", "mean"),
                   ARL_se=("ARL", lambda x: x.std(ddof=1) / max(np.sqrt(len(x)), 1.0)),
                   n_alarm_mean=("n_alarm", "mean"))
              .reset_index()
        )
        summary.to_csv(out_cfg / "summary.csv", index=False)

    print(f"[OK] {cfg_tag} done. Results in {out_cfg}")


def main():
    N_POINTS = 100
    if len(sys.argv) > 1:
        N_POINTS = int(sys.argv[1])

    DATA_ROOT = Path(os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent / "data"))
    datadir = DATA_ROOT / "discrete" / f"n{N_POINTS}"
    resdir  = datadir / f"results_discrete_all_methods_n{N_POINTS}"
    safe_mkdir(resdir)

    rep_ids = list(range(10))
    alpha0 = 0.05

    # Poisson spike injection (Fig. 2a)
    run_one_config_all_methods(
        datadir=datadir, resdir=resdir,
        cfg_tag="dust_poisson_1d",
        mode="dust_poisson_1d",
        rep_ids=rep_ids,
        alpha0=alpha0,
        m=6,
    )

    # Ordered categorical drift (Fig. 2b)
    run_one_config_all_methods(
        datadir=datadir, resdir=resdir,
        cfg_tag="cat_m6",
        mode="categorical",
        rep_ids=rep_ids,
        alpha0=alpha0,
        m=6,
    )


if __name__ == "__main__":
    main()
