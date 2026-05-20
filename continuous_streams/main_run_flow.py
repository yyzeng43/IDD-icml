# use confitional flow to estimate the OT maps and barycenter
# run all the simulation scenarios

# main_run_empirical_all.py
from __future__ import annotations
from pathlib import Path
import math, json, time, os
import numpy as np
import pandas as pd
import sys
import traceback
import gc
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "idd_core"))

from baselines import export_kde_for_mfpca, compute_mean_charts_for_rep
from common_mfpca import call_r_mfpca_scores, compute_RL_and_alarms, plot_T2_SPE_OC
from ot_mfpca_flow import *

from baselines_cpd import (
    compute_newma_batch_baseline,
    compute_scanb_batch_baseline,
    compute_frechet_batch_baseline,
)


def save_cfg_checkpoint(rows, resdir: Path, cfg_name: str, suffix: str = ""):
    """
    Save raw rows + aggregated summary for a single cfg_name.
    """
    out_cfg = Path(resdir) / cfg_name
    out_cfg.mkdir(parents=True, exist_ok=True)

    df_cfg = pd.DataFrame([r for r in rows if r.get("cfg_name") == cfg_name] if any("cfg_name" in r for r in rows)
                          else [r for r in rows if (r.get("d"), r.get("K")) == tuple(map(int, cfg_name.replace("phase_d","").split("_K")))])

    raw_path = out_cfg / f"checkpoint_rows{suffix}.csv"
    df_cfg.to_csv(raw_path, index=False)

    # Aggregated (same grouping you do at the end)
    if not df_cfg.empty:
        summary_cfg = (
            df_cfg.groupby(["method","d","K","scenario","stat"], dropna=False)
                  .agg(ARL_mean=("ARL","mean"),
                       ARL_se=("ARL", lambda x: x.std(ddof=1)/max(len(x)**0.5,1.0)),
                       n_alarm_mean=("n_alarm","mean"),
                       runtime_mean=("runtime","mean"))
                  .reset_index()
        )
        summary_path = out_cfg / f"checkpoint_summary{suffix}.csv"
        summary_cfg.to_csv(summary_path, index=False)

    meta_path = out_cfg / f"checkpoint_meta{suffix}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"cfg_name": cfg_name, "n_rows": int(len(df_cfg))}, f, indent=2)

    print(f"[CHECKPOINT] saved: {raw_path}")

# ============================================================
# Configuration
# ============================================================

ALPHA0 = 0.05
# Three shift scenarios used in the paper:
#   "barycenter"  – barycenter change (shifting central mass)
#   "mm_reweight" – multimodal reweight (altering mixture weights)
#   "copula_shift"– copula shift (altering variable dependencies, fixed marginals)
SCENARIOS = ["IC", "mm_reweight", "copula_shift", "barycenter"]
D_LIST = [1, 5, 10, 50]   # dimensions
K_LIST = [4]               # number of mFPCA components
N_REP = 10                 # replications

N_POINTS = 100             # default batch size; overridden by CLI: python main_run_flow.py <N>
if len(sys.argv) > 1:
    N_POINTS = int(sys.argv[1])
print(f"Running with N_POINT = {N_POINTS}")

# ------------------------------------------------------------------
# DATA PATHS – set DATA_ROOT to the folder produced by data_generation/
# ------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("DIDO_DATA_ROOT", Path(__file__).parent.parent / "data"))
datadir = DATA_ROOT / "continuous" / f"n{N_POINTS}"
resdir = datadir / f"results_n{N_POINTS}"
resdir.mkdir(parents=True, exist_ok=True)
ot_export_root = datadir / f"ot_flow_exports_n{N_POINTS}"
ot_export_root.mkdir(parents=True, exist_ok=True)

rows = []

# ============================================================
# Main loops
# ============================================================

for d in D_LIST:
    for K in K_LIST:
        cfg_name = f"phase_d{d}_K{K}"
        print(f"\n=== CONFIG {cfg_name} ===")

        try:
            print("  [CPD] running F-CPD / NEWMA / Scan-B baselines...")

            cpd_rows = []
            cpd_rows += compute_newma_batch_baseline(
                datadir=datadir, resdir=resdir,
                d=d, K=K, scenarios=SCENARIOS, n_rep=N_REP,
                feature="rff", rff_m=256, rff_seed=0,
                lam_slow=0.01, lam_fast=0.05,
                alpha=ALPHA0,
            )
            cpd_rows += compute_scanb_batch_baseline(
                datadir=datadir, resdir=resdir,
                d=d, K=K, scenarios=SCENARIOS, n_rep=N_REP,
                feature="rff", rff_m=256, rff_seed=0,
                B=10, N=5,
                alpha=ALPHA0,
            )
            cpd_rows += compute_frechet_batch_baseline(
                datadir=datadir, resdir=resdir,
                d=d, K=K, scenarios=SCENARIOS, n_rep=N_REP,
                feature="rff", rff_m=256, rff_seed=0,
                swl=64, scan_c=0.1,
                alpha=ALPHA0,
            )
            cpd_rows = [r for r in cpd_rows if r.get("scenario") != "IC"]
            rows.extend(cpd_rows)




        except Exception as e:
            print(f"[ERROR] cfg={cfg_name} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            # keep going to next cfg instead of stopping the whole sweep
        finally:
            # always checkpoint whatever we have for this cfg so far
            save_cfg_checkpoint(rows, resdir=resdir, cfg_name=cfg_name)

# ============================================================
# Summary aggregation
# ============================================================

df = pd.DataFrame(rows)
summary = (
    df.groupby(["method", "d", "K", "scenario", "stat"], dropna=False)
      .agg(ARL_mean=("ARL", "mean"),
           ARL_se=("ARL", lambda x: x.std(ddof=1)/math.sqrt(len(x))),
           n_alarm_mean=("n_alarm", "mean"),
           runtime_mean=("runtime", "mean"))
      .reset_index()
)
summary_csv = resdir / "summary_cpd_baselines_n{}.csv".format(N_POINTS)
summary.to_csv(summary_csv, index=False)
print(f"\n[OK] Summary written to {summary_csv}")
