# OT for the mfpca

# ot_for_mfpca.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time, json
import numpy as np
from simulation.OT_KDE_Comp.run_scripts_all import save_tangents_for_R
from simulation.OT_KDE_Comp.ot_mfpca_closed import (process_phaseI,
                                                    process_phaseII,
                                                    barycenter_estimation,
                                                    _safe_cov)




# expects you already have these
# from ot_mfpca_closed import process_phaseI, process_phaseII, barycenter_estimation, _safe_cov
# from run_scripts_all2 import save_tangents_for_R


@dataclass
class OTFitPhaseI:
    cfg_name: str
    base_prefix: str          # ..._repXX
    out_dir: Path             # cfg folder under ot_export_root
    Xb: np.ndarray
    wb: np.ndarray
    tangI: list[np.ndarray]   # list of (G,d)
    cloudsI: list[np.ndarray] # raw Phase-I clouds
    timing: dict              # t_phaseI


def fit_ot_phaseI_for_rep(
    phaseI_file: Path,
    cfg_name: str,
    ot_export_root: Path,
    reg: float = 0.05,
    bary_mode: str = "empirical",   # {"empirical","gaussian"}
    ot_mode: str = "empirical",     # {"empirical","gaussian"}
    n_bary: int | None = 512,
) -> OTFitPhaseI:
    """
    Phase-I only:
      - estimate barycenter (empirical or Gaussian)
      - compute Phase-I OT tangents
    Saves nothing. Returns a fit object that can be reused for many Phase-II scenarios.
    """
    phaseI_file = Path(phaseI_file)
    cfg_ot_dir = Path(ot_export_root) / cfg_name
    cfg_ot_dir.mkdir(parents=True, exist_ok=True)

    stem = phaseI_file.stem
    base_prefix = stem.rsplit("_phaseI", 1)[0]

    print(f"[OT][FIT] {cfg_name}, {stem}: bary_mode={bary_mode}, ot_mode={ot_mode}")

    # ---- Phase I fit ----
    t0 = time.perf_counter()
    Xb, wb, tangI, ThatI, paramsI, cloudsI, tan_norms_I = process_phaseI(
        phaseI_file,
        reg=reg,
        n_bary=n_bary,
        bary_mode=bary_mode,
        ot_mode=ot_mode,
    )
    t_phaseI = time.perf_counter() - t0

    timing = {"t_phaseI": t_phaseI}


    return OTFitPhaseI(
        cfg_name=cfg_name,
        base_prefix=base_prefix,
        out_dir=cfg_ot_dir,
        Xb=Xb, wb=wb,
        tangI=tangI,
        cloudsI=cloudsI,
        timing=timing,
    )


def export_ot_phaseII_ic(
    fit: OTFitPhaseI,
    phaseII_ic_file: Path,
    reg: float = 0.05,
    ot_mode: str = "empirical",
    do_compare_ot: bool = False,
) -> tuple[Path, dict]:
    """
    Export IC Phase-II tangents (one file per rep) using the Phase-I fit.
    Returns:
      base_ic, timing_ic, compare_report_ic (optional)
    """
    phaseII_ic_file = Path(phaseII_ic_file)

    print(f"[OT][IC] {fit.cfg_name}, {fit.base_prefix}: ot_mode={ot_mode}")

    t1 = time.perf_counter()
    tangII_ic, ThatII_ic, paramsII_ic, streamsII_ic, tan_norms_ic, Txs_true_ic = process_phaseII(
        phaseII_ic_file, fit.Xb, fit.wb, reg=reg, ot_mode=ot_mode
    )
    t_phaseII_ic = time.perf_counter() - t1

    base_ic = fit.out_dir / f"{fit.base_prefix}_OT_IC"
    save_tangents_for_R(fit.Xb, fit.wb, fit.tangI, tangII_ic, base_ic)

    timing_ic = {"t_phaseII": t_phaseII_ic}

    return base_ic, timing_ic


def export_ot_phaseII_oc(
    fit: OTFitPhaseI,
    phaseII_oc_file: Path,
    scenario: str,
    reg: float = 0.05,
    ot_mode: str = "empirical",
) -> tuple[Path, dict]:
    """
    Export one OC scenario Phase-II tangents using the Phase-I fit.
    The scenario string is used in the filename suffix.
    Returns base_oc, timing_oc.
    """
    phaseII_oc_file = Path(phaseII_oc_file)
    scen_tag = str(scenario)

    print(f"[OT][OC] {fit.cfg_name}, {fit.base_prefix}, scen={scen_tag}: ot_mode={ot_mode}")

    t2 = time.perf_counter()
    tangII_oc, ThatII_oc, paramsII_oc, streamsII_oc, tan_norms_oc, Txs_true_oc = process_phaseII(
        phaseII_oc_file, fit.Xb, fit.wb, reg=reg, ot_mode=ot_mode
    )
    t_phaseII_oc = time.perf_counter() - t2

    base_oc = fit.out_dir / f"{fit.base_prefix}_OT_{scen_tag}"
    save_tangents_for_R(fit.Xb, fit.wb, fit.tangI, tangII_oc, base_oc)

    timing_oc = {"t_phaseII": t_phaseII_oc}
    return base_oc, timing_oc

