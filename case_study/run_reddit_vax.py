# -*- coding: utf-8 -*-
"""
Reddit CovidVaccine case-study runner (no replications)

This script mirrors the structure of FLOWCAP/run_charts.py:
  - OT -> mFPCA (proposed)
  - KDE -> mFPCA (baseline)
  - Hotelling T2 on daily means (baseline)

Case-study specifics:
  - Builds *daily* distribution-valued samples (each day = one cloud), cap <= 500 points/day.
  - Caps Phase I to the *last N=50* valid days before a chosen cutoff event.
  - Marks multiple policy/news events as vertical lines on the plots.

Configure via environment variables:
  - DIDO_REDDIT_CSV:     path to SummaryResults_Covid_All.csv
  - DIDO_REDDIT_DATADIR: working directory for exports and results
Optionally edit in-script:
  - CUTOFF_NAME, N_PHASE1, MIN_PER_DAY, N_PHASE2
  - REPRESENTATIONS to include "embed_pca20" if you want embedding-based distributions
"""

from __future__ import annotations
from pathlib import Path
import json, math, time, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "continuous_streams"))
sys.path.insert(0, str(Path(__file__).parent.parent / "idd_core"))

from common_mfpca import call_r_mfpca_scores, compute_RL_and_alarms
from ot_mfpca_flow import fit_ot_phaseI_for_rep, export_ot_phaseII_ic
from baselines import export_kde_for_mfpca, load_clouds

# ============================================================
# Config – set paths via environment variables or edit below
# ============================================================

# CSV_PATH: path to SummaryResults_Covid_All.csv from the dataset
# Download from: https://github.com/msamogh/covid-vaccine-twitter (or dataset README)
CSV_PATH = Path(os.environ.get(
    "DIDO_REDDIT_CSV",
    Path(__file__).parent / "data" / "SummaryResults_Covid_All.csv"
))

# DATADIR: working directory for intermediate npz exports and results
DATADIR = Path(os.environ.get(
    "DIDO_REDDIT_DATADIR",
    Path(__file__).parent / "output"
))

# Windowing controls
MAX_PER_DAY = 500
MIN_PER_DAY = 20
N_PHASE1    = 50
N_PHASE2    = None   # set to 50 if you want balanced Phase II
SEED        = 0

# Cutoff for Phase I/II
CUTOFF_NAME = "jj_eua"   # good default; gives you more post-cutoff days than jj_pause
ALPHA0      = 0.05

# Representations to run
# - "sentiment3d": [Polarity, Subjectivity, SentimentScore]
# - "embed_pca20": SBERT embeddings -> PCA(20) (requires sentence-transformers)
REPRESENTATIONS = ["sentiment3d"]  # add "embed_pca20" when ready
EMBED_MODEL = "all-MiniLM-L6-v2"
PCA_DIM     = 20

# OT settings
BARY_MODE = "flow"
OT_MODE   = "sb"
REG       = 0.05
N_BARY    = 512

# Results folders
RESDIR = DATADIR / "results_reddit_case"
RESDIR.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT = DATADIR / "exports_reddit_case"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

# Events to mark on plots (UTC)
EVENTS_UTC = {
    "pfizer_efficacy_pr":   pd.Timestamp("2020-11-09", tz="UTC"),
    "pfizer_fda_eua":       pd.Timestamp("2020-12-11", tz="UTC"),
    "moderna_fda_eua":      pd.Timestamp("2020-12-18", tz="UTC"),
    "jj_eua":               pd.Timestamp("2021-02-27", tz="UTC"),
    "jj_pause":             pd.Timestamp("2021-04-13", tz="UTC"),
    "all_adults_eligible":  pd.Timestamp("2021-04-19", tz="UTC"),
    "jj_pause_lifted":      pd.Timestamp("2021-04-23", tz="UTC"),
}


# ============================================================
# Helpers
# ============================================================

def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # NOTE: for this case study, we set y_true_II = 1 for all Phase II days.
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc  = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1   = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    bal_acc = 0.5 * (rec + spec)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) else 0.0
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0

    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=acc, precision=prec, recall=rec, specificity=spec,
        f1=f1, balanced_accuracy=bal_acc, mcc=mcc,
    )

def _split_phase(df_scores: pd.DataFrame):
    ph = df_scores["phase"].astype(str).str.upper().str.strip()
    dfI  = df_scores.loc[ph.str.startswith("I") & ~ph.str.startswith("II")].copy()
    dfII = df_scores.loc[ph.str.startswith("II")].copy()
    return dfI, dfII

def sentiment_to_score(s: str) -> float:
    s = str(s).strip().lower()
    if s.startswith("neg"):
        return -1.0
    if s.startswith("pos"):
        return  1.0
    return 0.0

def load_reddit_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    df = df[df["created_utc"].notna()].copy()

    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() > 0]
    df = df[~df["text"].str.lower().isin(["[deleted]", "[removed]"])]

    df["is_root"] = df["Parent"].astype(str).eq("1")
    return df

def make_daily_windows(df_comments: pd.DataFrame) -> list[tuple[pd.Timestamp, np.ndarray]]:
    """
    Returns list of (day_start_utc, row_indices) with MIN_PER_DAY <= n <= MAX_PER_DAY.
    """
    rng = np.random.default_rng(SEED)
    dfc = df_comments.copy()
    dfc["day"] = dfc["created_utc"].dt.floor("D")

    win_list = []
    for d, g in dfc.groupby("day"):
        idx = g.index.to_numpy()
        if len(idx) < MIN_PER_DAY:
            continue
        if len(idx) > MAX_PER_DAY:
            idx = rng.choice(idx, size=MAX_PER_DAY, replace=False)
        win_list.append((pd.Timestamp(d).tz_localize("UTC"), idx))

    win_list.sort(key=lambda x: x[0])
    return win_list

def split_by_cutoff_capped(
    win_list: list[tuple[pd.Timestamp, np.ndarray]],
    cutoff: pd.Timestamp,
    *,
    n_phase1: int = 50,
    n_phase2: int | None = None,
):
    pre = [(w, idx) for (w, idx) in win_list if w < cutoff]
    post = [(w, idx) for (w, idx) in win_list if w >= cutoff]
    if n_phase1 is not None and len(pre) > n_phase1:
        pre = pre[-n_phase1:]
    if n_phase2 is not None and len(post) > n_phase2:
        post = post[:n_phase2]
    return pre, post

def build_cloud_sentiment3d(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    sub = df.loc[idx]
    X = np.column_stack([
        sub["Polarity"].astype(float).to_numpy(),
        sub["Subjectivity"].astype(float).to_numpy(),
        sub["Sentiment"].map(sentiment_to_score).astype(float).to_numpy(),
    ])
    # clamp to known bounds
    X[:, 0] = np.clip(X[:, 0], -1.0, 1.0)
    X[:, 1] = np.clip(X[:, 1],  0.0, 1.0)
    X[:, 2] = np.clip(X[:, 2], -1.0, 1.0)
    return X

def build_clouds_embed_pca(
    df: pd.DataFrame,
    win_all: list[tuple[pd.Timestamp, np.ndarray]],
    phase1: list[tuple[pd.Timestamp, np.ndarray]],
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Embed all texts once, fit PCA on Phase I only, return dict(day -> n x PCA_DIM cloud).
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
    except Exception as e:
        raise RuntimeError(
            "Embedding mode requires sentence-transformers + scikit-learn.\n"
            "pip install sentence-transformers scikit-learn"
        ) from e

    all_idx = np.concatenate([idx for (_, idx) in win_all], axis=0)
    all_texts = df.loc[all_idx, "text"].tolist()

    cache_path = EXPORT_ROOT / "embed_cache_all.npz"
    emb = None
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        emb = cache.get("emb", None)
        cached_idx = cache.get("indices", None)
        if emb is None or cached_idx is None or len(cached_idx) != len(all_idx) or (cached_idx != all_idx).any():
            emb = None

    if emb is None:
        model = SentenceTransformer(EMBED_MODEL)
        emb = model.encode(
            all_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.savez_compressed(cache_path, emb=emb, indices=all_idx.astype(np.int64))

    idx_to_pos = {int(i): j for j, i in enumerate(all_idx)}

    phase1_idx = np.concatenate([idx for (_, idx) in phase1], axis=0)
    phase1_pos = np.array([idx_to_pos[int(i)] for i in phase1_idx], dtype=int)

    pca = PCA(n_components=PCA_DIM, random_state=0)
    pca.fit(emb[phase1_pos])

    np.savez_compressed(
        EXPORT_ROOT / "embed_pca_meta.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    clouds = {}
    for (w, idx) in win_all:
        pos = np.array([idx_to_pos[int(i)] for i in idx], dtype=int)
        clouds[w] = pca.transform(emb[pos])
    return clouds

def save_cloud_npz(path: Path, key: str, clouds: list[np.ndarray]):
    """
    Save list of variable-size clouds as npz with dtype=object.
    Keys match your FlowCAP conventions: Phase I -> 'phaseI', Phase II -> 'streams'.
    """
    arr = np.empty(len(clouds), dtype=object)
    for i, c in enumerate(clouds):
        arr[i] = c
    np.savez_compressed(path, **{key: arr})

def write_meta_csv(path: Path, windows: list[pd.Timestamp], clouds: list[np.ndarray], *, is_ooc: int | None = None):
    dfm = pd.DataFrame({
        "window_start": [w.isoformat() for w in windows],
        "n": [int(c.shape[0]) for c in clouds],
    })
    if is_ooc is not None:
        dfm["is_ooc"] = int(is_ooc)
    dfm.to_csv(path, index=False)

def write_sample_sizes_csv(path: Path, winI: list[pd.Timestamp], winII: list[pd.Timestamp],
                           clouds_I: list[np.ndarray], clouds_II: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "idx": np.arange(1, len(winI) + len(winII) + 1, dtype=int),
        "phase": ["I"] * len(winI) + ["II"] * len(winII),
        "window_start": [w.isoformat() for w in (winI + winII)],
        "n": [int(c.shape[0]) for c in (clouds_I + clouds_II)],
    }).to_csv(path, index=False)

def plot_T2_SPE_with_events(
    df_scores: pd.DataFrame,
    windows_all: list[pd.Timestamp],
    UCL_T2: float,
    UCL_SPE: float,
    out_prefix: Path,
    method_label: str,
    title: str,
    events: dict[str, pd.Timestamp],
):
    """
    Two-panel plot (T2 and SPE) vs time index, with:
      - Phase boundary (first II index)
      - vertical lines for each event (mapped to first window >= event time)
    """
    import matplotlib.pyplot as plt

    x = np.arange(1, len(df_scores) + 1)
    phase = df_scores["phase"].astype(str).str.upper().str.strip().to_numpy()

    # Map window_start -> x index (1-based)
    w_to_x = {w: i+1 for i, w in enumerate(windows_all)}

    # Event x positions (first day >= event time)
    event_x = []
    for name, t in events.items():
        cand = [w for w in windows_all if w >= t]
        if not cand:
            continue
        w_evt = cand[0]
        event_x.append((name, w_to_x[w_evt]))

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    axes[0].plot(x, df_scores["T2"].to_numpy())
    axes[0].axhline(UCL_T2, linestyle="--")
    axes[0].set_ylabel("T2")

    axes[1].plot(x, df_scores["SPE"].to_numpy())
    if np.isfinite(UCL_SPE):
        axes[1].axhline(UCL_SPE, linestyle="--")
    axes[1].set_ylabel("SPE")
    axes[1].set_xlabel("Day index (Phase I then Phase II)")

    # Phase boundary
    ii_idx = np.where(np.char.startswith(phase, "II"))[0]
    if ii_idx.size > 0:
        xb = int(ii_idx[0] + 1)
        for ax in axes:
            ax.axvline(xb, linestyle=":")

    # Events
    for name, xe in event_x:
        for ax in axes:
            ax.axvline(xe, linestyle="-", linewidth=1.0, alpha=0.5)
        axes[0].text(
            xe, axes[0].get_ylim()[1],
            name, rotation=90, va="top", ha="right", fontsize=8
        )

    fig.suptitle(f"{title}\n{method_label}")
    fig.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_prefix) + "_T2_SPE_events.png", dpi=200)
    plt.close(fig)


# ============================================================
# Main runner
# ============================================================

def run_one_representation(rep_name: str):
    print(f"\n=== Reddit case-study: {rep_name} ===")

    df = load_reddit_csv(CSV_PATH)
    df_comments = df[~df["is_root"]].copy()

    # Build daily windows (filtered by MIN_PER_DAY and capped by MAX_PER_DAY)
    win_all = make_daily_windows(df_comments)
    if len(win_all) < (N_PHASE1 + 5):
        raise RuntimeError(
            f"Too few valid daily windows ({len(win_all)}). "
            "Lower MIN_PER_DAY or expand dataset."
        )

    cutoff = EVENTS_UTC[CUTOFF_NAME]
    phase1, phase2 = split_by_cutoff_capped(win_all, cutoff, n_phase1=N_PHASE1, n_phase2=N_PHASE2)

    print(f"Cutoff={CUTOFF_NAME} ({cutoff.date()})  Phase I={len(phase1)}  Phase II={len(phase2)}")
    if phase1:
        print("  Phase I range:", phase1[0][0].date(), "→", phase1[-1][0].date())
    if phase2:
        print("  Phase II range:", phase2[0][0].date(), "→", phase2[-1][0].date())

    # Build point-clouds per day
    if rep_name == "sentiment3d":
        clouds_I  = [build_cloud_sentiment3d(df_comments, idx) for (_, idx) in phase1]
        clouds_II = [build_cloud_sentiment3d(df_comments, idx) for (_, idx) in phase2]
    elif rep_name == "embed_pca20":
        clouds_map = build_clouds_embed_pca(df_comments, win_all, phase1)
        clouds_I  = [clouds_map[w] for (w, _) in phase1]
        clouds_II = [clouds_map[w] for (w, _) in phase2]
    else:
        raise ValueError(f"Unknown representation: {rep_name}")

    rows = []
    out_cfg = RESDIR / rep_name
    out_cfg.mkdir(parents=True, exist_ok=True)

    # Export Phase I / II in FlowCAP-like npz format
    rep_tag = f"rep00_{rep_name}"
    phaseI_file = DATADIR / f"reddit_{rep_tag}_phaseI.npz"
    phaseII_file = DATADIR / f"reddit_{rep_tag}_phaseII.npz"
    metaI_csv  = DATADIR / f"reddit_{rep_tag}_phaseI_meta.csv"
    metaII_csv = DATADIR / f"reddit_{rep_tag}_phaseII_meta.csv"

    save_cloud_npz(phaseI_file, "phaseI", clouds_I)
    save_cloud_npz(phaseII_file, "streams", clouds_II)

    winI  = [w for (w, _) in phase1]
    winII = [w for (w, _) in phase2]
    write_meta_csv(metaI_csv, winI, clouds_I, is_ooc=0)
    # In event-based case-study reporting, treat Phase II as post-change (is_ooc=1)
    write_meta_csv(metaII_csv, winII, clouds_II, is_ooc=1)
    write_sample_sizes_csv(out_cfg / "sample_sizes" / "sample_sizes.csv", winI, winII, clouds_I, clouds_II)

    # Timeline mapping (for debugging / paper appendix)
    timeline_csv = DATADIR / f"reddit_{rep_tag}_timeline.csv"
    pd.DataFrame({
        "phase": ["I"] * len(winI) + ["II"] * len(winII),
        "window_start": [w.isoformat() for w in (winI + winII)],
        "n": [int(c.shape[0]) for c in (clouds_I + clouds_II)],
    }).to_csv(timeline_csv, index=False)

    # ========================================================
    # 1) Proposed: OT -> mFPCA
    # ========================================================
    print("  [OT] running OT-MFPCA...")
    out_dir_ot = out_cfg / "OT"
    out_dir_ot.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    fit = fit_ot_phaseI_for_rep(
        phaseI_file=phaseI_file,
        cfg_name=f"reddit_{rep_name}",
        ot_export_root=(EXPORT_ROOT / rep_name / "ot"),
        reg=REG,
        bary_mode=BARY_MODE,
        ot_mode=OT_MODE,
        n_bary=N_BARY,
    )
    t_phaseI = float(getattr(fit, "timing", {}).get("t_phaseI", time.perf_counter() - t0))

    base_II, timing_II = export_ot_phaseII_ic(
        fit=fit,
        phaseII_ic_file=phaseII_file,
        reg=REG,
        ot_mode=OT_MODE,
        do_compare_ot=False,
    )
    t_phaseII = float(timing_II.get("t_phaseII", 0.0))

    df_scores, t_mfpca = call_r_mfpca_scores(
        base_prefix=base_II,
        out_dir=out_dir_ot,
        export_stem=f"reddit_{rep_tag}_OT",
    )
    t_mfpca = float(t_mfpca or 0.0)

    dfI, dfII = _split_phase(df_scores)
    UCL_T2  = float(np.quantile(dfI["T2"].to_numpy(),  1 - ALPHA0))
    UCL_SPE = float(np.quantile(dfI["SPE"].to_numpy(), 1 - ALPHA0))

    plot_T2_SPE_with_events(
        df_scores=df_scores,
        windows_all=(winI + winII),
        UCL_T2=UCL_T2,
        UCL_SPE=UCL_SPE,
        out_prefix=(out_dir_ot / f"reddit_{rep_tag}_OT"),
        method_label="OT",
        title=f"Reddit CovidVaccine ({rep_name})",
        events=EVENTS_UTC,
    )

    # For case study: y_true_II not observed; treat all Phase II as post-change.
    y_true_II = np.ones(len(dfII), dtype=int)
    pred_T2  = (dfII["T2"].to_numpy()  > UCL_T2).astype(int)
    pred_SPE = (dfII["SPE"].to_numpy() > UCL_SPE).astype(int)
    pred_OR  = ((pred_T2 == 1) | (pred_SPE == 1)).astype(int)

    RL_T2,  n_alarm_T2  = compute_RL_and_alarms(dfII["T2"].to_numpy(),  UCL_T2)
    RL_SPE, n_alarm_SPE = compute_RL_and_alarms(dfII["SPE"].to_numpy(), UCL_SPE)
    stat_or = np.maximum(
        dfII["T2"].to_numpy()  / max(UCL_T2,  1e-12),
        dfII["SPE"].to_numpy() / max(UCL_SPE, 1e-12),
    )
    RL_OR, n_alarm_OR = compute_RL_and_alarms(stat_or, 1.0)

    m_T2  = binary_classification_metrics(y_true_II, pred_T2)
    m_SPE = binary_classification_metrics(y_true_II, pred_SPE)
    m_OR  = binary_classification_metrics(y_true_II, pred_OR)

    runtime_total = t_phaseI + t_phaseII + t_mfpca
    for rule, RL, n_alarm, mm in [
        ("T2",  RL_T2,  n_alarm_T2,  m_T2),
        ("SPE", RL_SPE, n_alarm_SPE, m_SPE),
        ("OR",  RL_OR,  n_alarm_OR,  m_OR),
    ]:
        ucl_rule = UCL_T2 if rule == "T2" else UCL_SPE if rule == "SPE" else 1.0
        rows.append(dict(
            representation=rep_name,
            method="OT",
            rule=rule,
            RL=float(RL),
            n_alarm=float(n_alarm),
            runtime=float(runtime_total),
            runtime_phaseI=float(t_phaseI),
            runtime_phaseII=float(t_phaseII),
            runtime_mfpca=float(t_mfpca),
            UCL=float(ucl_rule),
            UCL_T2=float(UCL_T2),
            UCL_SPE=float(UCL_SPE),
            **mm,
        ))

    # ========================================================
    # 2) Baseline: KDE -> mFPCA
    # ========================================================
    # For higher dims (PCA20), "marginal" is typically more stable than "full"
    kde_modes = ["full", "marginal"] if rep_name == "sentiment3d" else ["marginal"]

    for kde_mode in kde_modes:
        print(f"  [KDE-{kde_mode}] running KDE baseline...")
        out_dir_kde = out_cfg / f"KDE_{kde_mode}"
        out_dir_kde.mkdir(parents=True, exist_ok=True)

        base_prefix, timing = export_kde_for_mfpca(
            phaseI_file=phaseI_file,
            phaseII_file=phaseII_file,
            cfg_name=f"reddit_{rep_name}",
            kde_export_root=(EXPORT_ROOT / rep_name / "kde"),
            bandwidth=None,
            mode=kde_mode,
        )

        df_scores, t_mfpca = call_r_mfpca_scores(
            base_prefix=base_prefix,
            out_dir=out_dir_kde,
            export_stem=f"reddit_{rep_tag}_KDE_{kde_mode}",
        )
        t_mfpca = float(t_mfpca or 0.0)

        dfI, dfII = _split_phase(df_scores)
        UCL_T2  = float(np.quantile(dfI["T2"].to_numpy(),  1 - ALPHA0))
        UCL_SPE = float(np.quantile(dfI["SPE"].to_numpy(), 1 - ALPHA0))

        plot_T2_SPE_with_events(
            df_scores=df_scores,
            windows_all=(winI + winII),
            UCL_T2=UCL_T2,
            UCL_SPE=UCL_SPE,
            out_prefix=(out_dir_kde / f"reddit_{rep_tag}_KDE_{kde_mode}"),
            method_label=f"KDE-{kde_mode}",
            title=f"Reddit CovidVaccine ({rep_name})",
            events=EVENTS_UTC,
        )

        y_true_II = np.ones(len(dfII), dtype=int)
        pred_T2  = (dfII["T2"].to_numpy()  > UCL_T2).astype(int)
        pred_SPE = (dfII["SPE"].to_numpy() > UCL_SPE).astype(int)
        pred_OR  = ((pred_T2 == 1) | (pred_SPE == 1)).astype(int)

        RL_T2,  n_alarm_T2  = compute_RL_and_alarms(dfII["T2"].to_numpy(),  UCL_T2)
        RL_SPE, n_alarm_SPE = compute_RL_and_alarms(dfII["SPE"].to_numpy(), UCL_SPE)
        stat_or = np.maximum(
            dfII["T2"].to_numpy()  / max(UCL_T2,  1e-12),
            dfII["SPE"].to_numpy() / max(UCL_SPE, 1e-12),
        )
        RL_OR, n_alarm_OR = compute_RL_and_alarms(stat_or, 1.0)

        m_T2  = binary_classification_metrics(y_true_II, pred_T2)
        m_SPE = binary_classification_metrics(y_true_II, pred_SPE)
        m_OR  = binary_classification_metrics(y_true_II, pred_OR)

        t_kde_phaseI  = float(timing.get("t_phaseI", 0.0))
        t_kde_phaseII = float(timing.get("t_phaseII", 0.0))
        runtime_total = t_kde_phaseI + t_kde_phaseII + t_mfpca

        for rule, RL, n_alarm, mm in [
            ("T2",  RL_T2,  n_alarm_T2,  m_T2),
            ("SPE", RL_SPE, n_alarm_SPE, m_SPE),
            ("OR",  RL_OR,  n_alarm_OR,  m_OR),
        ]:
            ucl_rule = UCL_T2 if rule == "T2" else UCL_SPE if rule == "SPE" else 1.0
            rows.append(dict(
                representation=rep_name,
                method=f"KDE_{kde_mode}",
                rule=rule,
                RL=float(RL),
                n_alarm=float(n_alarm),
                runtime=float(runtime_total),
                runtime_phaseI=float(t_kde_phaseI),
                runtime_phaseII=float(t_kde_phaseII),
                runtime_mfpca=float(t_mfpca),
                UCL=float(ucl_rule),
                UCL_T2=float(UCL_T2),
                UCL_SPE=float(UCL_SPE),
                **mm,
            ))

    # ========================================================
    # 3) Baseline: Hotelling T2 on daily means
    # ========================================================
    print("  [HotellingT2] running mean/Hotelling baseline...")

    cloudsI = load_clouds(phaseI_file, "phaseI")
    meansI = np.vstack([c.mean(axis=0, keepdims=True) for c in cloudsI])
    d = int(meansI.shape[1])

    mu0 = meansI.mean(axis=0)
    Sigma0 = np.cov(meansI, rowvar=False, ddof=1) + 1e-8 * np.eye(d)
    Sigma0_inv = np.linalg.pinv(Sigma0)

    Xm = meansI - mu0[None, :]
    T2_I = np.einsum("ni,ij,nj->n", Xm, Sigma0_inv, Xm)

    UCL_T2 = float(np.quantile(T2_I, 1 - ALPHA0))
    UCL_SPE = float("inf")

    cloudsII = load_clouds(phaseII_file, "streams")
    meansII = np.vstack([c.mean(axis=0, keepdims=True) for c in cloudsII])
    Xm2 = meansII - mu0[None, :]
    T2_II = np.einsum("ni,ij,nj->n", Xm2, Sigma0_inv, Xm2)

    df_scores_ht = pd.concat([
        pd.DataFrame({"phase": ["I"] * len(T2_I),  "T2": T2_I,  "SPE": 0.0}),
        pd.DataFrame({"phase": ["II"] * len(T2_II), "T2": T2_II, "SPE": 0.0}),
    ], ignore_index=True)

    plot_T2_SPE_with_events(
        df_scores=df_scores_ht,
        windows_all=(winI + winII),
        UCL_T2=UCL_T2,
        UCL_SPE=UCL_SPE,
        out_prefix=(out_cfg / "HotellingT2" / f"reddit_{rep_tag}_HotellingT2"),
        method_label="HotellingT2",
        title=f"Reddit CovidVaccine ({rep_name})",
        events=EVENTS_UTC,
    )
    hotelling_dir = out_cfg / "HotellingT2"
    hotelling_dir.mkdir(parents=True, exist_ok=True)
    df_scores_ht = df_scores_ht.copy()
    df_scores_ht["idx"] = np.arange(1, len(df_scores_ht) + 1, dtype=int)
    df_scores_ht.to_csv(hotelling_dir / f"reddit_{rep_tag}_HotellingT2_scores.csv", index=False)

    RL_T2, n_alarm_T2 = compute_RL_and_alarms(T2_II, UCL_T2)
    y_true_II = np.ones(len(T2_II), dtype=int)
    pred_T2 = (T2_II > UCL_T2).astype(int)
    m_T2 = binary_classification_metrics(y_true_II, pred_T2)

    rows.append(dict(
        representation=rep_name,
        method="HotellingT2",
        rule="T2",
        RL=float(RL_T2),
        n_alarm=float(n_alarm_T2),
        runtime=np.nan,  # add timing if you want
        runtime_phaseI=np.nan,
        runtime_phaseII=np.nan,
        runtime_mfpca=0.0,
        UCL=float(UCL_T2),
        UCL_T2=float(UCL_T2),
        UCL_SPE=float(UCL_SPE),
        **m_T2,
    ))

    # Save outputs
    rows_df = pd.DataFrame(rows)
    rows_path = out_cfg / f"rows_{rep_name}.csv"
    rows_df.to_csv(rows_path, index=False)

    summary = (
        rows_df.groupby(["representation", "method", "rule"], dropna=False)
               .agg(
                    RL_mean=("RL", "mean"),
                    n_alarm_mean=("n_alarm", "mean"),
                    runtime_mean=("runtime", "mean"),
                    accuracy_mean=("accuracy", "mean"),
                    precision_mean=("precision", "mean"),
                    recall_mean=("recall", "mean"),
                    specificity_mean=("specificity", "mean"),
                    f1_mean=("f1", "mean"),
                    balanced_accuracy_mean=("balanced_accuracy", "mean"),
                    mcc_mean=("mcc", "mean"),
                    UCL=("UCL", "mean"),
                    UCL_T2=("UCL_T2", "mean"),
                    UCL_SPE=("UCL_SPE", "mean"),
               ).reset_index()
    )
    summary_path = out_cfg / f"summary_{rep_name}.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(out_cfg / "summary.csv", index=False)

    meta = dict(
        representation=rep_name,
        cutoff_name=CUTOFF_NAME,
        cutoff_date=EVENTS_UTC[CUTOFF_NAME].isoformat(),
        min_per_day=MIN_PER_DAY,
        max_per_day=MAX_PER_DAY,
        n_phase1=N_PHASE1,
        n_phase2=N_PHASE2,
        alpha0=ALPHA0,
        events={k: v.isoformat() for k, v in EVENTS_UTC.items()},
    )
    with open(out_cfg / f"meta_{rep_name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    meta_with_tag = dict(meta, tag=f"reddit_rep00_{rep_name}")
    with open(out_cfg / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_with_tag, f, indent=2)

    print(f"  [OK] written:\n    {rows_path}\n    {summary_path}\n    {timeline_csv}")

def main():
    DATADIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if CUTOFF_NAME not in EVENTS_UTC:
        raise KeyError(f"CUTOFF_NAME={CUTOFF_NAME} not found in EVENTS_UTC")

    for rep in REPRESENTATIONS:
        run_one_representation(rep)

if __name__ == "__main__":
    main()
