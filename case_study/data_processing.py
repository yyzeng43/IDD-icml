from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


# -----------------------------
# 0) Event timeline (validated)
# -----------------------------
EVENTS_UTC = {
    "pfizer_efficacy_pr": pd.Timestamp("2020-11-09", tz="UTC"),
    "pfizer_fda_eua":     pd.Timestamp("2020-12-11", tz="UTC"),
    "moderna_fda_eua":    pd.Timestamp("2020-12-18", tz="UTC"),
    "jj_eua":             pd.Timestamp("2021-02-27", tz="UTC"),  # Phase I/II cutoff used in the paper
    "jj_pause":           pd.Timestamp("2021-04-13", tz="UTC"),
    "all_adults_eligible":pd.Timestamp("2021-04-19", tz="UTC"),
    "jj_pause_lifted":    pd.Timestamp("2021-04-23", tz="UTC"),
}


# -----------------------------------
# 1) Load + basic cleaning utilities
# -----------------------------------
def load_summaryresults_all(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse datetime (your file uses +00:00)
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")

    # Drop missing/empty text
    df["text"] = df["text"].astype(str)
    df = df[df["created_utc"].notna()]
    df = df[df["text"].str.len() > 0]

    # Optional: remove deleted/removed boilerplate
    bad = df["text"].str.lower().isin(["[deleted]", "[removed]"])
    df = df[~bad]

    # Identify roots vs comments:
    # In your file, roots have Parent == 1 (as string/int). Comments have Parent != 1.
    df["is_root"] = df["Parent"].astype(str).eq("1")

    return df


def add_week_start(df: pd.DataFrame) -> pd.DataFrame:
    # Week anchored to Monday; timezone dropped by Period conversion, so we restore UTC afterwards.
    wk = df["created_utc"].dt.to_period("W-MON").dt.start_time
    df = df.copy()
    df["week_start"] = pd.to_datetime(wk).dt.tz_localize("UTC")
    return df


def sentiment_to_score(s: str) -> float:
    # Maps text tag to numeric; stable and interpretable
    # Negative -> -1, Neutral -> 0, Positive -> +1
    s = str(s).strip().lower()
    if s.startswith("neg"):
        return -1.0
    if s.startswith("pos"):
        return 1.0
    return 0.0


# ---------------------------------------
# 2) Weekly sampling (cap at 500 samples)
# ---------------------------------------
@dataclass
class WeeklySample:
    week_start: pd.Timestamp
    indices: np.ndarray  # row indices from the original df
    n: int


def build_weekly_index(
    df: pd.DataFrame,
    *,
    include_roots: bool = False,
    max_per_week: int = 500,
    min_per_week: int = 30,
    seed: int = 0,
) -> List[WeeklySample]:
    rng = np.random.default_rng(seed)

    use = df.copy()
    if not include_roots:
        use = use[~use["is_root"]]

    # group by week
    samples: List[WeeklySample] = []
    for wk, g in use.groupby("week_start"):
        idx = g.index.to_numpy()

        if len(idx) < min_per_week:
            continue

        if len(idx) > max_per_week:
            idx = rng.choice(idx, size=max_per_week, replace=False)

        samples.append(WeeklySample(week_start=wk, indices=idx, n=len(idx)))

    samples.sort(key=lambda x: x.week_start)
    return samples


def pick_phase_windows(
    weekly: List[WeeklySample],
    phase1_end_utc: pd.Timestamp,
) -> Tuple[List[WeeklySample], List[WeeklySample]]:
    phase1 = [w for w in weekly if w.week_start < phase1_end_utc]
    phase2 = [w for w in weekly if w.week_start >= phase1_end_utc]
    return phase1, phase2


# ---------------------------------------
# 3) Approach 1: 3D sentiment feature set
# ---------------------------------------
def compute_features_sentiment3d(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    sub = df.loc[idx]
    x = np.column_stack([
        sub["Polarity"].astype(float).to_numpy(),
        sub["Subjectivity"].astype(float).to_numpy(),
        sub["Sentiment"].map(sentiment_to_score).astype(float).to_numpy(),
    ])
    # Optional: clip to documented ranges to avoid any parsing artifacts
    x[:, 0] = np.clip(x[:, 0], -1.0, 1.0)  # polarity
    x[:, 1] = np.clip(x[:, 1],  0.0, 1.0)  # subjectivity
    x[:, 2] = np.clip(x[:, 2], -1.0, 1.0)  # sentiment score
    return x


# ---------------------------------------
# 4) Approach 2: embeddings -> PCA (<=20D)
# ---------------------------------------
def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # helps stabilize distances
    )
    return emb


def compute_features_embedding_pca(
    df: pd.DataFrame,
    weekly: List[WeeklySample],
    phase1: List[WeeklySample],
    *,
    pca_dim: int = 20,
    model_name: str = "all-MiniLM-L6-v2",
    cache_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Dict[pd.Timestamp, np.ndarray], PCA]:
    """
    Returns:
      week_to_X: dict mapping week_start -> (n_week x pca_dim) array
      pca: fitted PCA object (fit on Phase I)
    """
    # Collect all indices used across all weeks
    all_idx = np.concatenate([w.indices for w in weekly], axis=0)
    all_texts = df.loc[all_idx, "text"].tolist()

    # Cache raw embeddings if requested
    if cache_path and os.path.exists(cache_path):
        raw = np.load(cache_path)["emb"]
    else:
        raw = embed_texts(all_texts, model_name=model_name, device=device)
        if cache_path:
            np.savez_compressed(cache_path, emb=raw, indices=all_idx.astype(np.int64))

    # Map index -> row in `raw`
    idx_to_pos = {int(i): j for j, i in enumerate(all_idx)}

    # Fit PCA on Phase I subset only (stable coordinate system for monitoring)
    phase1_idx = np.concatenate([w.indices for w in phase1], axis=0)
    phase1_pos = np.array([idx_to_pos[int(i)] for i in phase1_idx], dtype=int)

    pca = PCA(n_components=pca_dim, random_state=0)
    pca.fit(raw[phase1_pos])

    # Transform per week
    week_to_X: Dict[pd.Timestamp, np.ndarray] = {}
    for w in weekly:
        pos = np.array([idx_to_pos[int(i)] for i in w.indices], dtype=int)
        week_to_X[w.week_start] = pca.transform(raw[pos])

    return week_to_X, pca


# ---------------------------------------
# 5) End-to-end runner (both approaches)
# ---------------------------------------
def main(
    csv_path: str,
    out_dir: str,
    *,
    include_roots: bool = False,
    max_per_week: int = 500,
    min_per_week: int = 30,
    phase1_end: pd.Timestamp = EVENTS_UTC["pfizer_fda_eua"],  # default: Dec 11, 2020
    pca_dim: int = 20,
    embed_model: str = "all-MiniLM-L6-v2",
):
    os.makedirs(out_dir, exist_ok=True)

    df = load_summaryresults_all(csv_path)
    df = add_week_start(df)

    # Weekly sampling
    weekly = build_weekly_index(
        df,
        include_roots=include_roots,
        max_per_week=max_per_week,
        min_per_week=min_per_week,
        seed=0,
    )
    phase1, phase2 = pick_phase_windows(weekly, phase1_end)

    # Quick diagnostics
    counts = [(w.week_start.date().isoformat(), w.n) for w in weekly]
    pd.DataFrame(counts, columns=["week_start", "n"]).to_csv(
        os.path.join(out_dir, "weekly_counts.csv"), index=False
    )

    # ---- Approach 1: sentiment 3D ----
    week_to_X3 = {}
    for w in weekly:
        week_to_X3[w.week_start] = compute_features_sentiment3d(df, w.indices)

    np.savez_compressed(
        os.path.join(out_dir, "weekly_sentiment3d.npz"),
        weeks=np.array([w.week_start.value for w in weekly], dtype=np.int64),
        X=object_to_npobject_array([week_to_X3[w.week_start] for w in weekly]),
        phase1_mask=np.array([w.week_start < phase1_end for w in weekly], dtype=bool),
    )

    # ---- Approach 2: embeddings -> PCA <= 20D ----
    emb_cache = os.path.join(out_dir, "raw_embeddings_cache.npz")
    week_to_Xemb, pca = compute_features_embedding_pca(
        df,
        weekly=weekly,
        phase1=phase1,
        pca_dim=pca_dim,
        model_name=embed_model,
        cache_path=emb_cache,
    )

    np.savez_compressed(
        os.path.join(out_dir, f"weekly_embed_pca{pca_dim}.npz"),
        weeks=np.array([w.week_start.value for w in weekly], dtype=np.int64),
        X=object_to_npobject_array([week_to_Xemb[w.week_start] for w in weekly]),
        phase1_mask=np.array([w.week_start < phase1_end for w in weekly], dtype=bool),
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_ratio_,
    )

    print(f"Saved outputs to: {out_dir}")
    print(f"Total weekly samples kept: {len(weekly)}")
    print(f"Phase I weeks: {len(phase1)}, Phase II weeks: {len(phase2)}")
    print(f"Phase I end date (exclusive): {phase1_end.isoformat()}")


def object_to_npobject_array(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Saves ragged arrays (variable n per week) into a single np.ndarray(dtype=object)
    for npz export.
    """
    out = np.empty(len(list_of_arrays), dtype=object)
    for i, a in enumerate(list_of_arrays):
        out[i] = a
    return out


if __name__ == "__main__":
    import os, argparse
    parser = argparse.ArgumentParser(description="Preprocess Reddit COVID-vaccine dataset.")
    parser.add_argument("--csv", required=True, help="Path to SummaryResults_Covid_All.csv")
    parser.add_argument("--out", default=str(Path(__file__).parent / "output"), help="Output directory")
    parser.add_argument("--phase1_end", default="jj_eua",
                        choices=list(EVENTS_UTC.keys()), help="Phase I cutoff event (default: jj_eua)")
    args = parser.parse_args()

    main(
        args.csv,
        args.out,
        include_roots=False,
        max_per_week=500,
        min_per_week=30,
        phase1_end=EVENTS_UTC[args.phase1_end],
        pca_dim=20,
    )
