"""
Create a Phase-II monitoring figure for the Reddit vaccine case study.

Loads saved outputs from run_reddit_vax.py and produces a multi-panel
control chart comparing IDD (OT-MFPCA) against CPD baselines.

Usage:
    python visualization_cpd.py --results-base /path/to/results/sentiment3d
    python visualization_cpd.py --results-base /path/to/results/sentiment3d --out fig.pdf

Configure via environment variable:
    DIDO_REDDIT_DATADIR: working directory (default: case_study/output/)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


_DATADIR = Path(os.environ.get("DIDO_REDDIT_DATADIR", Path(__file__).parent / "output"))
DEFAULT_RESULTS_BASE = _DATADIR / "results_reddit_case" / "sentiment3d"
OUTPUT_DIR = _DATADIR / "figures"
DEFAULT_OUT = OUTPUT_DIR / "reddit_monitoring_baselines_phaseII.pdf"


EVENTS = [
    ("2021-02-27", "J&J EUA", "regulatory"),
    ("2021-03-05", "J&J rejected", "controversy"),
    ("2021-03-11", "Biden 100M", "policy"),
    ("2021-03-19", "J&J data question", "trust"),
    ("2021-04-02", "4th wave concern", "epi"),
    ("2021-04-13", "J&J pause", "shock"),
    ("2021-04-19", "All adults eligible", "policy"),
    ("2021-04-23", "J&J resumed", "regulatory"),
    ("2021-04-27", "Mask relaxed", "policy"),
    ("2021-05-04", "July 4 goal", "policy"),
]

CAT_COLORS = {
    "regulatory": "#1E88E5",
    "controversy": "#FB8C00",
    "policy": "#43A047",
    "trust": "#F4511E",
    "epi": "#8E24AA",
    "shock": "#E53935",
}

MAJOR_SPANS = [
    ("2021-04-13", "2021-04-23", "#E53935", "J&J pause period"),
]


@dataclass(frozen=True)
class PanelSpec:
    display_name: str
    summary_method: str
    summary_rule: str
    subdir: str
    file_candidates: tuple[str, ...]
    value_candidates: tuple[str, ...]
    ylabel: str
    color: str


LOT_PANEL_SPECS = [
    PanelSpec(
        display_name="IDD (SPE)",
        summary_method="OT",
        summary_rule="SPE",
        subdir="OT",
        file_candidates=("{tag}_OT_scores.csv", "*_OT_scores.csv", "scores.csv"),
        value_candidates=("SPE",),
        ylabel="SPE",
        color="#C62828",
    ),
    PanelSpec(
        display_name="IDD (T2)",
        summary_method="OT",
        summary_rule="T2",
        subdir="OT",
        file_candidates=("{tag}_OT_scores.csv", "*_OT_scores.csv", "scores.csv"),
        value_candidates=("T2",),
        ylabel="T2",
        color="#1565C0",
    ),
]

BASELINE_PANEL_SPECS = [
    PanelSpec(
        display_name="Hotelling T2",
        summary_method="HotellingT2",
        summary_rule="T2",
        subdir="HotellingT2",
        file_candidates=("{tag}_HotellingT2_scores.csv", "*HotellingT2_scores.csv", "scores.csv"),
        value_candidates=("T2",),
        ylabel="T2",
        color="#6A1B9A",
    ),
    PanelSpec(
        display_name="Log-KDE",
        summary_method="KDE_full",
        summary_rule="T2",
        subdir="KDE_full",
        file_candidates=("{tag}_KDE_full_scores.csv", "*KDE_full_scores.csv", "scores.csv"),
        value_candidates=("T2",),
        ylabel="T2",
        color="#EF6C00",
    ),
    PanelSpec(
        display_name="KDE-marginal",
        summary_method="KDE_marginal",
        summary_rule="T2",
        subdir="KDE_marginal",
        file_candidates=("{tag}_KDE_marginal_scores.csv", "*KDE_marginal_scores.csv", "scores.csv"),
        value_candidates=("T2",),
        ylabel="T2",
        color="#00897B",
    ),
    PanelSpec(
        display_name="F-CPD",
        summary_method="F-CPD",
        summary_rule="score",
        subdir="FRECHET_rff",
        file_candidates=("{tag}_FRECHET_rff_series.csv", "*FRECHET*_series.csv", "series.csv"),
        value_candidates=("stat", "score"),
        ylabel="Score",
        color="#455A64",
    ),
    PanelSpec(
        display_name="NEWMA",
        summary_method="NEWMA",
        summary_rule="dist",
        subdir="NEWMA_rff",
        file_candidates=("{tag}_NEWMA_rff_series.csv", "*NEWMA*_series.csv", "series.csv"),
        value_candidates=("stat", "dist"),
        ylabel="Distance",
        color="#2E7D32",
    ),
    PanelSpec(
        display_name="Scan-B",
        summary_method="ScanB",
        summary_rule="dist",
        subdir="ScanB_rff",
        file_candidates=("{tag}_ScanB_rff_series.csv", "*ScanB*_series.csv", "series.csv"),
        value_candidates=("stat", "dist"),
        ylabel="Distance",
        color="#5D4037",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-base", type=Path, default=DEFAULT_RESULTS_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--phase2-start",
        default=None,
        help="Phase-II cutoff date (default: from meta.json or 2021-02-27).",
    )
    parser.add_argument(
        "--exclude-lot",
        action="store_true",
        help="Only plot baselines, excluding IDD SPE/T2 panels.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def coerce_datetime(values: pd.Series) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)


def load_summary(results_base: Path) -> pd.DataFrame:
    summary_path = results_base / "summary.csv"
    if not summary_path.exists():
        matches = sorted(results_base.glob("summary_*.csv"))
        if matches:
            summary_path = matches[0]
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return pd.read_csv(summary_path)


def load_sample_dates(results_base: Path) -> Optional[pd.DataFrame]:
    sample_path = results_base / "sample_sizes" / "sample_sizes.csv"
    if not sample_path.exists():
        return None
    sample_df = pd.read_csv(sample_path)
    if "window_start" not in sample_df.columns:
        return None
    sample_df = sample_df.copy()
    sample_df["date"] = coerce_datetime(sample_df["window_start"])
    sample_df["idx"] = np.arange(1, len(sample_df) + 1)
    cols = ["idx", "date"]
    if "phase" in sample_df.columns:
        cols.append("phase")
    return sample_df[cols]


def find_first_existing(method_dir: Path, tag: str, candidates: tuple[str, ...]) -> Optional[Path]:
    if not method_dir.exists():
        return None
    for candidate in candidates:
        pattern = candidate.format(tag=tag)
        if any(ch in pattern for ch in "*?[]"):
            matches = sorted(method_dir.glob(pattern))
            if matches:
                return matches[0]
        else:
            path = method_dir / pattern
            if path.exists():
                return path
    return None


def attach_dates(df: pd.DataFrame, sample_dates: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = coerce_datetime(df["date"])
        return df
    if sample_dates is None:
        return df
    if "idx" in df.columns:
        merged = df.merge(sample_dates, on="idx", how="left", suffixes=("", "_sample"))
        if "phase" not in merged.columns and "phase_sample" in merged.columns:
            merged["phase"] = merged["phase_sample"]
        if "phase_sample" in merged.columns:
            merged = merged.drop(columns=["phase_sample"])
        return merged
    if len(sample_dates) < len(df):
        return df
    aligned = df.reset_index(drop=True).copy()
    sample_trim = sample_dates.iloc[: len(aligned)].reset_index(drop=True)
    aligned["date"] = sample_trim["date"]
    if "phase" not in aligned.columns and "phase" in sample_trim.columns:
        aligned["phase"] = sample_trim["phase"]
    return aligned


def filter_phase_two(df: pd.DataFrame, phase2_start: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    if "phase" in out.columns:
        mask = out["phase"].astype(str).str.upper().eq("II")
        if mask.any():
            out = out.loc[mask].copy()
    if "date" in out.columns:
        out = out.loc[out["date"].notna() & (out["date"] >= phase2_start)].copy()
    return out.reset_index(drop=True)


def lookup_threshold(summary_df: pd.DataFrame, method: str, rule: str) -> Optional[float]:
    rows = summary_df.loc[
        summary_df["method"].astype(str).eq(method) & summary_df["rule"].astype(str).eq(rule)
    ]
    if rows.empty:
        rows = summary_df.loc[summary_df["method"].astype(str).eq(method)]
    if rows.empty:
        return None
    col = "UCL"
    if col not in rows.columns:
        col = "UCL_T2" if rule == "T2" and "UCL_T2" in rows.columns else col
        col = "UCL_SPE" if rule == "SPE" and "UCL_SPE" in rows.columns else col
    if col not in rows.columns:
        return None
    value = pd.to_numeric(rows.iloc[0][col], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def choose_value_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_panel(
    spec: PanelSpec,
    *,
    results_base: Path,
    tag: str,
    summary_df: pd.DataFrame,
    sample_dates: Optional[pd.DataFrame],
    phase2_start: pd.Timestamp,
) -> Optional[dict]:
    method_dir = results_base / spec.subdir
    csv_path = find_first_existing(method_dir, tag, spec.file_candidates)
    if csv_path is None:
        print(f"[WARN] Missing result CSV for {spec.display_name} under {method_dir}")
        return None
    df = pd.read_csv(csv_path)
    df = attach_dates(df, sample_dates)
    df = filter_phase_two(df, phase2_start)
    if df.empty:
        print(f"[WARN] Phase-II data empty for {spec.display_name}: {csv_path}")
        return None
    value_col = choose_value_column(df, spec.value_candidates)
    if value_col is None:
        print(f"[WARN] No usable score column for {spec.display_name}: {csv_path}")
        return None
    if "date" not in df.columns or df["date"].isna().all():
        print(f"[WARN] Missing dates for {spec.display_name}: {csv_path}")
        return None
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    panel = {
        "method": spec.display_name,
        "label": spec.ylabel,
        "color": spec.color,
        "dates": df["date"].reset_index(drop=True),
        "values": values,
        "threshold": lookup_threshold(summary_df, spec.summary_method, spec.summary_rule),
        "source": csv_path,
        "value_col": value_col,
    }
    print(f"[OK] Loaded {spec.display_name}: {csv_path.name}")
    return panel


def draw_event_bar(ax: plt.Axes, x_start: pd.Timestamp, x_end: pd.Timestamp) -> None:
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Reddit Vaccine Sentiment: Phase-II Monitoring", fontsize=16, fontweight="bold", pad=10)

    y_positions = [0.92, 0.62, 0.30]
    for idx, (date_str, label, category) in enumerate(EVENTS):
        event_date = pd.Timestamp(date_str)
        if event_date < x_start or event_date > x_end:
            continue
        color = CAT_COLORS.get(category, "#666666")
        ax.axvline(event_date, color=color, linewidth=1.4, alpha=0.85)
        ax.text(
            event_date,
            y_positions[idx % len(y_positions)],
            label,
            ha="center", va="center", fontsize=7.5, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, linewidth=0.5, alpha=0.95),
        )


def plot_reddit_monitoring_figure(panels: list[dict], out_path: Path) -> None:
    if not panels:
        raise ValueError("No panels available to plot.")

    phase2_start = min(panel["dates"].min() for panel in panels)
    phase2_end = max(panel["dates"].max() for panel in panels)

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8})

    n_panels = len(panels)
    fig_height = 1.5 + 2.35 * n_panels
    fig, axes = plt.subplots(
        n_panels + 1, 1,
        figsize=(14.5, fig_height),
        gridspec_kw={"height_ratios": [0.8] + [1.0] * n_panels, "hspace": 0.08},
        sharex=True,
    )

    ax_events = axes[0]
    draw_event_bar(ax_events, phase2_start, phase2_end)

    for start_str, end_str, color, _label in MAJOR_SPANS:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        for ax in axes[1:]:
            ax.axvspan(start, end, alpha=0.08, color=color, zorder=0)

    for date_str, _label, category in EVENTS:
        event_date = pd.Timestamp(date_str)
        if event_date < phase2_start or event_date > phase2_end:
            continue
        color = CAT_COLORS.get(category, "#666666")
        for ax in axes[1:]:
            ax.axvline(event_date, color=color, linestyle="--", linewidth=0.7, alpha=0.45, zorder=1)

    for idx, panel in enumerate(panels, start=1):
        ax = axes[idx]
        dates = panel["dates"]
        values = panel["values"]
        threshold = panel["threshold"]
        color = panel["color"]

        finite_mask = np.isfinite(values)
        alarm_mask = finite_mask & (values > threshold) if threshold is not None else np.zeros_like(finite_mask, dtype=bool)
        normal_mask = finite_mask & ~alarm_mask

        ax.set_facecolor("#FAFAFA")
        ax.grid(True, linestyle="-", linewidth=0.3, alpha=0.3, color="#CCCCCC")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if threshold is not None:
            ax.axhline(threshold, color="#B71C1C", linewidth=1.1, alpha=0.7, zorder=2)
            ax.text(dates.iloc[-1], threshold, " UCL", va="bottom", ha="right", fontsize=8, color="#B71C1C")

        ax.plot(dates, values, "-", color=color, linewidth=1.3, alpha=0.9, zorder=3)
        if normal_mask.any():
            ax.scatter(dates.iloc[np.flatnonzero(normal_mask)], values[normal_mask],
                       s=20, color=color, alpha=0.75, edgecolors="none", zorder=4)
        if alarm_mask.any():
            ax.scatter(dates.iloc[np.flatnonzero(alarm_mask)], values[alarm_mask],
                       s=48, color="#D32F2F", alpha=0.92, edgecolors="#B71C1C", linewidths=0.8, zorder=5)

        ax.text(0.01, 0.92, panel["method"], transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="#333333", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#CCCCCC", alpha=0.92))
        ax.set_ylabel(panel["label"], fontsize=10)
        ax.tick_params(axis="y", labelsize=9)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("Date (2021)", fontsize=12)
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D32F2F", markeredgecolor="#B71C1C", markersize=8, label="Alarm"),
        Line2D([0], [0], color="#B71C1C", linewidth=1.1, label="UCL"),
        Line2D([0], [0], color=MAJOR_SPANS[0][2], linewidth=6, alpha=0.18, label=MAJOR_SPANS[0][3]),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure to {out_path}")


def main() -> None:
    args = parse_args()
    results_base = args.results_base.resolve()
    if not results_base.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_base}")

    meta = load_json(results_base / "meta.json")
    tag = str(meta.get("tag", results_base.name))

    phase2_start_raw = args.phase2_start or meta.get("cutoff_date") or "2021-02-27"
    phase2_start = pd.Timestamp(phase2_start_raw)
    if phase2_start.tzinfo is not None:
        phase2_start = phase2_start.tz_localize(None)

    summary_df = load_summary(results_base)
    sample_dates = load_sample_dates(results_base)

    panel_specs = list(BASELINE_PANEL_SPECS)
    if not args.exclude_lot:
        panel_specs = list(LOT_PANEL_SPECS) + panel_specs

    panels = []
    for spec in panel_specs:
        panel = build_panel(
            spec,
            results_base=results_base,
            tag=tag,
            summary_df=summary_df,
            sample_dates=sample_dates,
            phase2_start=phase2_start,
        )
        if panel is not None:
            panels.append(panel)

    if not panels:
        raise RuntimeError("No panels were loaded. Check the results path and saved CSV files.")

    plot_reddit_monitoring_figure(panels, args.out.resolve())


if __name__ == "__main__":
    main()
