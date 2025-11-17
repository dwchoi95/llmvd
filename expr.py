#!/usr/bin/env python3
"""
Comprehensive analysis pipeline for Func/File/Repo vulnerability detection runs.

The script loads every JSONL result under `results/`, harmonizes model/trial metadata,
and produces the ten deliverables requested by the research notebook:
    1. Accuracy table with mean±std and relative improvements.
    2. Scope-wise efficiency violin plot (time + tokens).
    3. Accuracy/efficiency trade-off bubble chart.
    4. Confusion-matrix style error table with FP/FN rates.
    5. CWE Venn diagram for majority-success scopes.
    6. Top-10 CWE F1 table + ANOVA & Tukey post-hoc.
    7. Model-by-scope F1/time table with improvement column.
    8. Pairwise agreement heatmaps (function/file/repo/overall).
    9. Language-by-scope F1/time table.
    10.Trial-wise F1 variability line chart.

Outputs are stored under `figures/` (plots) and `figures/tables/` (csv/txt summaries).
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn3
from scipy import stats as scipy_stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

RESULTS_DIR = Path("results")
DATASET_PATH = Path("data/FuncFileRepo.jsonl")
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")
SCOPE_ORDER = ["function", "file", "repository"]


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_binary(value) -> Optional[int]:
    """Best-effort conversion to 0/1 ints; returns None for invalid inputs."""
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return int(value)
        return None
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if value in (0.0, 1.0):
            return int(value)
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "t", "yes", "y"}:
            return 1
        if cleaned in {"0", "false", "f", "no", "n"}:
            return 0
    return None


def _normalize_scope(scope: str) -> str:
    if scope is None:
        return "unknown"
    return str(scope).strip().lower()


def _normalize_language(lang: str) -> str:
    if not isinstance(lang, str):
        return "unknown"
    return lang.strip().lower()


def _normalize_cwe_list(raw) -> Tuple[str, ...]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return tuple()
    if isinstance(raw, str):
        return (raw.strip(),)
    if isinstance(raw, (list, tuple, set)):
        cleaned = []
        for item in raw:
            if item is None:
                continue
            cleaned.append(str(item).strip())
        return tuple(cleaned)
    return tuple()


def _parse_result_filename(path: Path) -> Tuple[str, str, Optional[int]]:
    """Return dataset, model, trial (if numeric) extracted from filename stem."""
    stem = path.stem
    if "_" not in stem:
        return stem, stem, None
    dataset, remainder = stem.split("_", 1)
    model = remainder
    trial: Optional[int] = None
    if "_" in remainder:
        candidate_model, maybe_trial = remainder.rsplit("_", 1)
        if maybe_trial.isdigit():
            model = candidate_model
            trial = int(maybe_trial)
    return dataset, model, trial


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def classification_report(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Return core metrics + confusion counts given binary arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    total = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": total,
    }


def _format_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.3f} ± {std:.3f}"


def _format_percentage(value: float) -> str:
    return f"{value:+.1f}%"


def _render_table(df: pd.DataFrame, title: str, out_txt: Path) -> None:
    """Pretty-print a dataframe to the console and save the fixed-width text."""
    column_widths = {col: max(len(str(col)), df[col].astype(str).map(len).max()) for col in df.columns}

    def fmt_row(row_vals: Iterable[str]) -> str:
        return " | ".join(str(val).ljust(column_widths[col]) for val, col in zip(row_vals, df.columns))

    header = fmt_row(df.columns)
    divider = "-+-".join("-" * column_widths[col] for col in df.columns)
    lines = [title, header, divider]
    for _, row in df.iterrows():
        lines.append(fmt_row(row.tolist()))
    text = "\n".join(lines)
    print("\n" + text + "\n")
    out_txt.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------------------------

@dataclass
class Analyzer:
    results_dir: Path = RESULTS_DIR
    dataset_path: Path = DATASET_PATH

    def __post_init__(self) -> None:
        _ensure_dirs()
        self.raw_df = self._load_results()
        self.dataset_df = self._load_dataset()
        self.combo_count = (
            self.raw_df.loc[self.raw_df["trial_id"].notna(), ["model", "trial_id"]]
            .drop_duplicates()
            .shape[0]
        )
        if self.combo_count == 0:
            self.combo_count = self.raw_df["model"].nunique()
        self.scope_metrics_cache: Optional[pd.DataFrame] = None
        self.scope_metric_per_combo_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_results(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for path in sorted(self.results_dir.glob("*.jsonl")):
            dataset, model, trial = _parse_result_filename(path)
            df = pd.read_json(path, lines=True)
            df["dataset"] = dataset
            df["model"] = model
            df["trial_id"] = trial
            df["result_file"] = path.name
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No JSONL result files found under {self.results_dir}")
        df = pd.concat(frames, ignore_index=True)
        df["scope"] = df["scope"].map(_normalize_scope)
        df = df[df["scope"].isin(SCOPE_ORDER)].copy()

        df["y_true"] = df["vulnerable"].map(_normalize_binary)
        df["y_pred"] = df["predict"].map(_normalize_binary)
        df = df.dropna(subset=["y_true", "y_pred"])
        df["y_true"] = df["y_true"].astype(int)
        df["y_pred"] = df["y_pred"].astype(int)

        df["language"] = df["language"].map(_normalize_language)
        df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")
        df["time_sec"] = pd.to_numeric(df["Time (sec)"], errors="coerce")
        df["cwe_list"] = df["cwe_id"].map(_normalize_cwe_list)
        df["combo_key"] = df["model"] + "_trial_" + df["trial_id"].fillna(-1).astype(int).astype(str)

        return df.reset_index(drop=True)

    def _load_dataset(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"{self.dataset_path} is missing")
        df = pd.read_json(self.dataset_path, lines=True)
        df["cwe_list"] = df["cwe_id"].map(_normalize_cwe_list)
        return df

    # ------------------------------------------------------------------
    # Metrics caches
    # ------------------------------------------------------------------
    def metrics_per_combo_scope(self) -> pd.DataFrame:
        if self.scope_metric_per_combo_cache is not None:
            return self.scope_metric_per_combo_cache
        records = []
        for (model, trial_id, scope), group in self.raw_df.groupby(["model", "trial_id", "scope"], dropna=False):
            metrics = classification_report(group["y_true"], group["y_pred"])
            metrics.update({"model": model, "trial_id": trial_id, "scope": scope})
            metrics["time_mean"] = group["time_sec"].mean()
            metrics["tokens_mean"] = group["tokens"].mean()
            records.append(metrics)
        df = pd.DataFrame(records)
        self.scope_metric_per_combo_cache = df
        return df

    def metrics_per_scope(self) -> pd.DataFrame:
        if self.scope_metrics_cache is not None:
            return self.scope_metrics_cache
        per_combo = self.metrics_per_combo_scope()
        agg_rows = []
        for scope, group in per_combo.groupby("scope"):
            row = {"scope": scope, "samples": len(group)}
            for metric in ["accuracy", "precision", "recall", "f1"]:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=0)
            row["time_mean"] = group["time_mean"].mean()
            row["tokens_mean"] = group["tokens_mean"].mean()
            agg_rows.append(row)
        df = pd.DataFrame(agg_rows).set_index("scope").loc[SCOPE_ORDER].reset_index()
        self.scope_metrics_cache = df
        return df

    # ------------------------------------------------------------------
    # Requirement 1 - correctness table
    # ------------------------------------------------------------------
    def requirement_1(self) -> None:
        scope_df = self.metrics_per_scope()
        table_rows = []
        for _, row in scope_df.iterrows():
            table_rows.append(
                {
                    "Granularity": row["scope"].title(),
                    "Accuracy": _format_mean_std(row["accuracy_mean"], row["accuracy_std"]),
                    "Precision": _format_mean_std(row["precision_mean"], row["precision_std"]),
                    "Recall": _format_mean_std(row["recall_mean"], row["recall_std"]),
                    "F1-Score": _format_mean_std(row["f1_mean"], row["f1_std"]),
                }
            )
        improvements = []
        pairs = [("function", "file"), ("file", "repository"), ("repository", "function")]
        for a, b in pairs:
            row_a = scope_df.set_index("scope").loc[a]
            row_b = scope_df.set_index("scope").loc[b]
            improvements.append(
                {
                    "Granularity": f"{a.title()} vs {b.title()}",
                    "Accuracy": _format_percentage(100 * _safe_div(row_b["accuracy_mean"] - row_a["accuracy_mean"], row_a["accuracy_mean"])),
                    "Precision": _format_percentage(100 * _safe_div(row_b["precision_mean"] - row_a["precision_mean"], row_a["precision_mean"])),
                    "Recall": _format_percentage(100 * _safe_div(row_b["recall_mean"] - row_a["recall_mean"], row_a["recall_mean"])),
                    "F1-Score": _format_percentage(100 * _safe_div(row_b["f1_mean"] - row_a["f1_mean"], row_a["f1_mean"])),
                }
            )
        table_df = pd.DataFrame(table_rows + improvements)
        out_txt = TABLES_DIR / "scope_correctness_table.txt"
        table_df.to_csv(TABLES_DIR / "scope_correctness_table.csv", index=False)
        _render_table(table_df, "Scope-level correctness (mean ± std)", out_txt)

    # ------------------------------------------------------------------
    # Requirement 2 - efficiency violin
    # ------------------------------------------------------------------
    def requirement_2(self) -> None:
        df = self.raw_df.dropna(subset=["time_sec", "tokens"])
        df = df[df["scope"].isin(SCOPE_ORDER)]
        scope_labels = [s.title() for s in SCOPE_ORDER]
        positions = np.arange(len(SCOPE_ORDER))
        time_data = [df.loc[df["scope"] == scope, "time_sec"].values for scope in SCOPE_ORDER]
        token_data = [df.loc[df["scope"] == scope, "tokens"].values for scope in SCOPE_ORDER]

        def _trim(values: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
            if values.size == 0:
                return values
            low = np.quantile(values, lower)
            high = np.quantile(values, upper)
            trimmed = values[(values >= low) & (values <= high)]
            return trimmed if trimmed.size else values

        time_data = [_trim(arr) for arr in time_data]
        token_data = [_trim(arr) for arr in token_data]

        fig, ax_time = plt.subplots(figsize=(9, 5))
        ax_tokens = ax_time.twinx()

        time_v = ax_time.violinplot(time_data, positions=positions, widths=0.6, showmeans=True, showextrema=False)
        token_v = ax_tokens.violinplot(token_data, positions=positions, widths=0.6, showmeans=True, showextrema=False)

        def _clip_halves(violin, centers, direction: str):
            for body, center in zip(violin["bodies"], centers):
                verts = body.get_paths()[0].vertices
                if direction == "left":
                    verts[:, 0] = np.minimum(verts[:, 0], center)
                else:
                    verts[:, 0] = np.maximum(verts[:, 0], center)
                body.get_paths()[0].vertices = verts

        _clip_halves(time_v, positions, "left")
        _clip_halves(token_v, positions, "right")

        for body in time_v["bodies"]:
            body.set_facecolor("#1f77b4")
            body.set_alpha(0.35)
        for body in token_v["bodies"]:
            body.set_facecolor("#ff7f0e")
            body.set_alpha(0.35)

        ax_time.set_ylabel("Response Time (sec)", color="#1f77b4")
        ax_tokens.set_ylabel("Tokens", color="#ff7f0e")
        ax_time.set_xticks(positions)
        ax_time.set_xticklabels(scope_labels)
        ax_time.tick_params(axis="y", labelcolor="#1f77b4")
        ax_tokens.tick_params(axis="y", labelcolor="#ff7f0e")
        ax_time.set_xlabel("Scope")
        # ax_time.set_title("Scope-wise efficiency distribution (split violins)")
        ax_time.grid(False)
        ax_tokens.grid(False)

        legend_patches = [
            plt.Line2D([0], [0], color="#1f77b4", lw=8, alpha=0.35, label="Time (sec)"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=8, alpha=0.35, label="Tokens"),
        ]
        ax_time.legend(handles=legend_patches, loc="upper left", frameon=True, fontsize=10)

        fig.tight_layout()
        out_path = FIGURES_DIR / "scope_efficiency_violin.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Requirement 3 - trade-off bubble chart
    # ------------------------------------------------------------------
    def requirement_3(self) -> None:
        scope_df = self.metrics_per_scope()
        fig, ax = plt.subplots(figsize=(9, 6))
        palette = {"function": "#5DADE2", "file": "#EC7063", "repository": "#58D68D"}

        def _size_from_tokens(tokens: float) -> float:
            if pd.isna(tokens):
                return 0
            size_scale = 0.08  # tuned for aesthetics
            return np.clip(tokens * size_scale, 200, 2000)

        for _, row in scope_df.iterrows():
            size = _size_from_tokens(row["tokens_mean"])
            color = palette.get(row["scope"], "#7f7f7f")
            ax.scatter(
                row["time_mean"],
                row["f1_mean"],
                s=size,
                c=color,
                edgecolor="black",
                linewidth=1.2,
                alpha=0.9,
            )
            ax.annotate(
                row["scope"].title(),
                (row["time_mean"], row["f1_mean"]),
                xytext=(12, 10),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="0.5", lw=1.0),
            )

        ax.set_xlabel("Avg Time (sec)", fontsize=12)
        ax.set_ylabel("Avg F1-Score", fontsize=12)
        ax.grid(True, linestyle=":", linewidth=1, alpha=0.5)
        ax.tick_params(labelsize=11)

        x_margin = (scope_df["time_mean"].max() - scope_df["time_mean"].min()) * 0.1 if len(scope_df) > 1 else 1
        y_margin = (scope_df["f1_mean"].max() - scope_df["f1_mean"].min()) * 0.1 if len(scope_df) > 1 else 0.05
        ax.set_xlim(scope_df["time_mean"].min() - x_margin, scope_df["time_mean"].max() + x_margin)
        ax.set_ylim(scope_df["f1_mean"].min() - y_margin, scope_df["f1_mean"].max() + y_margin)

        legend_rows = scope_df.dropna(subset=["tokens_mean"]).sort_values("tokens_mean")
        legend_handles = []
        legend_labels = []
        for _, row in legend_rows.iterrows():
            tokens = row["tokens_mean"]
            size = _size_from_tokens(tokens)
            handle = ax.scatter([], [], s=size, edgecolors="black", facecolors=palette.get(row["scope"], "#7f7f7f"), linewidth=1.2)
            legend_handles.append(handle)
            token_label = f"{tokens/1000:.1f}K" if tokens >= 1000 else f"{tokens:.0f}"
            legend_labels.append(f"{token_label}")
        if legend_handles:
            ax.legend(
                legend_handles,
                legend_labels,
                title="Avg tokens per sample",
                loc="best",
                frameon=True,
                labelspacing=1.0,
                handleheight=2.0,
                borderpad=1.1,
                handletextpad=1.2,
                fontsize=10,
                title_fontsize=11,
            )
        fig.tight_layout()
        out_path = FIGURES_DIR / "scope_tradeoff_bubble.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Requirement 4 - confusion matrix table
    # ------------------------------------------------------------------
    def requirement_4(self) -> None:
        rows = []
        for scope, group in self.raw_df.groupby("scope"):
            metrics = classification_report(group["y_true"], group["y_pred"])
            fp_rate = _safe_div(metrics["fp"], metrics["fp"] + metrics["tn"])
            fn_rate = _safe_div(metrics["fn"], metrics["fn"] + metrics["tp"])
            rows.append(
                {
                    "Granularity": scope.title(),
                    "TP": metrics["tp"],
                    "FP": metrics["fp"],
                    "FN": metrics["fn"],
                    "TN": metrics["tn"],
                    "FP Rate": f"{fp_rate:.1%}",
                    "FN Rate": f"{fn_rate:.1%}",
                }
            )
        table_df = pd.DataFrame(rows)
        out_txt = TABLES_DIR / "scope_confusion_table.txt"
        table_df.to_csv(TABLES_DIR / "scope_confusion_table.csv", index=False)
        _render_table(table_df, "Confusion matrix counts by granularity", out_txt)

    # ------------------------------------------------------------------
    # Requirement 5 - CWE Venn
    # ------------------------------------------------------------------
    def _top_cwe_sets(self, top_n: int = 10) -> Dict[str, set]:
        exploded = self.raw_df.explode("cwe_list").dropna(subset=["cwe_list"])
        exploded = exploded[exploded["cwe_list"] != "NVD-CWE-noinfo"]
        per_combo = []
        for (model, trial_id, scope, cwe), group in exploded.groupby(
            ["model", "trial_id", "scope", "cwe_list"]
        ):
            metrics = classification_report(group["y_true"], group["y_pred"])
            per_combo.append(
                {
                    "model": model,
                    "trial_id": trial_id,
                    "scope": scope,
                    "cwe": cwe,
                    "f1": metrics["f1"],
                }
            )
        combo_df = pd.DataFrame(per_combo)
        scope_sets: Dict[str, set] = {scope: set() for scope in SCOPE_ORDER}
        for scope, scope_df in combo_df.groupby("scope"):
            top_cwes = (
                scope_df.groupby("cwe")["f1"].mean().sort_values(ascending=False).head(top_n).index
            )
            scope_sets[scope] = set(top_cwes)
        return scope_sets

    def requirement_5(self) -> None:
        scope_sets = self._top_cwe_sets(top_n=10)
        fig, ax = plt.subplots(figsize=(7, 7))
        v = venn3(
            subsets=[
                scope_sets["function"],
                scope_sets["file"],
                scope_sets["repository"],
            ],
            set_labels=("Function", "File", "Repository"),
            ax=ax,
        )

        def _label_text(cwes: set) -> str:
            if not cwes:
                return ""
            return "\n".join(sorted(cwes))

        regions = {
            "100": scope_sets["function"] - scope_sets["file"] - scope_sets["repository"],
            "010": scope_sets["file"] - scope_sets["function"] - scope_sets["repository"],
            "001": scope_sets["repository"] - scope_sets["function"] - scope_sets["file"],
            "110": (scope_sets["function"] & scope_sets["file"]) - scope_sets["repository"],
            "101": (scope_sets["function"] & scope_sets["repository"]) - scope_sets["file"],
            "011": (scope_sets["file"] & scope_sets["repository"]) - scope_sets["function"],
            "111": scope_sets["function"] & scope_sets["file"] & scope_sets["repository"],
        }

        labels = {
            "100": v.get_label_by_id("100"),
            "010": v.get_label_by_id("010"),
            "001": v.get_label_by_id("001"),
            "110": v.get_label_by_id("110"),
            "101": v.get_label_by_id("101"),
            "011": v.get_label_by_id("011"),
            "111": v.get_label_by_id("111"),
        }
        for key, label in labels.items():
            if label is not None:
                label.set_text(_label_text(regions[key]))
                label.set_fontsize(8)
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        out_path = FIGURES_DIR / "cwe_success_venn.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Requirement 6 - top-10 CWE table + stats
    # ------------------------------------------------------------------
    def _cwe_description(self, cwe: str) -> str:
        match = re.match(r"CWE-(\d+)", cwe)
        cwe_id = match.group(1)
        url = f"https://cwe.mitre.org/data/definitions/{cwe_id}.html"
        try:
            import requests

            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                title_match = re.search(rf"CWE-{cwe_id}:\s*([^<]+)", resp.text)
                if title_match:
                    description = html.unescape(title_match.group(1).strip())
                    return description
        except Exception:
            pass
        return cwe

    def requirement_6(self) -> None:
        base_counts = Counter()
        for cwes in self.dataset_df["cwe_list"]:
            for cwe in cwes:
                if cwe == "NVD-CWE-noinfo":
                    continue
                base_counts[cwe] += 1
        top10 = [c for c, _ in base_counts.most_common(10)]
        if not top10:
            return
        exploded = self.raw_df.explode("cwe_list").dropna(subset=["cwe_list"])
        exploded = exploded[exploded["cwe_list"] != "NVD-CWE-noinfo"]
        per_combo = []
        for (model, trial_id, scope, cwe), group in exploded.groupby(["model", "trial_id", "scope", "cwe_list"]):
            metrics = classification_report(group["y_true"], group["y_pred"])
            per_combo.append({"model": model, "trial_id": trial_id, "scope": scope, "cwe": cwe, "f1": metrics["f1"]})
        combo_df = pd.DataFrame(per_combo)
        rows = []
        stats_rows = []
        for cwe in top10:
            desc = self._cwe_description(cwe)
            name = desc.split("(")[0].strip() if "(" in desc else desc
            count = base_counts[cwe]
            row = {"CWE": cwe, "Name": name, "Count": count}
            cwe_records = combo_df[combo_df["cwe"] == cwe]
            for scope in SCOPE_ORDER:
                scope_vals = cwe_records.loc[cwe_records["scope"] == scope, "f1"]
                row[scope.title()] = scope_vals.mean() if not scope_vals.empty else np.nan
                stats_rows.extend([{"scope": scope.title(), "f1": val, "cwe": cwe} for val in scope_vals])
            rows.append(row)
        table_df = pd.DataFrame(rows)
        metric_cols = [scope.title() for scope in SCOPE_ORDER]
        table_df["Average"] = table_df[metric_cols].mean(axis=1)
        table_df["StdDev"] = table_df[metric_cols].std(axis=1, ddof=0)
        table_df = table_df[["CWE", "Name", "Count"] + metric_cols + ["Average", "StdDev"]]
        csv_df = table_df.copy()
        table_df.to_csv(TABLES_DIR / "top10_cwe_f1.csv", index=False)

        formatted_df = csv_df.copy()
        for col in metric_cols + ["Average", "StdDev"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "n/a")
        for idx, row in formatted_df.iterrows():
            values = [row[col] for col in metric_cols]
            numeric_vals = [float(val) if val != "n/a" else None for val in values]
            max_val = max([val for val in numeric_vals if val is not None], default=None)
            for col, num in zip(metric_cols, numeric_vals):
                if num is not None and num == max_val:
                    formatted_df.at[idx, col] = f"\\textbf{{{row[col]}}}"

        _render_table(formatted_df, "Top-10 CWE F1 by granularity", TABLES_DIR / "top10_cwe_f1.txt")

        stats_df = pd.DataFrame(stats_rows).dropna(subset=["f1"])
        if stats_df.empty:
            return
        grouped = [stats_df.loc[stats_df["scope"] == scope.title(), "f1"].values for scope in SCOPE_ORDER]
        anova_res = scipy_stats.f_oneway(*grouped)
        tukey = pairwise_tukeyhsd(endog=stats_df["f1"], groups=stats_df["scope"], alpha=0.05)
        stats_path = TABLES_DIR / "top10_cwe_stats.txt"
        stats_summary = [
            "One-way ANOVA on top-10 CWE F1 values",
            f"F-statistic: {anova_res.statistic:.3f}",
            f"p-value    : {anova_res.pvalue:.4f}",
            "",
            "Tukey HSD post-hoc (alpha=0.05):",
            str(tukey.summary()),
        ]
        stats_path.write_text("\n".join(stats_summary), encoding="utf-8")

    # ------------------------------------------------------------------
    # Requirement 7 - model table
    # ------------------------------------------------------------------
    def _model_family(self, model: str) -> str:
        model_lower = model.lower()
        proprietary_markers = ("gpt", "claude", "gemini")
        open_markers = ("llama", "phi", "mistral", "deepseek", "mixtral", "codellama")
        if any(marker in model_lower for marker in proprietary_markers):
            return "Proprietary/API"
        if any(marker in model_lower for marker in open_markers):
            return "Open-source/Local"
        return "Other"

    def requirement_7(self) -> None:
        per_combo = self.metrics_per_combo_scope()
        metrics = []
        for (model, scope), group in per_combo.groupby(["model", "scope"]):
            metrics.append(
                {
                    "Model": model,
                    "Scope": scope,
                    "f1_mean": group["f1"].mean(),
                    "time_mean": group["time_mean"].mean(),
                }
            )
        metric_df = pd.DataFrame(metrics)
        rows = []
        for model in sorted(metric_df["Model"].unique()):
            row = {"Group": self._model_family(model), "Model": model}
            model_df = metric_df[metric_df["Model"] == model]
            f1_vals = {}
            time_vals = {}
            for scope in SCOPE_ORDER:
                scope_df = model_df[model_df["Scope"] == scope]
                f1_vals[scope] = scope_df["f1_mean"].iloc[0] if not scope_df.empty else np.nan
                time_vals[scope] = scope_df["time_mean"].iloc[0] if not scope_df.empty else np.nan
                row[f"F1 ({scope.title()})"] = f"{f1_vals[scope]:.3f}" if not math.isnan(f1_vals[scope]) else "n/a"
                row[f"Time ({scope.title()})"] = f"{time_vals[scope]:.2f}" if not math.isnan(time_vals[scope]) else "n/a"
            if not math.isnan(f1_vals.get("function", np.nan)) and not math.isnan(f1_vals.get("repository", np.nan)):
                improvement = 100 * _safe_div(f1_vals["repository"] - f1_vals["function"], f1_vals["function"])
                row["Improvement"] = _format_percentage(improvement)
            else:
                row["Improvement"] = "n/a"
            rows.append(row)
        table_df = pd.DataFrame(rows)
        table_df = table_df.sort_values(by=["Group", "Model"])
        f1_cols = [f"F1 ({scope.title()})" for scope in SCOPE_ORDER]
        time_cols = [f"Time ({scope.title()})" for scope in SCOPE_ORDER]
        ordered_cols = ["Group", "Model"] + f1_cols + time_cols + ["Improvement"]
        table_df = table_df[ordered_cols]
        table_df.to_csv(TABLES_DIR / "model_scope_metrics.csv", index=False)
        _render_table(table_df, "Model-wise F1 & response time", TABLES_DIR / "model_scope_metrics.txt")

    # ------------------------------------------------------------------
    # Requirement 8 - agreement heatmaps
    # ------------------------------------------------------------------
    def _pairwise_agreement_matrix(self, scope: Optional[str]) -> pd.DataFrame:
        df = self.raw_df.copy()
        if scope:
            df = df[df["scope"] == scope]
        df = df.dropna(subset=["trial_id"])
        if df.empty:
            return pd.DataFrame()
        models = sorted(df["model"].unique())
        matrix = pd.DataFrame(np.eye(len(models)), index=models, columns=models, dtype=float)
        for i, model_a in enumerate(models):
            data_a = df[df["model"] == model_a][["trial_id", "index", "scope", "y_pred"]]
            for j, model_b in enumerate(models):
                if j <= i:
                    continue
                data_b = df[df["model"] == model_b][["trial_id", "index", "scope", "y_pred"]]
                merged = data_a.merge(data_b, on=["trial_id", "index", "scope"], suffixes=("_a", "_b"))
                if merged.empty:
                    agreement = np.nan
                else:
                    agreement = (merged["y_pred_a"] == merged["y_pred_b"]).mean()
                matrix.loc[model_a, model_b] = agreement
                matrix.loc[model_b, model_a] = agreement
        return matrix

    def requirement_8(self) -> None:
        matrices = {
            "Function": self._pairwise_agreement_matrix("function"),
            "File": self._pairwise_agreement_matrix("file"),
            "Repository": self._pairwise_agreement_matrix("repository"),
            "Overall": self._pairwise_agreement_matrix(None),
        }
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        titles = list(matrices.keys())
        for idx, title in enumerate(titles):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            matrix = matrices[title]
            if matrix.empty:
                ax.axis("off")
                # ax.set_title(f"{title}: insufficient overlapping trials")
                continue
            sns.heatmap(matrix, ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=False)
            # ax.set_title(f"{title} agreement")
        fig.tight_layout()
        out_path = FIGURES_DIR / "model_agreement_heatmaps.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Requirement 9 - language table
    # ------------------------------------------------------------------
    def requirement_9(self) -> None:
        lang_map = {"c": "c/c++", "c++": "c/c++"}
        df_lang = self.raw_df.copy()
        df_lang["language_group"] = df_lang["language"].map(lang_map).fillna(df_lang["language"])
        rows = []
        for (language, scope), group in df_lang.groupby(["language_group", "scope"]):
            metrics = classification_report(group["y_true"], group["y_pred"])
            rows.append(
                {
                    "language": language,
                    "scope": scope,
                    "f1": metrics["f1"],
                    "time": group["time_sec"].mean(),
                }
            )
        df = pd.DataFrame(rows)
        table_rows = []
        for language, group in df.groupby("language"):
            row = {"Language": language}
            f1_vals = {}
            for scope in SCOPE_ORDER:
                scope_df = group[group["scope"] == scope]
                f1_vals[scope] = scope_df["f1"].iloc[0] if not scope_df.empty else np.nan
                row[f"F1 ({scope.title()})"] = f"{f1_vals[scope]:.3f}" if not math.isnan(f1_vals[scope]) else "n/a"
                row[f"Time ({scope.title()})"] = (
                    f"{scope_df['time'].iloc[0]:.2f}" if not scope_df.empty and not math.isnan(scope_df["time"].iloc[0]) else "n/a"
                )
            if not math.isnan(f1_vals.get("function", np.nan)) and not math.isnan(f1_vals.get("repository", np.nan)):
                improvement = 100 * _safe_div(f1_vals["repository"] - f1_vals["function"], f1_vals["function"])
                row["Improvement"] = _format_percentage(improvement)
            else:
                row["Improvement"] = "n/a"
            table_rows.append(row)
        table_df = pd.DataFrame(table_rows).sort_values("Language")
        f1_cols = [f"F1 ({scope.title()})" for scope in SCOPE_ORDER]
        time_cols = [f"Time ({scope.title()})" for scope in SCOPE_ORDER]
        ordered_cols = ["Language"] + f1_cols + time_cols + ["Improvement"]
        table_df = table_df[ordered_cols]
        table_df.to_csv(TABLES_DIR / "language_scope_metrics.csv", index=False)
        _render_table(table_df, "Language-wise F1 & time", TABLES_DIR / "language_scope_metrics.txt")

    # ------------------------------------------------------------------
    # Requirement 10 - trial variability
    # ------------------------------------------------------------------
    def requirement_10(self) -> None:
        rows = []
        for (model, trial_id), group in self.raw_df.groupby(["model", "trial_id"]):
            if pd.isna(trial_id):
                continue
            metrics = classification_report(group["y_true"], group["y_pred"])
            rows.append({"model": model, "trial_id": int(trial_id), "f1": metrics["f1"]})
        df = pd.DataFrame(rows)
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, group in df.groupby("model"):
            group = group.sort_values("trial_id")
            ax.plot(group["trial_id"], group["f1"], linestyle="--", marker="o", label=model)
        ax.set_xlabel("Trial")
        ax.set_ylabel("F1-Score")
        # ax.set_title("F1 variability across 10 trials")
        ax.set_xticks(sorted(df["trial_id"].unique()))
        ax.legend(loc="best", fontsize="small", ncol=2)
        fig.tight_layout()
        out_path = FIGURES_DIR / "trial_variability.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # ------------------------------------------------------------------
    def run_all(self) -> None:
        self.requirement_1()
        self.requirement_2()
        self.requirement_3()
        self.requirement_4()
        self.requirement_5()
        self.requirement_6()
        self.requirement_7()
        self.requirement_8()
        self.requirement_9()
        self.requirement_10()


def main() -> None:
    analyzer = Analyzer()
    analyzer.run_all()


if __name__ == "__main__":
    main()
