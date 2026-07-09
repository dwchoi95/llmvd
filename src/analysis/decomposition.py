"""Bias-vs-discrimination decomposition for LLM vulnerability detection.

Runs on the EXISTING result JSONLs (binary `predict` + `vulnerable` label) and
separates each (model x scope x complexity-tier) cell into

  * a DISCRIMINATION component  -- how well the model ranks vuln vs non-vuln,
    reported threshold-free via signal-detection d' and an implied AUC, plus
    the base-rate-robust MCC / balanced accuracy;
  * a RESPONSE-BIAS component    -- the model's tendency to answer "vulnerable",
    reported as the positive-prediction rate (PPR) and the SDT criterion c.

The central claim of the study is that changing input *scope* (and later,
prompting strategy) moves the BIAS component (c / PPR) while leaving the
DISCRIMINATION component (d' / implied-AUC / MCC) statistically flat and near
trivial always-positive / always-negative baselines. This module quantifies
exactly that split so the claim is checkable rather than asserted.

Note: d' and implied-AUC here are derived from a single binary operating point
under an equal-variance Gaussian SDT model -- they are a principled *threshold-
free-ish* summary obtainable without token probabilities. The empirical
multi-threshold ROC/AUC (which needs captured logprobs) is a strict refinement
added by the score-capture pipeline; the two are reported side by side once
scores exist.

Usage:
    ./env/bin/python -m src.analysis.decomposition \
        --models "llama3.1:8b,mistral-nemo:12b,phi3:14b,qwen3-coder:30b" \
        --benchmark FuncFileRepo.eval --trials 1 --outdir results/analysis/decomposition
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

SCOPES = ["function", "file", "repository"]
TIERS = ["SFSF", "MFSF", "MFMF"]
# fields we actually need -- keep memory light while streaming 300MB files
_KEEP = ("vulnerable", "predict", "scope", "tag", "group_id", "language")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _to_bool(val):
    """Mirror Detector.convert_to_bool: coerce predict to True/False/None."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val in (1, 1.0):
            return True
        if val in (0, 0.0):
            return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0"):
            return False
    return None


def stream_records(path: str) -> list[dict]:
    """Stream a result JSONL, keeping only the small set of fields we need."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pred = _to_bool(r.get("predict"))
            lab = r.get("vulnerable")
            if pred is None or lab is None:
                continue  # unscored / unlabeled row -> excluded, counted separately
            sc = r.get("score")
            out.append(
                {
                    "y": bool(lab),
                    "p": pred,
                    "score": float(sc) if isinstance(sc, (int, float)) else None,
                    "scope": r.get("scope"),
                    "strategy": r.get("strategy"),
                    "tag": r.get("tag"),
                    "group_id": r.get("group_id"),
                    "language": r.get("language"),
                }
            )
    return out


def trial_paths(results_dir: str, model: str, benchmark: str, trials: str) -> list[str]:
    base = Path(results_dir) / model
    if trials == "all":
        # every {benchmark}_{k}.jsonl, else the un-suffixed file
        cands = sorted(base.glob(f"{benchmark}_*.jsonl"))
        cands = [c for c in cands if c.stem.split("_")[-1].isdigit()]
        if cands:
            return [str(c) for c in cands]
        single = base / f"{benchmark}.jsonl"
        return [str(single)] if single.exists() else []
    # explicit trial numbers "1" or "1-10" or "1,3,5"
    nums: list[int] = []
    for part in trials.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            nums.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            nums.append(int(part))
    paths = []
    for n in nums:
        p = base / f"{benchmark}_{n}.jsonl"
        if p.exists():
            paths.append(str(p))
    if not paths:  # fall back to un-suffixed single-trial file
        single = base / f"{benchmark}.jsonl"
        if single.exists():
            paths = [str(single)]
    return paths


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _rate_correct(hits: int, n: int) -> float:
    """Log-linear (Hautus) correction so z-scores stay finite at 0/1 rates."""
    return (hits + 0.5) / (n + 1.0)


def sdt(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Signal-detection decomposition from a single binary operating point.

    d'  = z(TPR) - z(FPR)      -> discrimination (threshold-free under SDT)
    c   = -0.5*(z(TPR)+z(FPR)) -> criterion / response bias
          (c<0 = liberal / over-predicts 'vulnerable'; c>0 = conservative)
    """
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()
    n_pos = tp + fn
    n_neg = fp + tn
    if n_pos == 0 or n_neg == 0:
        return {"dprime": np.nan, "criterion": np.nan, "auc_implied": np.nan,
                "tpr": np.nan, "fpr": np.nan}
    tpr = _rate_correct(tp, n_pos)
    fpr = _rate_correct(fp, n_neg)
    z_tpr, z_fpr = norm.ppf(tpr), norm.ppf(fpr)
    dprime = z_tpr - z_fpr
    crit = -0.5 * (z_tpr + z_fpr)
    # implied AUC under equal-variance Gaussian SDT: Phi(d'/sqrt2)
    auc_implied = float(norm.cdf(dprime / np.sqrt(2.0)))
    return {
        "dprime": float(dprime),
        "criterion": float(crit),
        "auc_implied": auc_implied,
        "tpr": tp / n_pos,     # raw (uncorrected) rates for reporting
        "fpr": fp / n_neg,
    }


def binary_metrics(recs: list[dict]) -> dict:
    """Full metric row for a list of records (one cell)."""
    if not recs:
        return {}
    y = np.array([r["y"] for r in recs], dtype=bool)
    p = np.array([r["p"] for r in recs], dtype=bool)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[False, True]).ravel()
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ppr = (tp + fp) / n  # positive-prediction rate == mean(predict)
    mcc = matthews_corrcoef(y, p) if (n_pos and n_neg) else 0.0
    bal = balanced_accuracy_score(y, p) if (n_pos and n_neg) else 0.5
    row = {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "prevalence": n_pos / n if n else np.nan,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "mcc": mcc, "balanced_acc": bal, "ppr": ppr,
    }
    row.update(sdt(y, p))
    # empirical threshold-free AUC from captured P(vulnerable) scores, when
    # available -- the strict refinement of the SDT-implied AUC above.
    sc = np.array([r.get("score") for r in recs], dtype=object)
    have = np.array([isinstance(v, (int, float)) and v is not None for v in sc])
    if have.sum() >= 2 and n_pos and n_neg:
        ys = y[have]
        ss = sc[have].astype(float)
        if len(set(ys.tolist())) > 1:
            row["auc_score"] = float(roc_auc_score(ys, ss))
            row["n_scored"] = int(have.sum())
    return row


def baseline_rows(prevalence: float, n: int, rng_seed: int = 0) -> list[dict]:
    """Trivial baselines evaluated at the cell's prevalence."""
    out = []
    # always-positive
    out.append({"row": "always-positive", "accuracy": prevalence,
                "precision": prevalence, "recall": 1.0,
                "f1": 2 * prevalence / (prevalence + 1) if prevalence else 0.0,
                "mcc": 0.0, "balanced_acc": 0.5, "ppr": 1.0,
                "dprime": 0.0, "criterion": np.nan, "auc_implied": 0.5})
    # always-negative
    out.append({"row": "always-negative", "accuracy": 1 - prevalence,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mcc": 0.0, "balanced_acc": 0.5, "ppr": 0.0,
                "dprime": 0.0, "criterion": np.nan, "auc_implied": 0.5})
    return out


# --------------------------------------------------------------------------- #
# aggregation over trials
# --------------------------------------------------------------------------- #
_METRIC_KEYS = ["n", "n_pos", "n_neg", "prevalence", "accuracy", "precision",
                "recall", "f1", "mcc", "balanced_acc", "ppr", "dprime",
                "criterion", "auc_implied", "auc_score", "tpr", "fpr"]


def _mean_std(dicts: list[dict]) -> dict:
    """Mean +/- std across trials for each metric key."""
    out = {}
    for k in _METRIC_KEYS:
        vals = np.array([d[k] for d in dicts if k in d and d[k] is not None],
                        dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(np.mean(vals)) if len(vals) else np.nan
        out[k + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    return out


def cell_over_trials(trials_recs: list[list[dict]], selector) -> dict:
    """Compute per-trial metrics for a filtered cell, then mean+/-std."""
    per_trial = []
    for recs in trials_recs:
        sub = [r for r in recs if selector(r)]
        if sub:
            per_trial.append(binary_metrics(sub))
    if not per_trial:
        return {}
    return _mean_std(per_trial)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _strategies_in(trials_recs) -> list:
    ss = set()
    for t in trials_recs:
        for r in t:
            if r.get("strategy"):
                ss.add(r["strategy"])
    return sorted(ss) if ss else [None]


def analyze_model(results_dir, model, benchmark, trials) -> dict | None:
    paths = trial_paths(results_dir, model, benchmark, trials)
    if not paths:
        print(f"[skip] {model}: no result files for benchmark={benchmark} trials={trials}")
        return None
    trials_recs = [stream_records(p) for p in paths]
    n_scored = sum(len(t) for t in trials_recs)
    strategies = _strategies_in(trials_recs)
    print(f"[load] {model}: {len(paths)} trial(s), {n_scored} scored rows, "
          f"strategies={[s or 'na' for s in strategies]}")

    def match(r, scope, strat, tier=None):
        if r["scope"] != scope:
            return False
        if strat is not None and r.get("strategy") != strat:
            return False
        if tier is not None and r.get("tag") != tier:
            return False
        return True

    rows_scope, rows_tier = [], []
    for strat in strategies:
        for scope in SCOPES:
            m = cell_over_trials(trials_recs,
                                 lambda r, s=scope, st=strat: match(r, s, st))
            if m:
                rows_scope.append({"model": model, "strategy": strat or "na",
                                   "scope": scope, **m})
            for tier in TIERS:
                mt = cell_over_trials(
                    trials_recs,
                    lambda r, s=scope, st=strat, t=tier: match(r, s, st, t))
                if mt:
                    rows_tier.append({"model": model, "strategy": strat or "na",
                                      "scope": scope, "tier": tier, **mt})
    return {"scope": rows_scope, "tier": rows_tier, "trials_recs": trials_recs}


def pooled_scope(all_trials_recs: dict) -> list[dict]:
    """Pooled-over-models per (strategy, scope) (headline table). Reported
    ALONGSIDE per-model rows -- pooling alone is called out as incoherent when
    models sit on opposite sides of ROC space, so per-model is the primary view."""
    strategies = set()
    for trecs in all_trials_recs.values():
        for t in trecs:
            for r in t:
                if r.get("strategy"):
                    strategies.add(r["strategy"])
    strategies = sorted(strategies) if strategies else [None]
    rows = []
    for strat in strategies:
        for scope in SCOPES:
            recs = []
            for model, trecs in all_trials_recs.items():
                for t in trecs:
                    recs.extend([r for r in t if r["scope"] == scope
                                 and (strat is None or r.get("strategy") == strat)])
            if recs:
                rows.append({"strategy": strat or "na", "scope": scope,
                             **binary_metrics(recs)})
    return rows


def decomposition_summary(scope_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model: how much does discrimination move vs bias across scopes?

    The thesis predicts spread(discrimination) ~ 0 while spread(bias) > 0.
    """
    out = []
    group_cols = ["model", "strategy"] if "strategy" in scope_df.columns else ["model"]
    for key, g in scope_df.groupby(group_cols):
        g = g.set_index("scope")
        model = key[0] if isinstance(key, tuple) else key
        strat = key[1] if isinstance(key, tuple) and len(key) > 1 else "na"
        def rng(col):
            v = g[col].dropna()
            return (float(v.max() - v.min()) if len(v) else np.nan)
        out.append({
            "model": model, "strategy": strat,
            "dprime_range": rng("dprime"),
            "mcc_range": rng("mcc"),
            "auc_implied_range": rng("auc_implied"),
            "criterion_range": rng("criterion"),
            "ppr_range": rng("ppr"),
            "dprime_by_scope": {s: round(float(g.loc[s, "dprime"]), 3)
                                for s in g.index},
            "ppr_by_scope": {s: round(float(g.loc[s, "ppr"]), 3)
                             for s in g.index},
        })
    return pd.DataFrame(out)


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    show = df.copy()
    for c in cols:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not pd.isna(x) else x)
    keep = [c for c in (["model", "strategy", "scope", "tier"] + cols) if c in show.columns]
    return show[keep].to_string(index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--models", required=True,
                    help="comma-separated model dirs, e.g. 'llama3.1:8b,phi3:14b'")
    ap.add_argument("--benchmark", default="FuncFileRepo.eval")
    ap.add_argument("--trials", default="1", help="'1', '1-10', '1,3,5', or 'all'")
    ap.add_argument("--outdir", default="results/analysis/decomposition")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scope_rows, tier_rows = [], []
    all_trials_recs = {}
    for model in models:
        res = analyze_model(args.results_dir, model, args.benchmark, args.trials)
        if not res:
            continue
        scope_rows.extend(res["scope"])
        tier_rows.extend(res["tier"])
        all_trials_recs[model] = res["trials_recs"]

    if not scope_rows:
        print("No data analyzed. Check --models/--benchmark/--trials.")
        return

    scope_df = pd.DataFrame(scope_rows)
    tier_df = pd.DataFrame(tier_rows)
    pooled_df = pd.DataFrame(pooled_scope(all_trials_recs))
    summary_df = decomposition_summary(scope_df)

    scope_df.to_csv(outdir / "per_model_scope.csv", index=False)
    tier_df.to_csv(outdir / "per_model_scope_tier.csv", index=False)
    pooled_df.to_csv(outdir / "pooled_scope.csv", index=False)
    summary_df.to_csv(outdir / "decomposition_summary.csv", index=False)

    disc = ["mcc", "balanced_acc", "auc_implied", "auc_score", "dprime"]
    bias = ["ppr", "criterion"]
    comp = ["accuracy", "precision", "recall", "f1"]

    print("\n" + "=" * 78)
    print("POOLED over models  (headline; reported WITH per-model to avoid pooling artifact)")
    print("=" * 78)
    print("  DISCRIMINATION:", _fmt(pooled_df, disc))
    print("\n  RESPONSE BIAS: ", _fmt(pooled_df, bias))
    print("\n  COMPARABILITY (VulnSage-style): ", _fmt(pooled_df, comp))

    print("\n" + "=" * 78)
    print("PER-MODEL x SCOPE")
    print("=" * 78)
    print(_fmt(scope_df, disc + bias))

    print("\n" + "=" * 78)
    print("DECOMPOSITION SUMMARY  (does discrimination move, or only bias?)")
    print("=" * 78)
    for _, r in summary_df.iterrows():
        strat = f" | {r['strategy']}" if 'strategy' in r else ""
        print(f"\n  {r['model']}{strat}")
        print(f"    d' by scope        : {r['dprime_by_scope']}   (range {r['dprime_range']:.3f})")
        print(f"    PPR by scope       : {r['ppr_by_scope']}   (range {r['ppr_range']:.3f})")
        print(f"    MCC range across scope   : {r['mcc_range']:.3f}")
        print(f"    criterion range across   : {r['criterion_range']:.3f}")

    # verdict line
    print("\n" + "=" * 78)
    md = summary_df["dprime_range"].mean()
    mc = summary_df["criterion_range"].mean()
    mppr = summary_df["ppr_range"].mean()
    print(f"MEAN across models:  discrimination spread (d' range) = {md:.3f} | "
          f"bias spread (criterion range) = {mc:.3f} | PPR range = {mppr:.3f}")
    print("Thesis pattern = small discrimination spread WITH larger bias spread.")
    print("=" * 78)
    print(f"\nSaved CSVs -> {outdir}/")


if __name__ == "__main__":
    main()
