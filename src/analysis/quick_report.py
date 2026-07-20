"""Quick interim report: differences by strategy (zero/rag), scope
(function/file/repository), and complexity tag (SFSF/MFSF/MFMF).

Separates DISCRIMINATION (threshold-free AUC on P(vulnerable); MCC) from
RESPONSE BIAS (PPR = positive-prediction rate) — the study's central split.
Metrics are computed per model, then aggregated as mean +/- std ACROSS MODELS
(models are the unit of analysis). Only fully-complete models are used for the
headline tables; partial models are listed separately.

  usage: python -m src.analysis.quick_report [results_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/final"
SCOPES = ["function", "file", "repository"]
STRATS = ["zero", "rag"]
TAGS = ["SFSF", "MFSF", "MFMF"]
N_SAMPLES = 960


def load_models(root):
    models = {}
    for d in sorted(glob.glob(f"{root}/*/")):
        label = os.path.basename(d.rstrip("/"))
        if label.endswith("-sft"):
            continue  # sft handled separately once trained
        rows = []
        for f in glob.glob(d + "*.jsonl"):
            rows += [json.loads(l) for l in open(f)]
        if rows:
            models[label] = rows
    return models


def cells(rows):
    return {(r["id"], r["scope"], r["strategy"]) for r in rows}


def metrics(rows):
    """Return dict of metrics on a row subset (drops null predict/score)."""
    yl = [(int(r["vulnerable"]), int(r["predict"]))
          for r in rows if r.get("predict") is not None]
    ss = [(int(r["vulnerable"]), r["score"])
          for r in rows if r.get("score") is not None]
    out = {}
    if yl:
        yt, yp = zip(*yl)
        out["acc"] = accuracy_score(yt, yp)
        out["f1"] = f1_score(yt, yp, zero_division=0)
        out["prec"] = precision_score(yt, yp, zero_division=0)
        out["rec"] = recall_score(yt, yp, zero_division=0)
        try:
            out["mcc"] = matthews_corrcoef(yt, yp)
        except Exception:
            out["mcc"] = float("nan")
        out["ppr"] = float(np.mean(yp))       # response bias
        out["n"] = len(yl)
    if ss:
        st, sc = zip(*ss)
        if len(set(st)) > 1:
            out["auc"] = roc_auc_score(st, sc)
    return out


def agg(per_model_vals):
    """mean +/- std across models for one metric (ignoring nan/missing)."""
    v = [x for x in per_model_vals if x is not None and not np.isnan(x)]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v)), len(v)


def fmt(m, s):
    if np.isnan(m):
        return "   -   "
    return f"{m:.3f}±{s:.3f}"


def table(title, keys, subset_fn, models, metric_order=("auc", "mcc", "f1", "acc", "ppr")):
    print(f"\n### {title}")
    hdr = "  " + f"{'':<14}" + "".join(f"{k.upper():>13}" for k in metric_order)
    print(hdr)
    rowvals = {}
    for key in keys:
        line = f"  {key:<14}"
        rowvals[key] = {}
        for mk in metric_order:
            per = []
            for label, rows in models.items():
                sub = subset_fn(rows, key)
                per.append(metrics(sub).get(mk))
            mean, std, n = agg(per)
            rowvals[key][mk] = mean
            line += f"{fmt(mean, std):>13}"
        print(line)
    return rowvals


def main():
    models = load_models(RESULTS)
    complete = {m: r for m, r in models.items() if len(cells(r)) >= N_SAMPLES * 3 * 2}
    partial = {m: r for m, r in models.items() if m not in complete}

    print("=" * 78)
    print(f"INTERIM REPORT  ({RESULTS})")
    print(f"complete models (used): {list(complete)}")
    if partial:
        print("partial (excluded from tables):")
        for m, r in partial.items():
            print(f"   {m}: {len(cells(r))}/{N_SAMPLES*3*2} cells")
    print("Discrimination = AUC (threshold-free), MCC.  "
          "Bias = PPR (pos-pred rate; base rate 0.50).")
    print("Values: mean±std across models.")
    print("=" * 78)

    M = complete

    # 1) by STRATEGY (pool scopes)
    rv_s = table("BY PROMPT STRATEGY (pooled over scope)", STRATS,
                 lambda rows, k: [r for r in rows if r["strategy"] == k], M)
    if all(s in rv_s for s in STRATS):
        print("  Δ(rag−zero):  " + "  ".join(
            f"{mk.upper()} {rv_s['rag'][mk]-rv_s['zero'][mk]:+.3f}"
            for mk in ("auc", "mcc", "f1", "ppr")))

    # 2) by SCOPE (pool strategies)
    rv_sc = table("BY INPUT SCOPE (pooled over strategy)", SCOPES,
                  lambda rows, k: [r for r in rows if r["scope"] == k], M)
    print("  scope trend (function→repository):  " + "  ".join(
        f"{mk.upper()} {rv_sc['function'][mk]:.3f}→{rv_sc['file'][mk]:.3f}"
        f"→{rv_sc['repository'][mk]:.3f}" for mk in ("auc", "ppr")))

    # 3) by TAG (pool strategies + scopes)
    table("BY COMPLEXITY TAG (pooled over strategy+scope)", TAGS,
          lambda rows, k: [r for r in rows if r.get("tag") == k], M)

    # 4) strategy × scope interaction on the two headline axes
    for mk, name in (("auc", "DISCRIMINATION (AUC)"), ("ppr", "BIAS (PPR)")):
        print(f"\n### {name}: strategy × scope")
        print("  " + f"{'':<10}" + "".join(f"{s:>13}" for s in SCOPES))
        for st in STRATS:
            line = f"  {st:<10}"
            for sc in SCOPES:
                per = [metrics([r for r in rows
                                if r["strategy"] == st and r["scope"] == sc]).get(mk)
                       for rows in M.values()]
                mean, std, _ = agg(per)
                line += f"{fmt(mean, std):>13}"
            print(line)

    # per-model AUC snapshot (sanity)
    print("\n### per-model AUC (pooled) — sanity")
    for label, rows in M.items():
        a = metrics(rows).get("auc")
        print(f"  {label:<22} AUC={a:.3f}" if a else f"  {label:<22} AUC=-")


if __name__ == "__main__":
    main()
