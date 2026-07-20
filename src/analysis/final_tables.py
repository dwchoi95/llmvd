"""Trial-aware final aggregation for the paper tables.

For each result label we treat each trial file as a SEPARATE trial and compute
the metric per trial, then report the mean (and std) ACROSS trials. This is the
metric-averaging estimand (NOT a majority/consensus vote, which would sharpen
predictions and distort the response-bias PPR). Every path is run at 3 trials
(`<bench>_1..3.jsonl`) because batch nondeterminism perturbs near-boundary
verdicts even for direct (non-reasoning/sft) decoding; the original single-trial
base run was reused as trial 1 (`<bench>.jsonl` renamed to `<bench>_1.jsonl`).
A leftover bare `<bench>.jsonl` is ignored whenever suffixed trial files exist.

  usage: python -m src.analysis.final_tables [results_dir]
"""
from __future__ import annotations
import glob, json, os, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             recall_score, precision_score, roc_auc_score)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/final"
BENCH = "FuncFileRepo.test"
SC = ["function", "file", "repository"]
ST = ["zero", "rag"]
TAGS = ["SFSF", "MFSF", "MFMF"]
REASON = {"qwen3-30b-thinking", "magistral-small:24b", "nemotron-nano-think:12b"}
BASE = ["qwen3-30b-instruct", "qwen3-30b-thinking", "mistral-small:24b",
        "magistral-small:24b", "nemotron-nano:12b", "nemotron-nano-think:12b"]


def trial_files(label_dir):
    suffixed = sorted(glob.glob(os.path.join(label_dir, f"{BENCH}_*.jsonl")))
    if suffixed:
        return suffixed
    bare = os.path.join(label_dir, f"{BENCH}.jsonl")
    return [bare] if os.path.exists(bare) else []


def load_label(label):
    d = os.path.join(ROOT, label)
    trials = []
    for f in trial_files(d):
        trials.append([json.loads(l) for l in open(f)])
    return trials  # list of trials, each a list of rows


def metric(rows, mk):
    yl = [(int(r["vulnerable"]), int(r["predict"])) for r in rows if r.get("predict") is not None]
    ss = [(int(r["vulnerable"]), r["score"]) for r in rows if r.get("score") is not None]
    if mk == "auc":
        if not ss: return None
        t, s = zip(*ss); return roc_auc_score(t, s) if len(set(t)) > 1 else None
    if not yl: return None
    t, p = zip(*yl)
    if mk == "ppr": return float(np.mean(p))
    if mk == "mcc":
        try: return matthews_corrcoef(t, p)
        except Exception: return float("nan")
    fn = {"f1": f1_score, "acc": accuracy_score, "rec": recall_score, "prec": precision_score}[mk]
    return fn(t, p, **({"zero_division": 0} if mk in ("f1", "rec", "prec") else {}))


def per_model_metric(label, filt, mk):
    """mean metric across trials for one label, on rows passing filt."""
    trials = load_label(label)
    vals = []
    for rows in trials:
        v = metric([r for r in rows if filt(r)], mk)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return float(np.mean(vals)) if vals else None  # trial-mean per model


def agg(models, filt, mk):
    """mean +/- std ACROSS models of the per-model trial-mean."""
    vals = [per_model_metric(m, filt, mk) for m in models]
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)


def main():
    models = [m for m in BASE if os.path.isdir(os.path.join(ROOT, m))]
    sft = [f"{m}-sft" for m in BASE if os.path.isdir(os.path.join(ROOT, f"{m}-sft"))]
    # report trial counts
    tc = {m: len(trial_files(os.path.join(ROOT, m))) for m in models}
    out = {"models": models, "trial_counts": tc, "sft_models": sft}

    MK = ["auc", "mcc", "f1", "ppr", "acc"]
    # per model (pooled scope+strategy = zero/rag)
    out["per_model"] = {m: {k: (round(per_model_metric(m, lambda r: True, k), 4)
                                if per_model_metric(m, lambda r: True, k) is not None else None)
                            for k in MK} for m in models}
    # groups
    out["group"] = {}
    for g, mem in [("reasoning", [m for m in models if m in REASON]),
                   ("nonreasoning", [m for m in models if m not in REASON])]:
        out["group"][g] = {k: agg(mem, lambda r: True, k)[0] for k in MK}
    # by scope / strategy / tag / language (mean over all models)
    for name, keys, fkey in [("by_scope", SC, "scope"), ("by_strategy", ST, "strategy"),
                             ("by_tag", TAGS, "tag")]:
        out[name] = {key: {k: agg(models, lambda r, key=key, fkey=fkey: r.get(fkey) == key, k)[0]
                           for k in ["auc", "mcc", "f1", "ppr"]} for key in keys}
    def lang(r):
        l = r.get("language", "").lower(); return "c/c++" if l in ("c", "c++") else l
    out["by_language"] = {lg: {k: agg(models, lambda r, lg=lg: lang(r) == lg, k)[0]
                               for k in ["auc", "mcc", "f1", "ppr"]} for lg in ["c/c++", "java", "python"]}
    # interaction strat x scope on auc, ppr
    out["interaction"] = {}
    for mk in ["auc", "ppr"]:
        out["interaction"][mk] = {st: {sc: agg(models, lambda r, st=st, sc=sc: r["strategy"] == st and r["scope"] == sc, mk)[0]
                                       for sc in SC} for st in ST}
    # sft aggregates (single trial each)
    if sft:
        out["sft"] = {"overall": {k: agg(sft, lambda r: True, k)[0] for k in MK},
                      "by_scope": {sc: {k: agg(sft, lambda r, sc=sc: r["scope"] == sc, k)[0]
                                        for k in ["auc", "mcc", "f1", "ppr"]} for sc in SC}}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
