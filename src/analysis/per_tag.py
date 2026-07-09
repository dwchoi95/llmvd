"""Per-tag (SFSF/MFSF/MFMF) x scope analysis of detection results.

Operates purely on the per-row result JSONL files produced by the Detector
(`results/{model}/{benchmark}_{trial}.jsonl`), which already carry `tag`,
`group_id`, `language`, `scope`, `predict`, `vulnerable`, `tokens`,
`Time (sec)`. No sklearn/scipy dependency: metrics and the paired McNemar /
Cohen's d are implemented directly so it runs in the data-pipeline venv.

Outputs:
  - per_tag_table.csv        : (scope x tag) F1/P/R/Acc/tokens/time, mean+/-std over trials
  - per_tag_common.csv       : same, restricted to languages present in all tags (C+Python)
  - group_level_table.csv    : CVE/group-level recall (any vuln row caught) & false-alarm
  - paired_tests.csv         : McNemar + Cohen's d for scope pairs, per tag (Bonferroni noted)
"""

import argparse
import glob
import json
import math
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # std of single-trial groups -> harmless RuntimeWarnings

SCOPES = ["function", "file", "repository"]
TAGS = ["SFSF", "MFSF", "MFMF"]


# --------------------------------------------------------------------------- #
def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v == 1:
            return True
        if v == 0:
            return False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0"):
            return False
    return None


def load_results(results_dir: str, model: str, benchmark: str) -> pd.DataFrame:
    pat = os.path.join(results_dir, model, f"{benchmark}_*.jsonl")
    files = sorted(glob.glob(pat))
    if not files:
        files = sorted(glob.glob(os.path.join(results_dir, model, f"{benchmark}.jsonl")))
    frames = []
    for fp in files:
        stem = os.path.basename(fp)[:-6]
        trial = stem.split("_")[-1] if "_" in stem else "1"
        df = pd.read_json(fp, lines=True)
        df["trial"] = trial
        df["model"] = model
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no result files under {results_dir}/{model} for {benchmark}")
    df = pd.concat(frames, ignore_index=True)
    df["pred"] = df["predict"].apply(to_bool)
    df["truth"] = df["vulnerable"].apply(to_bool)
    return df.dropna(subset=["pred", "truth"])


def _metrics(truth, pred) -> dict:
    tp = int(((pred) & (truth)).sum())
    fp = int(((pred) & (~truth)).sum())
    fn = int(((~pred) & (truth)).sum())
    tn = int(((~pred) & (~truth)).sum())
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {"F1": f1, "Precision": prec, "Recall": rec, "Accuracy": acc, "n": n}


def per_tag_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (trial, scope, tag), g in df.groupby(["trial", "scope", "tag"]):
        truth = g["truth"].to_numpy(dtype=bool)
        pred = g["pred"].to_numpy(dtype=bool)
        m = _metrics(truth, pred)
        m.update({"trial": trial, "scope": scope, "tag": tag,
                  "tokens": g["tokens"].mean() if "tokens" in g else np.nan,
                  "time": g["Time (sec)"].mean() if "Time (sec)" in g else np.nan})
        rows.append(m)
    per_trial = pd.DataFrame(rows)
    agg = per_trial.groupby(["scope", "tag"]).agg(
        F1_mean=("F1", "mean"), F1_std=("F1", "std"),
        Precision_mean=("Precision", "mean"), Recall_mean=("Recall", "mean"),
        Accuracy_mean=("Accuracy", "mean"),
        tokens_mean=("tokens", "mean"), time_mean=("time", "mean"),
        n=("n", "mean"),
    ).reset_index()
    # order
    agg["scope"] = pd.Categorical(agg["scope"], SCOPES, ordered=True)
    agg["tag"] = pd.Categorical(agg["tag"], TAGS, ordered=True)
    return agg.sort_values(["tag", "scope"]).reset_index(drop=True)


def group_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """CVE/group-level: a group is 'detected' if any vuln=True row in it is
    flagged; 'false alarm' if any vuln=False row is flagged. Reported per
    (scope, tag), averaged over trials."""
    rows = []
    for (trial, scope, tag), g in df.groupby(["trial", "scope", "tag"]):
        det = fa = n_groups = n_fa_groups = 0
        for gid, gg in g.groupby("group_id"):
            vuln = gg[gg["truth"]]
            clean = gg[~gg["truth"]]
            if len(vuln):
                n_groups += 1
                if vuln["pred"].any():
                    det += 1
            if len(clean):
                n_fa_groups += 1
                if clean["pred"].any():
                    fa += 1
        rows.append({"trial": trial, "scope": scope, "tag": tag,
                     "group_recall": det / n_groups if n_groups else 0.0,
                     "group_false_alarm": fa / n_fa_groups if n_fa_groups else 0.0,
                     "n_groups": n_groups})
    per_trial = pd.DataFrame(rows)
    agg = per_trial.groupby(["scope", "tag"]).agg(
        group_recall_mean=("group_recall", "mean"),
        group_false_alarm_mean=("group_false_alarm", "mean"),
        n_groups=("n_groups", "mean"),
    ).reset_index()
    agg["scope"] = pd.Categorical(agg["scope"], SCOPES, ordered=True)
    agg["tag"] = pd.Categorical(agg["tag"], TAGS, ordered=True)
    return agg.sort_values(["tag", "scope"]).reset_index(drop=True)


def _chi2_p_df1(stat: float) -> float:
    # survival function of chi-square with 1 dof = erfc(sqrt(stat/2))
    return math.erfc(math.sqrt(stat / 2.0)) if stat > 0 else 1.0


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """McNemar (paired binary correctness) + Cohen's d for each scope pair,
    within each tag. Pairing unit = row evaluated under both scopes."""
    df = df.copy()
    df["correct"] = (df["pred"] == df["truth"]).astype(int)
    # key identifies the SAME sample across the three scopes. The Detector
    # assigns a distinct `index` per (row, scope), so it must NOT be used; a
    # dataset sample is uniquely keyed by (group_id, file_name, vulnerable).
    keycols = [c for c in ("group_id", "file_name", "vulnerable", "trial") if c in df.columns]
    rows = []
    pairs = [("file", "function"), ("repository", "function"), ("repository", "file")]
    for tag in TAGS:
        sub = df[df["tag"] == tag]
        wide = sub.pivot_table(index=keycols, columns="scope", values="correct", aggfunc="first")
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            w = wide[[a, b]].dropna()
            ca = w[a].to_numpy(); cb = w[b].to_numpy()
            b_only = int(((ca == 1) & (cb == 0)).sum())   # a correct, b wrong
            c_only = int(((ca == 0) & (cb == 1)).sum())   # a wrong, b correct
            disc = b_only + c_only
            stat = (abs(b_only - c_only) - 1) ** 2 / disc if disc else 0.0
            p = _chi2_p_df1(stat)
            diff = ca - cb
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else 0.0
            rows.append({"tag": tag, "pair": f"{a}-{b}", "n": len(w),
                         "a_better": b_only, "b_better": c_only,
                         "mean_gain": float(diff.mean()), "cohens_d": float(d),
                         "mcnemar_chi2": float(stat), "p_value": float(p)})
    out = pd.DataFrame(rows)
    if len(out):
        out["bonferroni_family"] = len(out)
        out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
    return out


COMMON_LANGS = {"c", "python"}  # languages present in all three tags (MFSF has no java)


def main():
    ap = argparse.ArgumentParser(description="Per-tag x scope analysis of detection results")
    ap.add_argument("-r", "--results", default="results", help="results dir")
    ap.add_argument("-m", "--model", required=True, help="model subdir under results/")
    ap.add_argument("-b", "--benchmark", default="FuncFileRepo.eval", help="dataset stem")
    ap.add_argument("-o", "--outdir", default="results/analysis", help="output dir")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_results(args.results, args.model, args.benchmark)
    print(f"loaded {len(df)} scored rows for {args.model}")

    pt = per_tag_table(df)
    pt.to_csv(os.path.join(args.outdir, f"{args.model}_per_tag_table.csv"), index=False)
    print("\n=== per-tag x scope (F1 mean) ===")
    print(pt.pivot(index="tag", columns="scope", values="F1_mean").round(3))

    common = df[df["language"].isin(COMMON_LANGS)]
    per_tag_table(common).to_csv(os.path.join(args.outdir, f"{args.model}_per_tag_common.csv"), index=False)

    gl = group_level_table(df)
    gl.to_csv(os.path.join(args.outdir, f"{args.model}_group_level_table.csv"), index=False)
    print("\n=== group-level recall (any vuln row caught) ===")
    print(gl.pivot(index="tag", columns="scope", values="group_recall_mean").round(3))

    pv = paired_tests(df)
    pv.to_csv(os.path.join(args.outdir, f"{args.model}_paired_tests.csv"), index=False)
    print("\n=== paired McNemar (scope gains) ===")
    if len(pv):
        print(pv[["tag", "pair", "mean_gain", "cohens_d", "p_value", "p_bonferroni"]].round(4).to_string(index=False))
    print(f"\nwrote tables to {args.outdir}/")


if __name__ == "__main__":
    main()
