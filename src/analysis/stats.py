"""Confirmatory statistics for the scope x prompting x response-bias study.

Runs AFTER the full experiment, on the same result JSONLs consumed by
`src.analysis.decomposition`, and implements the HANDOFF §8 battery:

  1. Paired tests over model means (n = #models): mean Δ with 95% t-CI and
     Cohen's dz for every scope pair and strategy pair. The PRIMARY
     discrimination metric is the empirical, threshold-free `auc_score`
     (from captured P(vulnerable) scores). Single-threshold metrics
     (MCC / balanced-acc / d') are reported as OPERATING-POINT metrics: they
     mechanically move when the operating point slides along a fixed ROC
     curve, so a significant change there is NOT by itself evidence that
     discrimination moved — interpret jointly with auc_score.
  2. Row-level McNemar per (model, condition pair) — TWICE per pair:
       * on CORRECTNESS  (predict == label): did accuracy move
         (thesis expects mostly non-significant);
       * on POSITIVITY   (predict == 1): did the response bias move
         (thesis expects significant).
     Rows are paired on (sample, trial) — stratified over the trials both
     conditions share; NO consensus/majority vote over trials is formed
     anywhere (any such vote is a different, sharpened estimand that
     distorts PPR for multi-trial strategies). Plus a CVE/group-level
     Wilcoxon on per-group deltas, which respects the non-independence of
     rows drawn from the same CVE.
  3. Holm correction (on the paired-t p-values) within each (metric, slice)
     group of contrasts, and within each (model, family) for McNemar.
     NOTE: the exact Wilcoxon at n=6 models has a two-sided p floor of
     2/2^6 = 0.03125, so Holm across >=3 contrasts can never push it under
     0.05 — p_w is reported RAW for orientation only; inference rests on the
     t-based CIs and effect sizes.
  4. Realistic-prevalence reweighting: each cell's trial-marginal (TPR, FPR)
     is analytically reweighted to ~1-5% positive prevalence; uncertainty
     comes from bootstrap-resampling rows WITHIN each trial (native size)
     and reweighting each draw — so CI width reflects the real data's
     information, not a hypothetical tiny sample.
  5. Threshold-shift reconstruction: can a single per-(model,strategy)
     threshold move on the FUNCTION-scope score distribution reproduce the
     other scopes' observed (TPR, FPR)? Reported as within-TPR and
     within-FPR R² (computing one pooled R² over both rate clusters is
     inflated by the TPR-vs-FPR separation) plus RMSE. Score ties (e.g. mass
     at exactly 0/1) make some PPRs unattainable; the achieved PPR is
     recorded so such cells are visible.
  6. ROC operating-point figures per model + predicted-vs-observed
     reconstruction scatter.

Pairing note: rows are paired across conditions by group_id|file_name|label
(the Detector's resume key). On the shipped eval set 6 of 1281 rows share a
key with a byte-identical duplicate row; one of each is dropped at load so
trial-paired merges stay 1:1 — harmless.

Usage:
    ./env/bin/python -m src.analysis.stats \
        --models "llama3.1:8b,mistral-nemo:12b,phi3:14b,qwen2.5-coder:14b,deepseek-coder-v2:16b,qwen3-coder:30b" \
        --benchmark FuncFileRepo.eval --trials all \
        --results_dir results/full --outdir results/analysis/stats
"""
from __future__ import annotations

import json
import zlib
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from .decomposition import (
    SCOPES,
    _to_bool,
    _METRIC_KEYS,
    binary_metrics,
    trial_paths,
)

SCOPE_PAIRS = [("function", "file"), ("file", "repository"),
               ("function", "repository")]
STRATEGY_ORDER = ["zero_shot", "few_shot", "cot", "think_verify"]

# threshold-free, score-based: the primary discrimination evidence
PRIMARY_DISC_METRICS = ["auc_score"]
# single-threshold metrics: move mechanically with the operating point even
# on a fixed ROC curve — report, but never alone as "discrimination moved"
OPERATING_POINT_METRICS = ["mcc", "balanced_acc", "dprime"]
BIAS_METRICS = ["ppr", "criterion"]
COMP_METRICS = ["accuracy", "f1"]
ALL_METRICS = (PRIMARY_DISC_METRICS + OPERATING_POINT_METRICS
               + BIAS_METRICS + COMP_METRICS)

_CELL_METRIC_KEYS = _METRIC_KEYS + ["n_scored"]


def _strategy_order(present) -> list[str]:
    """Known strategies in canonical order, then any others alphabetically
    (never silently dropped)."""
    present = [s for s in pd.unique(pd.Series(list(present))) if s]
    known = [s for s in STRATEGY_ORDER if s in present]
    extra = sorted(s for s in present if s not in STRATEGY_ORDER)
    return known + extra


# --------------------------------------------------------------------------- #
# loading (keeps the per-sample key that decomposition.py drops, so rows can
# be PAIRED across scope/strategy conditions)
# --------------------------------------------------------------------------- #
def result_files(results_dir: str, model: str, benchmark: str,
                 trials: str) -> list[Path]:
    """All result files that hold trials for this model/benchmark.

    Unlike decomposition.trial_paths, `all` includes BOTH the un-suffixed
    {benchmark}.jsonl AND the {benchmark}_k.jsonl files when they coexist —
    the HANDOFF workflow produces exactly that mix (direct strategies with
    -e 1 -> un-suffixed; extra reasoning trials with -e 3 -> suffixed), and
    taking only the suffixed set silently erases the direct strategies.
    """
    base = Path(results_dir) / model
    if trials == "all":
        files = []
        single = base / f"{benchmark}.jsonl"
        if single.exists():
            files.append(single)
        suffixed = [p for p in sorted(base.glob(f"{benchmark}_*.jsonl"))
                    if p.stem.split("_")[-1].isdigit()]
        files.extend(suffixed)
        return files
    return [Path(p) for p in trial_paths(results_dir, model, benchmark, trials)]


def load_rows(results_dir: str, model: str, benchmark: str,
              trials: str) -> pd.DataFrame:
    """All scored rows of every trial file, with a per-sample pairing key.

    Each FILE is one independent run, so it gets its own trial id (the
    un-suffixed file and _k files never share one).
    """
    paths = result_files(results_dir, model, benchmark, trials)
    rows = []
    n_parse_fail = 0
    for ti, path in enumerate(paths, start=1):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lab = r.get("vulnerable")
                if lab is None:
                    continue
                pred = _to_bool(r.get("predict"))
                if pred is None:
                    n_parse_fail += 1
                    continue
                sc = r.get("score")
                if not isinstance(sc, (int, float)) or (
                        isinstance(sc, float) and np.isnan(sc)):
                    sc = np.nan
                rid = r.get("id")
                rows.append({
                    "trial": ti,
                    # unique row id when the dataset provides one (final
                    # datasets); legacy fallback = Detector's resume triple
                    "key": (f"id{int(rid)}" if rid is not None else
                            f"{r.get('group_id')}|{r.get('file_name')}|{lab}"),
                    # CVE-level cluster for the group-level tests
                    "group": r.get("group_id") or r.get("cve_id"),
                    "y": bool(lab),
                    "p": pred,
                    "score": sc,
                    "scope": r.get("scope"),
                    # tolerate legacy files without a strategy field
                    "strategy": r.get("strategy") or "na",
                    "tag": r.get("tag"),
                })
    df = pd.DataFrame(rows)
    if len(df):
        # the eval set ships a handful of byte-identical duplicate rows
        # (same pairing key); keep one per (key, trial, scope, strategy) so
        # trial-paired merges stay 1:1
        df = df.drop_duplicates(
            subset=["key", "trial", "scope", "strategy"], keep="first"
        ).reset_index(drop=True)
        df.attrs["n_parse_fail"] = n_parse_fail
        df.attrs["files"] = [p.name for p in paths]
    return df


def _trial_rates(g: pd.DataFrame) -> tuple[float, float, float]:
    """(TPR, FPR, PPR) of a cell as the equal-weight mean of per-trial rates.

    NO row-level consensus/majority vote is taken anywhere in this module:
    any deterministic aggregation of an even or odd number of stochastic
    trials into one row-level prediction is a DIFFERENT estimand that
    sharpens rates toward each row's majority side, distorting PPR — the
    study's primary bias metric — asymmetrically for multi-trial (reasoning)
    strategies. All rates are therefore trial-marginal: computed per trial,
    then averaged.
    """
    tprs, fprs, pprs = [], [], []
    for _, gt in g.groupby("trial"):
        pos = gt.loc[gt["y"], "p"]
        neg = gt.loc[~gt["y"], "p"]
        if len(gt):
            pprs.append(float(gt["p"].mean()))
        if len(pos):
            tprs.append(float(pos.mean()))
        if len(neg):
            fprs.append(float(neg.mean()))
    return (float(np.mean(tprs)) if tprs else np.nan,
            float(np.mean(fprs)) if fprs else np.nan,
            float(np.mean(pprs)) if pprs else np.nan)


# --------------------------------------------------------------------------- #
# per-cell metric table (metrics per trial, then mean over trials — matches
# decomposition.py's aggregation)
# --------------------------------------------------------------------------- #
def _recs(g: pd.DataFrame) -> list[dict]:
    """binary_metrics-ready records; NaN scores must become None (NaN would
    reach roc_auc_score and crash / poison it)."""
    return [{"y": bool(y), "p": bool(p),
             "score": None if pd.isna(s) else float(s)}
            for y, p, s in zip(g["y"], g["p"], g["score"])]


def _mean_std_ext(dicts: list[dict]) -> dict:
    """decomposition._mean_std over the extended key set (adds n_scored)."""
    out = {}
    for k in _CELL_METRIC_KEYS:
        vals = np.array([d[k] for d in dicts if k in d and d[k] is not None],
                        dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(np.mean(vals)) if len(vals) else np.nan
        out[k + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    return out


def cell_table(df: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for (strat, scope), g in df.groupby(["strategy", "scope"]):
        per_trial = []
        for _, gt in g.groupby("trial"):
            m = binary_metrics(_recs(gt))
            if m:
                per_trial.append(m)
        if per_trial:
            rows.append({"model": model, "strategy": strat, "scope": scope,
                         "n_trials": len(per_trial), **_mean_std_ext(per_trial)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. paired contrasts over model means
# --------------------------------------------------------------------------- #
def paired_stats(deltas) -> dict:
    """Mean Δ, 95% t-CI, Cohen's dz, paired-t and Wilcoxon p for one contrast.

    p_w (exact Wilcoxon) is orientation only: at n=6 its two-sided floor is
    0.03125, structurally incapable of surviving Holm across >=3 contrasts.
    """
    d = np.asarray(list(deltas), dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    out = {"n_models": n, "mean_delta": np.nan, "ci_lo": np.nan,
           "ci_hi": np.nan, "cohen_dz": np.nan, "p_t": np.nan, "p_w": np.nan}
    if n < 2:
        if n == 1:
            out["mean_delta"] = float(d[0])
        return out
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(sps.t.ppf(0.975, n - 1))
    out.update(mean_delta=mean, ci_lo=mean - tcrit * se,
               ci_hi=mean + tcrit * se,
               cohen_dz=(mean / sd) if sd > 0 else np.nan)
    if sd > 0:
        out["p_t"] = float(sps.ttest_1samp(d, 0.0).pvalue)
        try:
            out["p_w"] = float(sps.wilcoxon(
                d, zero_method="wilcox", alternative="two-sided",
                method="auto").pvalue)
        except ValueError:  # e.g. all deltas zero after dropping
            pass
    return out


def _contrast_rows(cells: pd.DataFrame, family: str,
                   pairs: list[tuple[str, str]], axis_col: str,
                   within_col: str) -> pd.DataFrame:
    """Paired model-mean contrasts along `axis_col`, both marginal over
    `within_col` ('ALL') and within each level of `within_col`."""
    rows = []
    if within_col == "strategy":
        within_levels = _strategy_order(cells[within_col].dropna())
    else:
        within_levels = [s for s in SCOPES
                         if s in set(cells[within_col].dropna())]
    for metric in ALL_METRICS:
        if metric not in cells.columns:
            continue
        piv = cells.pivot_table(index=["model", within_col],
                                columns=axis_col, values=metric)
        for a, b in pairs:
            if a not in piv.columns or b not in piv.columns:
                continue
            delta = piv[b] - piv[a]
            marg = delta.groupby(level="model").mean()
            rows.append({"family": family, "metric": metric,
                         "contrast": f"{b} - {a}", within_col: "ALL",
                         **paired_stats(marg.values)})
            for lvl in within_levels:
                try:
                    sub = delta.xs(lvl, level=within_col)
                except KeyError:
                    continue
                rows.append({"family": family, "metric": metric,
                             "contrast": f"{b} - {a}", within_col: lvl,
                             **paired_stats(sub.values)})
    return pd.DataFrame(rows)


def add_holm(df: pd.DataFrame, group_cols: list[str],
             pcols: tuple[str, ...] = ("p_t",)) -> pd.DataFrame:
    """Holm-adjust each p-value column within groups of contrasts.

    Only the t-based p is corrected by default: the exact Wilcoxon's n=6
    floor makes a Holm-corrected p_w a structurally dead column.
    """
    df = df.reset_index(drop=True)
    for pc in pcols:
        if pc not in df.columns:
            continue
        adj_col = np.full(len(df), np.nan)
        for _, idx in df.groupby(group_cols, dropna=False).groups.items():
            idx = np.asarray(list(idx))
            ps = df.loc[idx, pc].to_numpy(dtype=float)
            mask = ~np.isnan(ps)
            if mask.sum() == 0:
                continue
            adj = multipletests(ps[mask], method="holm")[1]
            adj_col[idx[mask]] = adj
        df[pc + "_holm"] = adj_col
    return df


def paired_contrasts(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = _strategy_order(cells["strategy"].dropna())
    strat_pairs = list(itertools.combinations(strategies, 2))
    scope_df = _contrast_rows(cells, "scope", SCOPE_PAIRS,
                              axis_col="scope", within_col="strategy")
    strat_df = _contrast_rows(cells, "strategy", strat_pairs,
                              axis_col="strategy", within_col="scope")
    # Holm across the contrasts of one metric within one view (e.g. the 3
    # scope pairs seen by the same metric & strategy slice)
    scope_df = add_holm(scope_df, ["metric", "strategy"])
    strat_df = add_holm(strat_df, ["metric", "scope"])
    return scope_df, strat_df


# --------------------------------------------------------------------------- #
# 2. McNemar (row level) + group-level Wilcoxon
# --------------------------------------------------------------------------- #
def _mcnemar_p(n01: int, n10: int) -> float:
    """Exact binomial McNemar for small discordant counts, else corrected chi2."""
    if n01 + n10 == 0:
        return 1.0
    table = [[0, n01], [n10, 0]]
    exact = (n01 + n10) < 25
    res = sm_mcnemar(table, exact=exact, correction=True)
    return float(res.pvalue)


def _pair_tests(sub: pd.DataFrame, cond_col: str, a: str, b: str) -> dict | None:
    """McNemar on correctness & positivity + group-level Wilcoxon for one
    paired condition (a vs b) inside one model/slice.

    Rows are paired on (sample key, trial) — a stratified-by-trial McNemar
    that sums discordant counts over the trials both conditions share. No
    consensus vote is formed (see _trial_rates for why). Conditions with no
    common trial (e.g. a strategy run only in extra trial files) simply
    contribute no pairs and the test is skipped.
    """
    pa = sub[sub[cond_col] == a]
    pb = sub[sub[cond_col] == b]
    m = pa.merge(pb, on=["key", "trial"], suffixes=("_a", "_b"))
    if m.empty:
        return None
    out = {"n_pairs": int(len(m)),
           "n_keys": int(m["key"].nunique()),
           "n_trials_paired": int(m["trial"].nunique())}

    for kind, va, vb in (
        ("correct", (m["p_a"] == m["y_a"]), (m["p_b"] == m["y_b"])),
        ("positive", m["p_a"].astype(bool), m["p_b"].astype(bool)),
    ):
        n01 = int((va & ~vb).sum())   # a yes, b no
        n10 = int((~va & vb).sum())   # a no, b yes
        out[f"{kind}_n01"] = n01
        out[f"{kind}_n10"] = n10
        out[f"{kind}_rate_a"] = float(va.mean())
        out[f"{kind}_rate_b"] = float(vb.mean())
        out[f"p_mcnemar_{kind}"] = _mcnemar_p(n01, n10)
        # group(CVE)-level: Wilcoxon over per-group delta rates — respects
        # within-CVE (and within-trial) dependence that row-level McNemar
        # ignores
        gd = ((vb.astype(float) - va.astype(float))
              .groupby(m["group_a"]).mean()
              .dropna().to_numpy(dtype=float))
        out[f"{kind}_n_groups"] = int(len(gd))
        if len(gd) >= 2 and np.any(gd != 0):
            try:
                out[f"p_group_{kind}"] = float(sps.wilcoxon(
                    gd, zero_method="wilcox", alternative="two-sided",
                    method="auto").pvalue)
            except ValueError:
                out[f"p_group_{kind}"] = np.nan
        else:
            out[f"p_group_{kind}"] = np.nan
    return out


def mcnemar_tests(df: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    strategies = _strategy_order(df["strategy"].dropna())
    # scope pairs within each strategy
    for strat in strategies:
        sub = df[df["strategy"] == strat]
        for a, b in SCOPE_PAIRS:
            r = _pair_tests(sub, "scope", a, b)
            if r:
                rows.append({"model": model, "family": "scope",
                             "strategy": strat, "scope": "-",
                             "contrast": f"{b} - {a}", **r})
    # strategy pairs within each scope
    for scope in SCOPES:
        sub = df[df["scope"] == scope]
        for a, b in itertools.combinations(strategies, 2):
            r = _pair_tests(sub, "strategy", a, b)
            if r:
                rows.append({"model": model, "family": "strategy",
                             "strategy": "-", "scope": scope,
                             "contrast": f"{b} - {a}", **r})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. realistic-prevalence reweighting + row-bootstrap
# --------------------------------------------------------------------------- #
def _mcc_from_rates(pi: float, tpr: float, fpr: float) -> float:
    tp, fn = pi * tpr, pi * (1 - tpr)
    fp, tn = (1 - pi) * fpr, (1 - pi) * (1 - fpr)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def _cell_rng(seed: int, model: str, strat: str, scope: str) -> np.random.Generator:
    """Deterministic per-cell RNG: independent of --models order/composition
    and of which other cells exist."""
    key = zlib.crc32(f"{model}|{strat}|{scope}".encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def prevalence_analysis(df: pd.DataFrame, model: str,
                        prevalences: list[float], n_boot: int,
                        seed: int) -> pd.DataFrame:
    """Analytic reweighting of each cell to target prevalence pi:
        PPR(pi) = pi*TPR + (1-pi)*FPR;  MCC(pi) from the rate-based
    confusion proportions, where (TPR, FPR) are trial-marginal (per-trial
    rates, averaged). CIs: bootstrap rows WITHIN each trial (native size,
    stratified by class), average the per-trial rates per draw, reweight —
    so the interval reflects the actual data's information, not a
    hypothetical tiny-positive sample."""
    rows = []
    for (strat, scope), g in df.groupby(["strategy", "scope"]):
        # deterministic row order (result files are written in asyncio
        # completion order, which is not stable across identical runs)
        g = g.sort_values(["trial", "key"], kind="mergesort")
        per_trial = []
        for _, gt in g.groupby("trial"):
            pos = gt.loc[gt["y"], "p"].to_numpy(dtype=bool)
            neg = gt.loc[~gt["y"], "p"].to_numpy(dtype=bool)
            if len(pos) and len(neg):
                per_trial.append((pos, neg))
        if not per_trial:
            continue
        rng = _cell_rng(seed, model, strat, scope)
        tpr = float(np.mean([p.mean() for p, _ in per_trial]))
        fpr = float(np.mean([n.mean() for _, n in per_trial]))
        _, _, ppr_native = _trial_rates(g)
        prev_native = float(g["y"].mean())
        # one bootstrap of the trial-averaged rates, reused for every pi
        tprs = np.empty(n_boot)
        fprs = np.empty(n_boot)
        for i in range(n_boot):
            ts, fs = [], []
            for pos, neg in per_trial:
                ts.append(pos[rng.integers(0, len(pos), len(pos))].mean())
                fs.append(neg[rng.integers(0, len(neg), len(neg))].mean())
            tprs[i] = np.mean(ts)
            fprs[i] = np.mean(fs)
        for pi in prevalences:
            pprs = pi * tprs + (1 - pi) * fprs
            mccs = np.array([_mcc_from_rates(pi, t, f)
                             for t, f in zip(tprs, fprs)])
            rows.append({
                "model": model, "strategy": strat, "scope": scope,
                "prevalence": pi,
                "n_pos": int(g["y"].sum()), "n_neg": int((~g["y"]).sum()),
                "n_trials": len(per_trial),
                "tpr": tpr, "fpr": fpr,
                "ppr_native": ppr_native,
                "mcc_native": _mcc_from_rates(prev_native, tpr, fpr),
                "ppr": pi * tpr + (1 - pi) * fpr,
                "mcc": _mcc_from_rates(pi, tpr, fpr),
                "ppr_ci_lo": float(np.percentile(pprs, 2.5)),
                "ppr_ci_hi": float(np.percentile(pprs, 97.5)),
                "mcc_ci_lo": float(np.percentile(mccs, 2.5)),
                "mcc_ci_hi": float(np.percentile(mccs, 97.5)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. threshold-shift reconstruction
# --------------------------------------------------------------------------- #
def _threshold_for_ppr(scores: np.ndarray, target_ppr: float) -> tuple[float, float]:
    """Threshold on `scores` whose achieved PPR = mean(scores >= thr) is as
    close as possible to target_ppr. With heavy score ties (e.g. mass at
    exactly 0/1) the target may be unattainable; returns (thr, achieved)."""
    cands = np.unique(scores)  # ascending; ppr(c) = mean(s >= c)
    achieved = (scores[None, :] >= cands[:, None]).mean(axis=1)
    # +inf candidate = predict-nothing (ppr 0)
    cands = np.append(cands, np.inf)
    achieved = np.append(achieved, 0.0)
    i = int(np.argmin(np.abs(achieved - target_ppr)))
    return float(cands[i]), float(achieved[i])


def threshold_shift(df: pd.DataFrame, model: str,
                    reference_scope: str = "function") -> pd.DataFrame:
    """Predict each non-reference scope's (TPR, FPR) by sliding a threshold on
    the reference scope's score distribution until PPR matches (as nearly as
    ties allow), then compare with the observed rates.

    Reference scores are pooled over trials (the trial-marginal score
    distribution); target rates are trial-marginal (_trial_rates)."""
    rows = []
    for strat, g in df.groupby("strategy"):
        ref = g[(g["scope"] == reference_scope) & g["score"].notna()]
        if len(ref) < 10 or ref["y"].nunique() < 2:
            continue
        s = ref["score"].to_numpy(dtype=float)
        y = ref["y"].to_numpy(dtype=bool)
        for target in SCOPES:
            if target == reference_scope:
                continue
            tg = g[g["scope"] == target]
            if len(tg) == 0 or tg["y"].nunique() < 2:
                continue
            tpr_obs, fpr_obs, ppr_t = _trial_rates(tg)
            if np.isnan(ppr_t):
                continue
            thr, ppr_achieved = _threshold_for_ppr(s, ppr_t)
            phat = s >= thr
            rows.append({
                "model": model, "strategy": strat,
                "reference": reference_scope, "target": target,
                "n_ref": int(len(ref)), "n_target": int(len(tg)),
                "ppr_target": ppr_t,
                "ppr_achieved": ppr_achieved,
                "tpr_pred": float(phat[y].mean()),
                "fpr_pred": float(phat[~y].mean()),
                "tpr_obs": tpr_obs,
                "fpr_obs": fpr_obs,
            })
    return pd.DataFrame(rows)


def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - float(np.sum((obs - pred) ** 2)) / ss_tot


def reconstruction_r2(df: pd.DataFrame) -> dict:
    """Within-rate R² (TPR and FPR separately) + pooled RMSE.

    A single R² over the concatenated TPR+FPR points is inflated by the
    TPR-vs-FPR mean separation (a predictor that only knows 'TPRs are higher
    than FPRs' already scores well) — so it is deliberately not reported.
    """
    if df.empty:
        return {"r2_tpr": np.nan, "r2_fpr": np.nan, "rmse": np.nan,
                "n_points": 0}
    t_obs = df["tpr_obs"].to_numpy(dtype=float)
    t_pred = df["tpr_pred"].to_numpy(dtype=float)
    f_obs = df["fpr_obs"].to_numpy(dtype=float)
    f_pred = df["fpr_pred"].to_numpy(dtype=float)
    obs = np.concatenate([t_obs, f_obs])
    pred = np.concatenate([t_pred, f_pred])
    return {"r2_tpr": _r2(t_obs, t_pred),
            "r2_fpr": _r2(f_obs, f_pred),
            "rmse": float(np.sqrt(np.mean((obs - pred) ** 2))),
            "n_points": int(len(obs))}


# --------------------------------------------------------------------------- #
# 6. figures
# --------------------------------------------------------------------------- #
_SCOPE_MARKERS = {"function": "o", "file": "s", "repository": "^"}


def roc_figure(df: pd.DataFrame, model: str, outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    strategies = _strategy_order(df["strategy"].dropna())
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(strategies)}

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=1, zorder=1)
    for (strat, scope), g in df.groupby(["strategy", "scope"]):
        if g["y"].nunique() < 2:
            continue
        # empirical ROC of this cell's trial-pooled scores (thin, background)
        sg = g[g["score"].notna()]
        if len(sg) >= 10 and sg["y"].nunique() > 1:
            fpr, tpr, _ = roc_curve(sg["y"].astype(int),
                                    sg["score"].astype(float))
            ax.plot(fpr, tpr, c=colors[strat], alpha=0.25, lw=0.8, zorder=2)
        # the trial-marginal binary operating point
        t, f, _ = _trial_rates(g)
        if np.isnan(t) or np.isnan(f):
            continue
        ax.scatter([f], [t], c=[colors[strat]],
                   marker=_SCOPE_MARKERS.get(scope, "x"), s=55,
                   edgecolors="black", linewidths=0.5, zorder=3)
    strat_handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[s],
                                label=s) for s in strategies]
    scope_handles = [plt.Line2D([0], [0], marker=_SCOPE_MARKERS[s], ls="",
                                color="gray", label=s) for s in SCOPES]
    ax.legend(handles=strat_handles + scope_handles, fontsize=7,
              loc="lower right", ncol=2)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{model}: operating points by scope x strategy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def reconstruction_figure(recon: pd.DataFrame, outpath: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=1)
    ax.scatter(recon["tpr_obs"], recon["tpr_pred"], s=30, alpha=0.75,
               label="TPR", marker="o")
    ax.scatter(recon["fpr_obs"], recon["fpr_pred"], s=30, alpha=0.75,
               label="FPR", marker="s")
    stats_ = reconstruction_r2(recon)
    ax.set_xlabel("observed rate (target scope)")
    ax.set_ylabel("predicted rate (threshold shift on function scope)")
    ax.set_title("Threshold-shift reconstruction "
                 f"(R²_TPR={stats_['r2_tpr']:.3f}, "
                 f"R²_FPR={stats_['r2_fpr']:.3f}, "
                 f"RMSE={stats_['rmse']:.3f})", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _sig_summary(df: pd.DataFrame, pcol: str, alpha: float = 0.05) -> str:
    """'k/m significant (j skipped)' — skipped (NaN p) tests stay visible;
    zero-movement pairs are exactly the ones that go NaN, so hiding them
    would overstate the significant fraction."""
    if df.empty or pcol not in df.columns:
        return "n/a"
    vals = df[pcol]
    n_skip = int(vals.isna().sum())
    tested = vals.dropna()
    s = f"{int((tested < alpha).sum())}/{len(tested)}"
    if n_skip:
        s += f" ({n_skip} skipped)"
    return s


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results/full")
    ap.add_argument("--models", required=True,
                    help="comma-separated model dirs")
    ap.add_argument("--benchmark", default="FuncFileRepo.eval")
    ap.add_argument("--trials", default="all")
    ap.add_argument("--outdir", default="results/analysis/stats")
    ap.add_argument("--prevalences", default="0.01,0.05")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prevalences = [float(x) for x in args.prevalences.split(",") if x.strip()]
    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    all_cells, all_mcnemar, all_prev, all_recon = [], [], [], []
    for model in models:
        df = load_rows(args.results_dir, model, args.benchmark, args.trials)
        if df.empty:
            print(f"[skip] {model}: no rows "
                  f"(results_dir={args.results_dir}, benchmark={args.benchmark})")
            continue
        print(f"[load] {model}: {len(df)} scored rows from "
              f"{df.attrs.get('files')}, "
              f"{df.attrs.get('n_parse_fail', 0)} parse failures excluded")
        cells = cell_table(df, model)
        all_cells.append(cells)
        all_mcnemar.append(mcnemar_tests(df, model))
        all_prev.append(prevalence_analysis(df, model, prevalences,
                                            args.bootstrap, args.seed))
        recon = threshold_shift(df, model)
        all_recon.append(recon)
        if not args.no_figures:
            roc_figure(df, model,
                       outdir / "figures" / f"roc_{model.replace(':', '_')}.png")

    if not all_cells:
        print("No data loaded — check --results_dir/--models/--benchmark.")
        return

    cells = pd.concat(all_cells, ignore_index=True)
    scope_con, strat_con = paired_contrasts(cells)
    mcnemar_df = pd.concat(all_mcnemar, ignore_index=True) \
        if all_mcnemar else pd.DataFrame()
    if not mcnemar_df.empty:
        mcnemar_df = add_holm(
            mcnemar_df, ["model", "family"],
            pcols=("p_mcnemar_correct", "p_mcnemar_positive",
                   "p_group_correct", "p_group_positive"))
    prev_df = pd.concat(all_prev, ignore_index=True) \
        if all_prev else pd.DataFrame()
    recon_df = pd.concat(all_recon, ignore_index=True) \
        if all_recon else pd.DataFrame()

    cells.to_csv(outdir / "cells.csv", index=False)
    scope_con.to_csv(outdir / "contrasts_scope.csv", index=False)
    strat_con.to_csv(outdir / "contrasts_strategy.csv", index=False)
    if not mcnemar_df.empty:
        mcnemar_df.to_csv(outdir / "mcnemar.csv", index=False)
    if not prev_df.empty:
        prev_df.to_csv(outdir / "prevalence.csv", index=False)
    if not recon_df.empty:
        recon_df.to_csv(outdir / "threshold_shift.csv", index=False)
        if not args.no_figures:
            reconstruction_figure(recon_df,
                                  outdir / "figures" / "threshold_shift.png")

    # ---------------- printed verdict ----------------
    print("\n" + "=" * 78)
    print("PAIRED MODEL-MEAN CONTRASTS (marginal 'ALL' rows; CI = 95% t; "
          "inference = CI + effect size, Holm on p_t)")
    print("NOTE: p_w (exact Wilcoxon, n=6) has a floor of 0.031 — reported "
          "raw, never Holm-corrected.")
    print("=" * 78)
    show_cols = ["family", "metric", "contrast", "mean_delta", "ci_lo",
                 "ci_hi", "cohen_dz", "p_t", "p_t_holm", "p_w", "n_models"]
    blocks = (
        ("PRIMARY discrimination (threshold-free, score-based)",
         PRIMARY_DISC_METRICS),
        ("operating-point metrics (threshold-coupled: move with bias even on "
         "a fixed ROC — do not read alone as discrimination)",
         OPERATING_POINT_METRICS),
        ("response bias", BIAS_METRICS),
    )
    for name, con, within in (("SCOPE", scope_con, "strategy"),
                              ("STRATEGY", strat_con, "scope")):
        marg = con[con[within] == "ALL"]
        for fam_name, mets in blocks:
            sub = marg[marg["metric"].isin(mets)]
            if sub.empty:
                continue
            print(f"\n-- {name} contrasts / {fam_name} --")
            with pd.option_context("display.width", 200,
                                   "display.float_format",
                                   lambda v: f"{v:.4f}"):
                print(sub[[c for c in show_cols if c in sub.columns]]
                      .to_string(index=False))

    if not mcnemar_df.empty:
        print("\n" + "=" * 78)
        print("McNEMAR (row-level) & GROUP-LEVEL WILCOXON — Holm-corrected "
              "significant fraction at alpha=0.05")
        print("=" * 78)
        print(f"  correctness moved (row McNemar) : "
              f"{_sig_summary(mcnemar_df, 'p_mcnemar_correct_holm')}"
              "   <- thesis expects LOW")
        print(f"  positivity moved  (row McNemar) : "
              f"{_sig_summary(mcnemar_df, 'p_mcnemar_positive_holm')}"
              "   <- thesis expects HIGH")
        print(f"  correctness moved (group level) : "
              f"{_sig_summary(mcnemar_df, 'p_group_correct_holm')}")
        print(f"  positivity moved  (group level) : "
              f"{_sig_summary(mcnemar_df, 'p_group_positive_holm')}")

    if not prev_df.empty:
        print("\n" + "=" * 78)
        print("REALISTIC-PREVALENCE REWEIGHTING (does the decomposition "
              "survive 1-5% prevalence?)")
        print("=" * 78)
        for pi, g in prev_df.groupby("prevalence"):
            ppr_rng = (g.groupby(["model", "strategy"])["ppr"]
                        .agg(lambda v: v.max() - v.min()))
            print(f"  prevalence={pi:.2f}: mean |MCC| = "
                  f"{g['mcc'].abs().mean():.4f}; "
                  f"mean PPR range across scopes = {ppr_rng.mean():.4f}")

    if not recon_df.empty:
        r = reconstruction_r2(recon_df)
        print("\n" + "=" * 78)
        print("THRESHOLD-SHIFT RECONSTRUCTION (function-scope scores -> other "
              "scopes' operating points)")
        print("=" * 78)
        print(f"  R²_TPR = {r['r2_tpr']:.4f}, R²_FPR = {r['r2_fpr']:.4f}, "
              f"RMSE = {r['rmse']:.4f} over {r['n_points']} rate points")
        print("  <- both R² near 1 = operating points slide along one ROC "
              "curve (pure bias shift)")
        n_unatt = int((np.abs(recon_df["ppr_achieved"]
                              - recon_df["ppr_target"]) > 0.02).sum())
        if n_unatt:
            print(f"  ({n_unatt}/{len(recon_df)} cells: score ties made the "
                  f"target PPR unattainable within 0.02 — see ppr_achieved)")

    print(f"\nSaved CSVs & figures -> {outdir}/")


if __name__ == "__main__":
    main()
