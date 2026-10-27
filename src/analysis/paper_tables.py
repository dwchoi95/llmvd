"""Paper tables, figures, and in-text numbers from the saved runs.

Reads data/FuncFileRepo.test.jsonl and results/{model}/{strategy}/*.jsonl and
reuses the metric and plotting code of expr.ipynb (cells 1, 2, 7-10).
Writes <out>/tables/*.tex, <out>/figures/results_{equivalence,cost,rq3cwe,rq3lang}.pdf, and
<out>/paper_numbers.json.

    python -m src.analysis.paper_tables [--out paper]
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = "FuncFileRepo.test"
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="paper", help="output root (tables/, figures/)")
args = ap.parse_args()
eval_rows = [json.loads(l) for l in open(ROOT / "data" / f"{BENCH}.jsonl")]
MODELS = ["qwen3-30b-instruct", "qwen3-30b-thinking", "mistral-small:24b",
          "magistral-small:24b", "nemotron-nano:12b", "nemotron-nano-think:12b"]

# ---- notebook cells ----------------------------------- #
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/llmvd-mpl")
import matplotlib
matplotlib.use("Agg")
nb = json.loads((ROOT / "expr.ipynb").read_text())
ns = {}
def cell(i):
    src = "".join(nb["cells"][i]["source"])
    exec(compile(src, f"expr.ipynb:cell{i}", "exec"), ns)
for i in (1, 2):
    cell(i)
import numpy as np
import pandas as pd
from scipy import stats as sps
df, trial_mean, metrics8 = ns["df"], ns["trial_mean"], ns["metrics8"]
SCOPES, STRATS, TIERS, REASONING = ns["SCOPES"], ns["STRATS"], ns["TIERS"], ns["REASONING"]
zr = df[df.prompts.isin(["zero", "rag"])]
OUTDIR = ROOT / args.out
TABLES = OUTDIR / "tables"; TABLES.mkdir(parents=True, exist_ok=True)
NUM = {}
KP = ("Acc", "Pre", "Rec", "F1", "PC", "PV", "PB", "PR")
RANKED = ("Acc", "Pre", "Rec", "F1", "PC")
f3 = lambda x: f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else f"{x:.3f}"
def sg(x, nd=3):
    """signed, leading-zero-free, math minus: +.025 / $-$.022"""
    t = f"{abs(x):.{nd}f}".replace("0.", ".", 1)
    return ("$-$" if x < 0 else "+") + t

def bold_rows(mus):
    """mus: {row_key: mu}; bold the max (3-dp ties too) of each ranked column."""
    best = {k: max(round(mu[k], 3) for mu in mus.values()) for k in RANKED}
    return {rk: [(r"\textbf{%s}" % f3(mu[k]) if k in RANKED and round(mu[k], 3) == best[k] else f3(mu[k]))
                 for k in KP] for rk, mu in mus.items()}

def grid_table(name, caption, label, groups, sel, sep, head):
    """Table IV/V: group (scope|strategy) x tier, bold best group per tier."""
    cells = {(g, t): trial_mean(sel(g) if t == "mean" else sel(g)[lambda d: d.complexity == t])[0]
             for g, _ in groups for t in ["mean"] + TIERS}
    for t in ["mean"] + TIERS:
        b = bold_rows({g: cells[(g, t)] for g, _ in groups})
        for g, _ in groups:
            cells[(g, t, "fmt")] = b[g]
    body = []
    for gi, (g, short) in enumerate(groups):
        stack = r"\shortstack{" + r"\\".join(short) + "}"
        for ti, t in enumerate(["mean"] + TIERS):
            lead = (r"\multirow{4}{*}{%s}" % stack) if ti == 0 else ""
            tier = r"\textbf{mean}" if t == "mean" else t
            body.append(f"    {lead} & {tier} & " + " & ".join(cells[(g, t, 'fmt')]) + r" \\")
        if gi < len(groups) - 1:
            body.append(r"    \midrule")
    text = "\n".join([
        r"\begin{table}[h]", r"  \centering", rf"  \caption{{{caption}}}", rf"  \label{{{label}}}",
        r"  \small", rf"  \setlength{{\tabcolsep}}{{{sep}}}", r"  \begin{tabular}{@{}cl *{8}{c}@{}}",
        r"    \toprule",
        r"    & & \multicolumn{4}{c}{\emph{Classification}} & \multicolumn{4}{c}{\emph{Pair-wise}} \\",
        r"    \cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        f"    {head} & Tier & Acc & Pre & Rec & $F_1$ & P-C & P-V & P-B & P-R \\\\",
        r"    \midrule", *body, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    (TABLES / name).write_text(text)
    return cells

# ---- Table I: dataset statistics ---------------------------------------- #
cwes = {c for r in eval_rows for c in (r.get("cwe_id") or [])}
lang = defaultdict(int)
for r in eval_rows:
    lang["C/C++" if r["language"] in ("c", "c++") else r["language"].capitalize()] += 1
tiers = defaultdict(int)
for r in eval_rows:
    tiers[r["tag"]] += 1
NUM["dataset"] = dict(pairs=len(eval_rows) // 2, functions=len(eval_rows),
                      projects=len({r["project"] for r in eval_rows}), cves=len({r["cve_id"] for r in eval_rows}),
                      cwe_labels=len(cwes), numbered_cwes=len([c for c in cwes if c.startswith("CWE-")]),
                      noinfo_functions=sum("NVD-CWE-noinfo" in (r.get("cwe_id") or []) for r in eval_rows),
                      c=sum(r["language"] == "c" for r in eval_rows), cpp=sum(r["language"] == "c++" for r in eval_rows),
                      lang=dict(lang), tiers=dict(tiers))
d = NUM["dataset"]
folds = [[("Vulnerable", d["pairs"]), ("Non-vulnerable", d["pairs"])],
         [(L, lang[L]) for L in ("C/C++", "Java", "Python")],
         [(t, tiers[t]) for t in ("SFSF", "MFSF", "MFMF")]]
height = max(map(len, folds))
cellrow = lambda i: " & ".join(f"{f[i][0]} & {f[i][1]:,}" if i < len(f) else " & " for f in folds) + r" \\"
(TABLES / "dataset_statistics.tex").write_text("\n".join([
    r"\begin{table}[h]", r"  \centering",
    rf"  \caption{{Statistics of the evaluation dataset ({d['pairs']:,} pairs from {d['projects']:,} projects and {d['cves']:,} CVEs)}}",
    r"  \label{tab:dataset_statistics}", r"  \small", r"  \begin{tabular}{@{}lr||lr||lr@{}}", r"  \toprule",
    r"  \textbf{Label} & \textbf{\#} & \textbf{Language} & \textbf{\#} & \textbf{Tier} & \textbf{\#} \\", r"  \midrule",
    *[f"  {cellrow(i)}" for i in range(height)], r"  \midrule",
    "  " + " & ".join(f"Total & {sum(n for _, n in f):,}" for f in folds) + r" \\",
    r"  \bottomrule", r"  \end{tabular}", r"\end{table}", ""]))

# ---- Tables IV, V, VI ----------------------------------------------------- #
tab4 = grid_table("rq1_scope.tex", r"Detection results by input scope and complexity tier",
                  "tab:rq1", [("function", "Func"), ("file", "File"), ("repository", "Repo")], lambda g: df[df.scope == g], "3.0pt", "Scope")
tab5 = grid_table("rq1_strategy.tex", r"Detection results by prompting strategy and complexity tier",
                  "tab:rq1strat", [("zero", "Zero"), ("rag", "RAG"), ("sft", "SFT")], lambda g: df[df.prompts == g], "2.4pt", "Strategy")
NAMES = ["Qwen3", r"Qwen3$^\dagger$", "Mistral", r"Magistral$^\dagger$", "Nemotron", r"Nemotron$^\dagger$"]
mm = {m: trial_mean(zr[zr.model == m])[0] for m in MODELS}
grp = {"Non-reasoning": trial_mean(zr[~zr.model.isin(REASONING)])[0], "Reasoning": trial_mean(zr[zr.model.isin(REASONING)])[0]}
b = bold_rows(mm)
lines = [f"    {n} & " + " & ".join(b[m]) + r" \\" for m, n in zip(MODELS, NAMES)]
lines += [r"    \midrule"] + [rf"    \emph{{{g}}} & " + " & ".join(f3(mu[k]) for k in KP) + r" \\" for g, mu in grp.items()]
(TABLES / "rq3_model.tex").write_text("\n".join([
    r"\begin{table}[h]", r"  \centering",
    r"  \caption{Detection results by model with zero-shot prompting and RAG pooled}",
    r"  \label{tab:rq2}", r"  \small", r"  \setlength{\tabcolsep}{3.4pt}", r"  \begin{tabular}{@{}l *{8}{c}@{}}", r"    \toprule",
    r"    & \multicolumn{4}{c}{\emph{Classification}} & \multicolumn{4}{c}{\emph{Pair-wise}} \\",
    r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"    Model & Acc & Pre & Rec & $F_1$ & P-C & P-V & P-B & P-R \\", r"    \midrule",
    *lines, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]))
NUM["tab4"] = {f"{g}|{t}": {k: round(v, 4) for k, v in tab4[(g, t)].items()} for g in SCOPES for t in ["mean"] + TIERS}
NUM["tab5"] = {f"{g}|{t}": {k: round(v, 4) for k, v in tab5[(g, t)].items()} for g in STRATS for t in ["mean"] + TIERS}
NUM["tab6"] = {m: {k: round(v, 4) for k, v in mu.items()} for m, mu in {**mm, **grp}.items()}

# ---- small-table helper (same layout as revision_tables.py) ------------- #
def table(file, caption, label, columns, header, rows):
    """columns: tabular spec incl. || and |; header: LaTeX header row(s); a row "midrule" draws a group line."""
    text = r"\begin{table}[h]" + "\n" + rf"\centering\caption{{{caption}}}\label{{{label}}}" + "\n"
    text += r"\small\setlength{\tabcolsep}{3pt}" + "\n" + rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}" + "\n\\toprule\n"
    text += header + " \\\\\n\\midrule\n" + "\n".join(r"\midrule" if r == "midrule" else " & ".join(r) + r" \\" for r in rows)
    (TABLES / file).write_text(text + "\n\\bottomrule\n\\end{tabular}\n" + r"\end{table}" + "\n")

def head2(label, groups):
    """2-level header: groups = [(group title, [sub-columns])]; the last group gets no right rule."""
    n = sum(len(c) for _, c in groups)
    top = " & ".join(rf"\multicolumn{{{len(c)}}}{{c{'|' if i < len(groups) - 1 else ''}}}{{\textbf{{{g}}}}}" if len(c) > 1
                     else rf"\multicolumn{{1}}{{c{'|' if i < len(groups) - 1 else ''}}}{{\textbf{{{g}}}}}" for i, (g, c) in enumerate(groups))
    sub = " & ".join(x for _, c in groups for x in c)
    return (rf"\multirow{{2}}{{*}}{{\textbf{{{label}}}}} & {top} \\" + "\n" + rf"\cline{{2-{n + 1}}}" + "\n"
            + rf"\rule{{0pt}}{{2.6ex}} & {sub}")
fp = lambda p: "$<$.001" if p < .001 else f"{p:.3f}".replace("0.", ".", 1)

# ---- RQ1 statistics (cell 4 / 6 logic) ----------------------------------- #
MS = {(m, sc): trial_mean(zr[(zr.model == m) & (zr.scope == sc)])[0] for m in MODELS for sc in SCOPES}
MT = {(m, s): trial_mean(df[(df.model == m) & (df.prompts == s)])[0] for m in MODELS for s in STRATS}
def paired(a, b):
    d = np.array(b) - np.array(a)
    return dict(delta=d.mean(), p=sps.ttest_rel(b, a).pvalue, dz=d.mean() / d.std(ddof=1))
NUM["rq1_scope_tests"] = {f"{k}|{a}->{b2}": paired([MS[(m, a)][k] for m in MODELS], [MS[(m, b2)][k] for m in MODELS])
                          for k in ("PC", "PV", "PB", "Rec", "F1")
                          for a, b2 in [("function", "file"), ("function", "repository"), ("file", "repository")]}
NUM["rq1_strategy_tests"] = {f"{k}|{a}->{b2}": paired([MT[(m, a)][k] for m in MODELS], [MT[(m, b2)][k] for m in MODELS])
                             for k in ("PC", "PV", "PB") for a, b2 in [("zero", "rag"), ("zero", "sft"), ("rag", "sft")]}
stds = []
for sc in SCOPES:
    for t in ["mean"] + TIERS:
        sub = zr[zr.scope == sc] if t == "mean" else zr[(zr.scope == sc) & (zr.complexity == t)]
        stds += list(trial_mean(sub)[1].values())
NUM["trial_std"] = dict(median=float(np.median(stds)), max=float(max(stds)))

def contrast(d, margin):
    d = np.asarray(d, float); n = len(d); se = sps.sem(d)
    ci = sps.t.interval(.90, n - 1, loc=d.mean(), scale=se)
    p_lo = sps.t.sf((d.mean() + margin) / se, n - 1); p_hi = sps.t.cdf((d.mean() - margin) / se, n - 1)
    return dict(delta=d.mean(), lo=ci[0], hi=ci[1], p=max(p_lo, p_hi), p_t=sps.ttest_1samp(d, 0).pvalue)
eq = []
for a, b2, nm in [("function", "file", r"Func.$\to$File"), ("function", "repository", r"Func.$\to$Repo."),
                  ("file", "repository", r"File$\to$Repo.")]:
    eq.append((nm, .03, contrast([MS[(m, b2)]["PC"] - MS[(m, a)]["PC"] for m in MODELS], .03)))
for a, b2, nm in [("zero", "rag", r"Zero$\to$RAG"), ("zero", "sft", r"Zero$\to$SFT"), ("rag", "sft", r"RAG$\to$SFT")]:
    eq.append((nm, .05, contrast([MT[(m, b2)]["PC"] - MT[(m, a)]["PC"] for m in MODELS], .05)))
# forest plot: mean paired difference, 90% CI, margin band; filled = equivalent at its margin
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
FIGS = OUTDIR / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 8, "pdf.fonttype": 42}):
    fig, ax = plt.subplots(figsize=(3.0, 1.85))
    right = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (nm, mg, r) in enumerate(eq):
        y = len(eq) - 1 - i
        ax.fill_betweenx([y - .38, y + .38], -mg, mg, color="0.87", lw=0)
        ax.plot([r["lo"], r["hi"]], [y, y], color="k", lw=1)
        ax.plot(r["delta"], y, "o", ms=4, mec="k", mew=.9, mfc="k" if r["p"] < .05 else "w")
        ax.text(1.03, y, fp(r["p"]), transform=right, va="center", fontsize=7)
    ax.text(1.03, len(eq) - .45, r"$p_{\rm TOST}$", transform=right, va="bottom", fontsize=7)
    ax.axvline(0, color="k", lw=.5, ls=":"); ax.axhline(2.5, color="0.6", lw=.5)
    ax.set_yticks(range(len(eq))); ax.set_yticklabels([nm for nm, _, _ in eq][::-1])
    ax.set_ylim(-.6, len(eq) - .4); ax.set_xlim(-.1, .11)
    ax.set_xticks([-.1, -.05, 0, .05, .1]); ax.set_xticklabels(["$-$.10", "$-$.05", "0", ".05", ".10"])
    ax.set_xlabel(r"P-C difference $\Delta$ (90% CI)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(FIGS / "results_equivalence.pdf", bbox_inches="tight"); plt.close(fig)
NUM["tost"] = {nm: r for nm, _, r in eq}

table("model_contrasts.tex", "Per-model changes in P-C and signed separation $D$",
      "tab:modelcontrasts", "l||rrr|r", head2("Model", [("P-C", [r"Func.$\to$File", r"Func.$\to$Repo.", r"Zero$\to$RAG"]), ("$D$", [r"Zero$\to$RAG"])]),
      [[n, sg(MS[(m, 'file')]['PC'] - MS[(m, 'function')]['PC']), sg(MS[(m, 'repository')]['PC'] - MS[(m, 'function')]['PC']),
        sg(MT[(m, 'rag')]['PC'] - MT[(m, 'zero')]['PC']), sg((MT[(m, 'rag')]['PC'] - MT[(m, 'rag')]['PR']) - (MT[(m, 'zero')]['PC'] - MT[(m, 'zero')]['PR']))] for m, n in zip(MODELS, NAMES)])
NUM["model_contrasts"] = {m: dict(file=MS[(m, "file")]["PC"] - MS[(m, "function")]["PC"],
                                  repo=MS[(m, "repository")]["PC"] - MS[(m, "function")]["PC"],
                                  rag=MT[(m, "rag")]["PC"] - MT[(m, "zero")]["PC"], rag_D=(MT[(m, "rag")]["PC"] - MT[(m, "rag")]["PR"]) - (MT[(m, "zero")]["PC"] - MT[(m, "zero")]["PR"])) for m in MODELS}

# ---- reasoning gains: cell-6 cluster bootstrap, same seed ------------------ #
def pair_outcomes(m, strats=("zero", "rag"), scope=None):
    out = defaultdict(lambda: [0, 0, 0])   # [P-C, pairs, P-R]
    sub = df[(df.model == m) & (df.prompts.isin(strats)) & ((df.scope == scope) if scope else True)].dropna(subset=["predict"])
    for (t, s, sc, p), g in sub.groupby(["trial", "prompts", "scope", "pair"]):
        lab = dict(zip(g["label"].astype(bool), g["predict"].astype(bool)))
        if True in lab and False in lab:
            out[p][1] += 1; out[p][0] += int(lab[True] and not lab[False])
            out[p][2] += int(lab[False] and not lab[True])
    return out
PO = {m: pair_outcomes(m) for m in MODELS}
pairs = sorted(set(df["pair"])); random.seed(42)
gains = []
for fam, (nr, re_) in {"Qwen": MODELS[0:2], "Mistral": MODELS[2:4], "Nemotron": MODELS[4:6]}.items():
    boot, bootd = [], []
    for _ in range(2000):
        samp = [random.choice(pairs) for _ in pairs]
        nb_ = max(sum(PO[nr][p][1] for p in samp), 1); nr_ = max(sum(PO[re_][p][1] for p in samp), 1)
        bb = sum(PO[nr][p][0] for p in samp) / nb_
        rr = sum(PO[re_][p][0] for p in samp) / nr_
        boot.append(rr - bb)
        bootd.append((rr - sum(PO[re_][p][2] for p in samp) / nr_) - (bb - sum(PO[nr][p][2] for p in samp) / nb_))
    boot.sort(); bootd.sort()
    dpc = mm[re_]["PC"] - mm[nr]["PC"]
    dd = (mm[re_]["PC"] - mm[re_]["PR"]) - (mm[nr]["PC"] - mm[nr]["PR"])
    gains.append((fam, dpc, boot[49], boot[1949], dd, bootd[49], bootd[1949]))
table("reasoning_gains.tex", r"Reasoning-variant minus non-reasoning P-C and signed separation $D$",
      "tab:reasoninggain", "l||rr|rr", head2("Family", [("P-C", [r"$\Delta$", r"95\% CI"]), ("$D$", [r"$\Delta$", r"95\% CI"])]),
      [[f, sg(d1), f"[{sg(lo)}, {sg(hi)}]", sg(d2), f"[{sg(dlo)}, {sg(dhi)}]"] for f, d1, lo, hi, d2, dlo, dhi in gains])
NUM["reasoning_gains"] = {f: dict(dPC=d1, lo=lo, hi=hi, dD=d2, dlo=dlo, dhi=dhi) for f, d1, lo, hi, d2, dlo, dhi in gains}
NUM["groups"] = {g: dict(PC=mu["PC"], D=mu["PC"] - mu["PR"]) for g, mu in grp.items()}
cellbest = max([((m, s, sc), trial_mean(df[(df.model == m) & (df.prompts == s) & (df.scope == sc)])[0]["PC"])
                for m in MODELS for s in ("zero", "rag") for sc in SCOPES], key=lambda x: x[1])
NUM["best_cell"] = dict(cell=cellbest[0], PC=cellbest[1])

# ---- per-model D contrasts against zero-shot (RAG, SFT): same bootstrap ---- #
def boot_dD(a, b):
    """95% pair-cluster bootstrap interval of D(b) - D(a) over pair-outcome dicts a, b."""
    random.seed(42); out = []
    d = lambda o, S: (sum(o[p][0] for p in S) - sum(o[p][2] for p in S)) / max(sum(o[p][1] for p in S), 1)
    for _ in range(2000):
        samp = [random.choice(pairs) for _ in pairs]; out.append(d(b, samp) - d(a, samp))
    out.sort(); return out[49], out[1949]
Dmt = lambda m, s, sc=None: (lambda mu: mu["PC"] - mu["PR"])(MT[(m, s)] if sc is None else
                               trial_mean(df[(df.model == m) & (df.prompts == s) & (df.scope == sc)])[0])
NUM["dD_vs_zero"] = {}
for m in MODELS:
    z = pair_outcomes(m, ("zero",))
    NUM["dD_vs_zero"][m] = {
        "rag": (Dmt(m, "rag") - Dmt(m, "zero"), *boot_dD(z, pair_outcomes(m, ("rag",)))),
        "sft": (Dmt(m, "sft") - Dmt(m, "zero"), *boot_dD(z, pair_outcomes(m, ("sft",)))),
        "sft_function": (Dmt(m, "sft", "function") - Dmt(m, "zero", "function"),
                         *boot_dD(pair_outcomes(m, ("zero",), "function"), pair_outcomes(m, ("sft",), "function"))),
        "zero_D": Dmt(m, "zero"), "sft_D": Dmt(m, "sft"),
        "zero_D_function": Dmt(m, "zero", "function"), "sft_D_function": Dmt(m, "sft", "function")}

# ---- SFT prediction distributions ----------------------------------------- #
rows = []
NUM["sft"] = {}
for m, n in zip(MODELS, NAMES):
    sub = df[(df.prompts == "sft") & (df.model == m)].dropna(subset=["predict"])
    if not len(sub):
        continue
    per = [sub[sub.trial == t] for t in sorted(sub.trial.unique())]
    q = np.mean([g.predict.astype(int).mean() for g in per])
    tpr = np.mean([g[g.label.astype(bool)].predict.astype(int).mean() for g in per])
    fpr = np.mean([g[~g.label.astype(bool)].predict.astype(int).mean() for g in per])
    mu = MT[(m, "sft")]
    rows.append([n] + [f3(x) for x in (q, tpr, fpr, mu["PC"], mu["PR"])] + [sg(Dmt(m, "zero")), sg(mu["PC"] - mu["PR"])])
    NUM["sft"][m] = dict(q=q, TPR=tpr, FPR=fpr, PC=mu["PC"], PR=mu["PR"], D=mu["PC"] - mu["PR"])
table("sft_predictions.tex", r"SFT prediction distributions",
      "tab:sftpred", "l||rrr|rr|rr", head2("Model", [("Classification", ["$q$", "TPR", "FPR"]), ("Pair-wise", ["P-C", "P-R"]),
                                                     ("$D$", ["Base", "SFT"])]), rows)

# ---- tokens / response time ------------------------------------------------ #
eff = df.groupby(["prompts", "scope"])[["tokens", "time(s)"]].mean()
df["reasoning"] = df.model.isin(REASONING)
rt = df.groupby(["reasoning", "prompts", "scope"])["time(s)"].mean().groupby(["reasoning", "prompts"]).mean()
# every (model group, strategy, scope) cell on the token-time plane, unpooled; lines join the three scopes
cost = df.groupby(["reasoning", "prompts", "scope"])[["tokens", "time(s)"]].mean()
from matplotlib.lines import Line2D
with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42}):
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    SC = ["function", "repository", "file"]   # left to right
    MK = {"zero": ("o", "-"), "rag": ("s", "--"), "sft": ("^", ":")}
    for r, fc in [(False, "w"), (True, "k")]:
        for s_, (mk, ls) in MK.items():
            xs = [cost.loc[(r, s_, sc), "tokens"] for sc in SC]; ys = [cost.loc[(r, s_, sc), "time(s)"] for sc in SC]
            ax.plot(xs, ys, ls=ls, color="0.45", lw=.8, zorder=1)
            ax.plot(xs, ys, mk, ls="", ms=4.5, mfc=fc, mec="k", mew=.8, zorder=2)
    from matplotlib.patches import Patch
    st = [Line2D([], [], marker=MK[s_][0], ls=MK[s_][1], color="0.45", mfc="w", mec="k") for s_ in MK]
    gr = [Patch(fc=fc, ec="k", lw=.8) for fc in ("w", "k")] + [Line2D([], [], ls="")]
    # column-major fill: first row strategies, second row model groups
    ax.legend([x for pair in zip(st, gr) for x in pair], [x for pair in zip(["Zero-shot", "RAG", "SFT"], ["Non-reasoning", "Reasoning", ""]) for x in pair],
              loc="center left", bbox_to_anchor=(0, .58), fontsize=6.5, frameon=False, ncol=3, handlelength=2.2, columnspacing=1)
    ax.set_xlim(0, 24000); ax.set_xticks(range(0, 24001, 3000)); ax.set_xticklabels([f"{k}K" if k else "0" for k in range(0, 25, 3)])
    ax.set_ylim(0, 60); ax.set_xlabel("Mean estimated prompt tokens"); ax.set_ylabel("Mean response time (s)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(FIGS / "results_cost.pdf", bbox_inches="tight"); plt.close(fig)
NUM["cost_cells"] = {f"{'reason' if r else 'direct'}|{s_}|{sc}": dict(tokens=float(cost.loc[(r, s_, sc), "tokens"]), time=float(cost.loc[(r, s_, sc), "time(s)"]))
                     for r in (False, True) for s_ in STRATS for sc in SCOPES}
NUM["tokens"] = {f"{s}|{sc}": float(eff.loc[(s, sc), "tokens"]) for s in STRATS for sc in SCOPES}
NUM["response_time"] = {f"{'reason' if r else 'direct'}|{s}": float(rt.loc[(r, s)]) for r in (False, True) for s in STRATS}

# ---- coverage ------------------------------------------------------------- #
NUM["records"] = int(len(df)); NUM["missing"] = int(df.predict.isna().sum())
NUM["missing_by_model"] = {m: float(df[df.model == m].predict.isna().mean()) for m in MODELS}
pp = df.dropna(subset=["predict"]).groupby(["model", "prompts", "trial", "scope", "pair"]).size()
NUM["complete_pairs"] = int((pp == 2).sum()); NUM["possible_pairs"] = int(len(df) // 2)

# ---- RQ3: language / CWE slices + figures (cells 7-10) -------------------- #
cell(7)
zr = df[df.prompts.isin(["zero", "rag"])]          # re-derive: cell 7 added cwe/lang3 columns
lang_pc = ns["lang_pc"]
ns_lang = {L: trial_mean(zr[zr.lang3 == L])[0]["PC"] for L in ns["langs"]}
NUM["lang_pc_zr"] = ns_lang
NUM["lang_pairs"] = {L: int(df[df.lang3 == L].pair.nunique()) for L in ns["langs"]}
spread = {L: max(lang_pc[(L, sc, s)] for sc in SCOPES for s in ("zero", "rag")) - min(lang_pc[(L, sc, s)] for sc in SCOPES for s in ("zero", "rag")) for L in ns["langs"]}
NUM["lang_spread_zr"] = spread
NUM["java_highest_cells"] = sum(max(ns["langs"], key=lambda L: lang_pc[(L, sc, s)]) == "Java" for sc in SCOPES for s in STRATS)
NUM["cc_below_java_models"] = sum(trial_mean(zr[(zr.model == m) & (zr.lang3 == "C/C++")])[0]["PC"] <
                                  trial_mean(zr[(zr.model == m) & (zr.lang3 == "Java")])[0]["PC"] for m in MODELS)
heat = ns["heat_scope"]; CWES = ns["CWES"]
cwe_mean = {c: float(np.nanmean([trial_mean(zr[(zr.model == m) & (zr.cwe == c)])[0]["PC"] for m in MODELS])) for c in CWES}
NUM["cwe"] = dict(n=len(CWES), counts={c: len(ns["cnt"][c]) for c in CWES}, mean_pc=cwe_mean,
                  best_model_reasoning=sum(max(MODELS, key=lambda m: trial_mean(zr[(zr.model == m) & (zr.cwe == c)])[0]["PC"]) in REASONING for c in CWES),
                  spearman_file=float(sps.spearmanr(heat["function"].mean(), heat["file"].mean()).correlation),
                  spearman_repo=float(sps.spearmanr(heat["function"].mean(), heat["repository"].mean()).correlation),
                  multi_label_pairs=sum(len(r.get("cwe_id") or []) > 1 for r in eval_rows if r["vulnerable"]),
                  first_label_categories=len(ns["cnt"]))
tier_pc = {t: trial_mean(zr[zr.complexity == t])[0]["PC"] for t in TIERS}
comp = {t: {L: float((df[df.complexity == t].drop_duplicates("pair").lang3 == L).mean()) for L in ns["langs"]} for t in TIERS}
NUM["tiers"] = dict(pc_zr=tier_pc, composition=comp)
segs = {**{f"tier:{t}": zr[zr.complexity == t] for t in TIERS}, **{f"lang:{L}": zr[zr.lang3 == L] for L in ns["langs"]},
        **{f"cwe:{c}": zr[zr.cwe == c] for c in CWES if len(ns["cnt"][c]) >= 20}}
lead = {}
for k, sub in segs.items():
    v = sorted(((trial_mean(sub[sub.scope == sc])[0]["PC"], sc) for sc in SCOPES), reverse=True)
    lead[k] = dict(best=v[0][1], gap=v[0][0] - v[1][0])
NUM["segments"] = lead
cell(8)
ns["FIGDIR"] = OUTDIR / "figures"; ns["FIGDIR"].mkdir(parents=True, exist_ok=True)
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})   # IEEE: raster parts >= 300 dpi
cell(9); cell(10)

(OUTDIR / "paper_numbers.json").write_text(json.dumps(NUM, indent=1, default=float) + "\n")
print(f"wrote {OUTDIR}/tables, {OUTDIR}/figures, {OUTDIR}/paper_numbers.json")
