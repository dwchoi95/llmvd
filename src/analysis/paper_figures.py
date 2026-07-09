"""Generate paper tables (LaTeX) and figures from the new 5-model results.

Reads results/{model}/FuncFileRepo.eval.jsonl for the completed models and
writes:
  - paper/figures/tradeoff.png            (overall scope F1 vs time, bubble=tokens)
  - paper/figures/interaction.png         (scope x tag F1 -- the new contribution)
  - paper/figures/model_tradeoff.png      (per-model F1 vs time by scope)
  - paper/figures/language_tradeoff.png   (per-language F1 vs time by scope)
  - paper/generated_tables.tex            (Table 2, per-tag table, CWE table)
"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCOPES = ["function", "file", "repository"]
SCOPE_LABEL = {"function": "Function", "file": "File", "repository": "Repository"}
TAGS = ["SFSF", "MFSF", "MFMF"]
FIGDIR = "paper/figures"
OUTTEX = "paper/generated_tables.tex"

# Models excluded from the aggregate panel (none: all 8 candidate models reported).
EXCLUDE_MODELS = set()
# A model must cover at least this fraction of the 3843 (sample x scope) cells
# to enter the panel (guards against any other partially-run model).
MIN_COVERAGE = 0.95
N_CELLS = 3843

# 2025 CWE Top-25 Most Dangerous Software Weaknesses (IDs)
CWE_TOP25 = {"CWE-79","CWE-787","CWE-89","CWE-352","CWE-22","CWE-125","CWE-78","CWE-416",
             "CWE-862","CWE-434","CWE-94","CWE-20","CWE-77","CWE-287","CWE-269","CWE-502",
             "CWE-200","CWE-863","CWE-918","CWE-119","CWE-476","CWE-798","CWE-190","CWE-400","CWE-306"}
CWE_NAME = {
    "CWE-79":"Cross-site Scripting","CWE-787":"Out-of-bounds Write","CWE-89":"SQL Injection",
    "CWE-22":"Path Traversal","CWE-125":"Out-of-bounds Read","CWE-78":"OS Command Injection",
    "CWE-416":"Use After Free","CWE-94":"Code Injection","CWE-20":"Improper Input Validation",
    "CWE-77":"Command Injection","CWE-287":"Improper Authentication","CWE-502":"Deserialization of Untrusted Data",
    "CWE-200":"Exposure of Sensitive Information","CWE-119":"Improper Restriction of Memory Buffer",
    "CWE-476":"NULL Pointer Dereference","CWE-798":"Use of Hard-coded Credentials","CWE-190":"Integer Overflow",
    "CWE-400":"Uncontrolled Resource Consumption","CWE-306":"Missing Authentication","CWE-862":"Missing Authorization",
    "CWE-269":"Improper Privilege Management","CWE-863":"Incorrect Authorization","CWE-918":"Server-Side Request Forgery",
    "CWE-352":"Cross-Site Request Forgery","CWE-434":"Unrestricted Upload of File",
}


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return True if v == 1 else (False if v == 0 else None)
    if isinstance(v, str):
        s = v.strip().lower()
        return True if s in ("true", "1") else (False if s in ("false", "0") else None)
    return None


def load_all():
    frames = []
    for path in sorted(glob.glob("results/*/FuncFileRepo.eval.jsonl")):
        model = path.split("/")[1]
        if model in EXCLUDE_MODELS:
            continue
        df = pd.read_json(path, lines=True)
        scored = int(df["predict"].notna().sum())
        if scored < MIN_COVERAGE * N_CELLS:
            print(f"  skip {model}: only {scored}/{N_CELLS} scored (partial)")
            continue
        df["model"] = model
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["pred"] = df["predict"].apply(to_bool)
    df["truth"] = df["vulnerable"].apply(to_bool)
    df = df.dropna(subset=["pred", "truth"]).copy()
    df["correct"] = (df["pred"] == df["truth"]).astype(int)
    return df


def metrics(g):
    p = g["pred"].to_numpy(bool); t = g["truth"].to_numpy(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    acc = (tp+tn)/(tp+fp+fn+tn) if (tp+fp+fn+tn) else 0.0
    return pd.Series({"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1,
                      "tokens":g["tokens"].mean(),"time":g["Time (sec)"].mean()})


def per_model_scope(df):
    return df.groupby(["model","scope"]).apply(metrics).reset_index()


def mcnemar_p(df, a, b, tagmask=None):
    sub = df if tagmask is None else df[df["tag"] == tagmask]
    wide = sub.pivot_table(index=["model","group_id","file_name","vulnerable"],
                           columns="scope", values="correct", aggfunc="first")
    if a not in wide or b not in wide:
        return float("nan"), 0
    w = wide[[a, b]].dropna()
    bb = int(((w[a]==1)&(w[b]==0)).sum()); cc = int(((w[a]==0)&(w[b]==1)).sum())
    disc = bb+cc
    if not disc:
        return 1.0, 0
    stat = (abs(bb-cc)-1)**2/disc
    return math.erfc(math.sqrt(stat/2.0)), (bb-cc)


def cohens_d_paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    sd = d.std(ddof=1)
    return float(d.mean()/sd) if sd else 0.0


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = load_all()
    models = sorted(df["model"].unique())
    print("models:", models, "rows:", len(df))

    pm = per_model_scope(df)                       # per (model, scope)
    _num = ["Accuracy","Precision","Recall","F1","tokens","time"]
    agg = pm.groupby("scope")[_num].agg(["mean","std"])  # across models
    tex = []

    # ---- Table 2: overall performance across scopes -----------------------
    def cell(scope, m):
        return f"{agg.loc[scope,(m,'mean')]:.3f} $\\pm$ {agg.loc[scope,(m,'std')]:.3f}"
    rows = []
    for s in SCOPES:
        rows.append(f"{SCOPE_LABEL[s]} & {cell(s,'Accuracy')} & {cell(s,'Precision')} & "
                    f"{cell(s,'Recall')} & {cell(s,'F1')} & {agg.loc[s,('tokens','mean')]:.0f} & "
                    f"{agg.loc[s,('time','mean')]:.2f} \\\\")
    # pairwise McNemar + Cohen's d on per-model F1
    from scipy import stats
    pivotF1 = pm.pivot(index="model", columns="scope", values="F1")
    pair_rows = []
    for a, b in [("function","file"),("function","repository"),("file","repository")]:
        t, p = stats.ttest_rel(pivotF1[a], pivotF1[b])
        d = cohens_d_paired(pivotF1[a], pivotF1[b])
        dF1 = (pivotF1[a]-pivotF1[b]).mean()
        star = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
        pair_rows.append(f"{SCOPE_LABEL[a]} vs {SCOPE_LABEL[b]} & {dF1:+.3f}{star} & {p:.3f} & {d:+.2f} \\\\")
    tex.append("% ===== Table 2: overall performance =====\n"
        "\\begin{table}[t]\\centering\\caption{Performance Comparison Across Input Scope Levels "
        f"(mean $\\pm$ SD over {len(models)} models)}}\\label{{tab:performance}}\\scriptsize\n"
        "\\begin{tabular}{l|cccc|cc}\n\\toprule\n"
        "\\textbf{Scope} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-score} & \\textbf{AVG Tokens} & \\textbf{AVG Time (s)} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\\\[4pt]\n\\begin{tabular}{l|ccc}\n\\toprule\n"
        "\\textbf{Pair (F1)} & \\textbf{$\\Delta$F1} & \\textbf{McNemar $p$} & \\textbf{Cohen's $d$} \\\\\n\\midrule\n"
        + "\n".join(pair_rows) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\end{table}\n")

    # ---- Per-tag (scope x tag) table + interaction ------------------------
    pt = df.groupby(["model","scope","tag"]).apply(metrics).reset_index()
    tagF1 = pt.groupby(["tag","scope"])["F1"].agg(["mean","std"]).reset_index()
    def f1c(tag, s):
        r = tagF1[(tagF1.tag==tag)&(tagF1.scope==s)].iloc[0]
        return f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}"
    trows = []
    for tag in TAGS:
        trows.append(f"{tag} & {f1c(tag,'function')} & {f1c(tag,'file')} & {f1c(tag,'repository')} \\\\")
    tex.append("% ===== per-tag scope x complexity F1 =====\n"
        "\\begin{table}[t]\\centering\\caption{F1-score by Input Scope and Vulnerability Complexity Tier "
        f"(mean $\\pm$ SD over {len(models)} models)}}\\label{{tab:per_tag}}\\scriptsize\n"
        "\\begin{tabular}{l|ccc}\n\\toprule\n"
        "\\textbf{Tier} & \\textbf{Function} & \\textbf{File} & \\textbf{Repository} \\\\\n\\midrule\n"
        + "\n".join(trows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ---- CWE Top-25 table -------------------------------------------------
    ex = df.explode("cwe_id")
    cwe_rows = []
    for cwe in sorted(CWE_TOP25):
        sub = ex[ex["cwe_id"] == cwe]
        if sub["group_id"].nunique() < 2:
            continue
        f1s = {s: metrics(sub[sub.scope==s])["F1"] for s in SCOPES}
        best = max(SCOPES, key=lambda s: f1s[s])
        name = CWE_NAME.get(cwe, cwe)
        cells = " & ".join((f"\\textbf{{{f1s[s]:.3f}}}" if s==best else f"{f1s[s]:.3f}") for s in SCOPES)
        cwe_rows.append((cwe, name, cells, f1s))
    body = "\n".join(f"{c} & {n} & {cells} \\\\" for c,n,cells,_ in cwe_rows)
    means = {s: np.mean([r[3][s] for r in cwe_rows]) for s in SCOPES}
    tex.append("% ===== CWE Top-25 table =====\n"
        "\\begin{table}[t]\\centering\\caption{Detection F1 by Scope for 2025 CWE Top-25 Weaknesses present in the dataset}"
        "\\label{tab:cwe_performance}\\scriptsize\n\\begin{tabular}{llccc}\n\\toprule\n"
        "\\textbf{CWE} & \\textbf{Name} & \\textbf{Func.} & \\textbf{File} & \\textbf{Repo.} \\\\\n\\midrule\n"
        + body + "\n\\midrule\n"
        + f"\\multicolumn{{2}}{{l}}{{\\textbf{{Overall}}}} & {means['function']:.3f} & {means['file']:.3f} & {means['repository']:.3f} \\\\\n"
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    with open(OUTTEX, "w") as f:
        f.write("\n".join(tex))
    print("wrote", OUTTEX)

    # ================= FIGURES =================
    scope_color = {"function":"#1b9e77","file":"#d95f02","repository":"#7570b3"}

    # (1) overall tradeoff: F1 vs time, bubble = tokens; scope shown via legend
    fig, ax = plt.subplots(figsize=(6.4,4.6))
    handles = []
    for s in SCOPES:
        r = agg.loc[s]
        ax.scatter(r[("time","mean")], r[("F1","mean")],
                   s=max(r[("tokens","mean")]/10.0, 160),
                   color=scope_color[s], alpha=0.75, edgecolors="black", linewidths=1.3, zorder=3)
        handles.append(plt.Line2D([0],[0], marker="o", ls="", color=scope_color[s],
                       markersize=13, markeredgecolor="black",
                       label=f"{SCOPE_LABEL[s]}  ({r[('tokens','mean')]:.0f} tok)"))
    ax.set_xlabel("Avg. Response Time (s)", weight="bold", fontsize=15)
    ax.set_ylabel("F1-score", weight="bold", fontsize=15)
    ax.set_title("Correctness–Efficiency Trade-off", fontsize=15, weight="bold")
    ax.tick_params(labelsize=13)
    ax.legend(handles=handles, title="Scope (bubble size = avg tokens)",
              fontsize=12, title_fontsize=12, loc="lower left", framealpha=0.95)
    ax.margins(0.18)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/tradeoff.png", dpi=160); plt.close(fig)

    # (2) scope x tag interaction (the new headline figure)
    fig, ax = plt.subplots(figsize=(6,4.2))
    x = np.arange(len(TAGS))
    piv = tagF1.pivot(index="tag", columns="scope", values="mean").reindex(TAGS)
    for s in SCOPES:
        ax.plot(x, piv[s], marker="o", label=SCOPE_LABEL[s], color=scope_color[s], lw=2)
    ax.set_xticks(x); ax.set_xticklabels(["SFSF\n(single-func)","MFSF\n(multi-func)","MFMF\n(multi-file)"])
    ax.set_ylabel("F1-score", weight="bold")
    ax.set_xlabel("Vulnerability Complexity Tier", weight="bold")
    ax.set_title("Scope $\\times$ Complexity Interaction")
    ax.legend(title="Scope"); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/interaction.png", dpi=150); plt.close(fig)

    # (3) per-model tradeoff
    fig, ax = plt.subplots(figsize=(7,4.6))
    markers = {"function":"o","file":"s","repository":"^"}
    cmap = plt.cm.tab10(np.linspace(0,1,len(models)))
    for mi, m in enumerate(models):
        for s in SCOPES:
            r = pm[(pm.model==m)&(pm.scope==s)].iloc[0]
            ax.scatter(r["time"], r["F1"], marker=markers[s], s=110, color=cmap[mi],
                       alpha=0.8, edgecolors="black", linewidths=0.6)
    mh = [plt.Line2D([0],[0], marker="o", ls="", color=cmap[i], label=m, markersize=9) for i,m in enumerate(models)]
    sh = [plt.Line2D([0],[0], marker=markers[s], ls="", color="gray", label=SCOPE_LABEL[s], markersize=9) for s in SCOPES]
    l1 = ax.legend(handles=mh, title="Model", loc="lower right", fontsize=8); ax.add_artist(l1)
    ax.legend(handles=sh, title="Scope", loc="upper right", fontsize=8)
    ax.set_xlabel("Avg. Response Time (s)", weight="bold"); ax.set_ylabel("F1-score", weight="bold")
    ax.set_xscale("log"); ax.set_title("Per-model Correctness–Efficiency by Scope")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/model_tradeoff.png", dpi=150); plt.close(fig)

    # (4) per-language tradeoff (F1 by scope, grouped bars)
    df["lang"] = df["language"].replace({"c":"C/C++","c++":"C/C++","cpp":"C/C++","java":"Java","python":"Python"})
    langs = ["C/C++","Java","Python"]
    fig, ax = plt.subplots(figsize=(6.5,4.2))
    w = 0.25
    for i, s in enumerate(SCOPES):
        vals = []
        for lg in langs:
            sub = df[(df.lang==lg)&(df.scope==s)]
            # mean per-model F1
            f1s = [metrics(sub[sub.model==m])["F1"] for m in models if len(sub[sub.model==m])]
            vals.append(np.mean(f1s) if f1s else 0)
        ax.bar(np.arange(len(langs))+i*w, vals, w, label=SCOPE_LABEL[s], color=scope_color[s], alpha=0.85)
    ax.set_xticks(np.arange(len(langs))+w); ax.set_xticklabels(langs)
    ax.set_ylabel("F1-score", weight="bold"); ax.set_title("F1 by Programming Language and Scope")
    ax.legend(title="Scope"); ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/language_tradeoff.png", dpi=150); plt.close(fig)

    print("figures written to", FIGDIR)

    # console summary for the writeup
    print("\n=== overall (mean over models) ===")
    print(agg[[("F1","mean"),("F1","std"),("Accuracy","mean"),("Precision","mean"),
               ("Recall","mean"),("tokens","mean"),("time","mean")]].round(3))
    print("\n=== per-tag F1 ===")
    print(piv.round(3))


if __name__ == "__main__":
    main()
