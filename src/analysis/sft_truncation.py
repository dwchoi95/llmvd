"""SFT inputs beyond the 4,096-token training limit, per base tokenizer.

Uses src.core.sft.SFT (_user renders the prompt, _encode applies the chat
template). Target-block position comes from encoding the prompt cut at the
target's start/end (difference of prefix lengths; +-1-2 tokens at the cut).
Writes <out>/tables/training_reconstruction.tex and <out>/sft_truncation.json.

    python -m src.analysis.sft_truncation [--out paper]
"""
import argparse
import collections
import json
from pathlib import Path

from transformers import AutoTokenizer

from src.core.sft import SFT

ROOT = Path(__file__).resolve().parents[2]
TRAIN = str(ROOT / "data" / "FuncFileRepo.train.jsonl")
SCOPES = ("function", "file", "repository")
TOKENIZERS = {"Qwen3": ("Qwen/Qwen3-30B-A3B-Instruct-2507", None),      # Instruct/Thinking share it
              "Mistral/Magistral": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", None),
              "Nemotron": ("nvidia/NVIDIA-Nemotron-Nano-12B-v2", "/no_think")}


def stats(repo, ctl, rows):
    sft = SFT(model=repo, train_path=TRAIN, out_dir="/tmp/sft-truncation-unused", control_token=ctl)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    mtok = None
    if tok.chat_template is None:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        mtok = MistralTokenizer.from_hf_hub(repo)
    system = sft._system()
    sft.max_seq_len = 10 ** 9                       # measure untruncated lengths

    def plen(user):
        ids, labels = sft._encode(tok, system, user, '{"vulnerable": 1}', mtok)
        tgt = sum(l != -100 for l in labels)
        return len(ids) - tgt, tgt

    c = collections.Counter()
    for r in rows:
        for scope in SCOPES:
            user = sft._user(r, scope)
            full, tgt = plen(user)
            budget = 4096 - tgt
            s = user.index(r["function"], user.index("Target Function"))
            c[scope, "n"] += 1
            c[scope, "truncated"] += full > budget
            c[scope, "target_full"] += full - plen(user[:s])[0] <= budget
            c[scope, "target_absent"] += full - plen(user[:s + len(r["function"])])[0] >= budget
    return {f"{sc}|{k}": c[sc, k] for sc in SCOPES for k in ("n", "truncated", "target_full", "target_absent")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper")
    out = ROOT / ap.parse_args().out
    rows = [json.loads(l) for l in open(TRAIN)]
    res = {name: stats(repo, ctl, rows) for name, (repo, ctl) in TOKENIZERS.items()}
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "sft_truncation.json").write_text(json.dumps(res, indent=1) + "\n")
    pct = lambda v, sc, k: (lambda x: "-" if x == 0 else f"{x:.1f}")(100 * v[f"{sc}|{k}"] / v[f"{sc}|n"])
    body = []
    for si, sc in enumerate(SCOPES):
        body += [r"\midrule"] if si else []
        body += [(rf"\multirow{{{len(res)}}}{{*}}{{{sc.capitalize()}}}" if ni == 0 else "") + f" & {name} & "
                 + " & ".join(pct(v, sc, k) for k in ("truncated", "target_full", "target_absent")) + r" \\"
                 for ni, (name, v) in enumerate(res.items())]
    (out / "tables" / "training_reconstruction.tex").write_text("\n".join([
        r"\begin{table}[h]",
        r"\centering\caption{SFT training inputs exceeding the 4,096-token limit}\label{tab:truncation}",
        r"\small\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}ll||r|rr@{}}", r"\toprule",
        r"\multirow{2}{*}{\textbf{Scope}} & \multirow{2}{*}{\textbf{Tokenizer}} & \multicolumn{1}{c|}{\textbf{Truncated}} & \multicolumn{2}{c}{\textbf{Target block (\%)}} \\",
        r"\cline{3-5}",
        r"\rule{0pt}{2.6ex} & & (\%) & Full & Absent \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}",
        r"\end{table}", ""]))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
