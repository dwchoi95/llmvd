"""ReposVul.jsonl -> FuncFileRepo.final.jsonl.

The study's dataset is built in four stages (`Dataset.build()` runs them all):

  1. load_pairs         — from each ReposVul record, take the vulnerable
                          (before-patch) target function and its patched
                          (after) twin as a (vul, non-vul) pair, per changed
                          file. Complexity tags (SFSF/MFSF/MFMF) derive from
                          how many functions/files the patch touched.
  2. CallGraph.collect  — real call-graph extraction around the target
                          function at the exact commit (see callgraph.py;
                          parsing lives in treesitter.py).
  3. filter_pairs       — keep only complete pairs where BOTH sides got a
                          non-empty call graph, then fit_context_budget drops
                          pairs whose rendered prompt would exceed the
                          panel's common 128K context under ANY panel
                          tokenizer.
  4. save               — slim schema (function/file/repository/language/
                          vulnerable + tag + provenance) with a sequential
                          integer `id`; pairs stay adjacent (vul, then twin).
  5. build_examples     — retrieve one vulnerable + one secure example per
                          test row (GRACE-style) into a SEPARATE file keyed
                          by test_id (two rows per id), so the test schema
                          stays clean and non-RAG runs ignore it.
  6. build_sft          — render, for every (train row x scope), the SAME
                          zero-shot prompt the detector uses at eval paired
                          with the `{"vulnerable": <digit>}` target (model-
                          agnostic; tokenization/truncation happen at train
                          time). Repository neighbours keep natural order —
                          eval's CodeBLEU re-ranking is a context-limit
                          device, immaterial for teaching the verdict.

Outputs:
    data/FuncFileRepo.test.jsonl     — the evaluation set (960 rows)
    data/FuncFileRepo.example.jsonl  — RAG examples ({test_id, example,
                                       vulnerable}, 2 rows per test_id)

Usage:
    ./env/bin/python -m src.utils.dataset          # data/ReposVul.jsonl -> test + example
    ./env/bin/python -m src.utils.dataset --build-sft data/FuncFileRepo.train.jsonl
"""
from __future__ import annotations

import re
import json
import argparse
import collections
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .callgraph import CallGraph, LANG_NORM

# panel served with a common 128K context (min native ctx across the panel)
MAX_MODEL_LEN = 131072
GEN_RESERVE = 1024      # largest strategy max_tokens
MARGIN = 256            # chat-template overhead
PANEL_TOKENIZERS = {
    "llama3.1:8b": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "mistral-nemo:12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "phi3:14b": "microsoft/Phi-3-medium-128k-instruct",
    "deepseek-coder-v2:16b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "qwen3-coder:30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}

KEEP_FIELDS = ["function", "file", "repository", "language", "vulnerable",
               "tag", "cve_id", "cwe_id", "cvss", "project", "commit_id",
               "file_name"]

# retrieved-example (RAG) settings — GRACE-style two-stage retrieval:
# fast lexical prefilter (token Jaccard) to top-K, then CodeBLEU re-rank
# (lexical+syntactic mix) to top-1. Per test row we retrieve one vulnerable
# and one secure example, both from a corpus that is CVE-disjoint from the
# test set, same-language, and never from the target's project (near-
# duplicate / label-leak guard). Examples are written to a SEPARATE file
# (FuncFileRepo.example.jsonl), keyed by test_id, so the test schema stays
# clean and non-RAG runs ignore them entirely.
RAG_PREFILTER_K = 50


class Dataset:
    """Build the final evaluation dataset from a raw ReposVul dump."""

    def __init__(self,
                 input_path: str = "data/ReposVul.jsonl",
                 output_path: str = "data/FuncFileRepo.test.jsonl",
                 example_path: str = "data/FuncFileRepo.example.jsonl",
                 repos_dir: str = "codeql",
                 budget: int = 30):
        self.input_path = input_path
        self.output_path = output_path
        self.example_path = example_path
        self.callgraph = CallGraph(repos_dir=repos_dir, budget=budget)

    # ------------------------------------------------------------------ #
    # stage 1: ReposVul -> (vul, non-vul) pair rows
    # ------------------------------------------------------------------ #
    @staticmethod
    def _valid_detail_language(detail: Dict[str, Any], lang: str) -> bool:
        ext = (detail.get("file_language") or "").lower()
        return {"python": "py", "c++": "cpp", "c": "c", "java": "java"}.get(lang) == ext

    @staticmethod
    def _collect_functions(detail: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
        fs = []
        for f in detail.get(key, []) or []:
            code = f.get("function")
            if code and isinstance(code, str):
                fs.append({"function": code, "target": f.get("target", 0)})
        return fs

    @staticmethod
    def _choose_primary(functions: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        # prefer the first target=1 function, else the longest one
        for i, f in enumerate(functions):
            if f.get("target", 0) == 1:
                return i, f
        if not functions:
            return -1, {}
        idx = max(range(len(functions)), key=lambda i: len(functions[i].get("function", "")))
        return idx, functions[idx]

    def load_pairs(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """One (vul, non-vul) pair per valid changed file of each ReposVul
        record: vul = before-patch target function at the parent commit,
        non-vul = its after-patch version at the fix commit."""
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        with open(self.input_path, encoding="utf-8") as fin:
            for line in fin:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("outdated", 0) == 1:
                    continue
                language = (row.get("cve_language") or "").lower()
                parents = row.get("parents") or []
                commit_before = parents[-1]["commit_id_before"] if parents else None
                details = [d for d in (row.get("details") or [])
                           if self._valid_detail_language(d, language)]
                if not details or not commit_before:
                    continue

                # complexity tag from how many functions/files the patch touched
                n_funcs = sum(len(d.get("function_before") or []) for d in details)
                multi_func = n_funcs > 1
                multi_file = len(details) > 1
                tag = ("MFMF" if multi_file else "MFSF") if multi_func else "SFSF"

                meta = {
                    "cve_id": row.get("cve_id"),
                    "cwe_id": list(row.get("cwe_id") or []),
                    "cvss": row.get("cvss"),
                    "language": language,
                    "project": row.get("project"),
                    "tag": tag,
                }
                for d in details:
                    fb = self._collect_functions(d, "function_before")
                    fa = self._collect_functions(d, "function_after")
                    idx_b, prim_before = self._choose_primary(fb)
                    prim_after = fa[idx_b] if 0 <= idx_b < len(fa) else (fa[0] if fa else {})
                    if not (prim_before and d.get("code_before")
                            and prim_after and d.get("code")):
                        continue
                    empty = {"callee": [], "caller": []}
                    vul = {**meta, "commit_id": commit_before,
                           "file_name": d.get("file_name"),
                           "function": prim_before["function"],
                           "file": d["code_before"],
                           "repository": dict(empty), "vulnerable": True}
                    nonvul = {**meta, "commit_id": row.get("commit_id"),
                              "file_name": d.get("file_name"),
                              "function": prim_after["function"],
                              "file": d["code"],
                              "repository": dict(empty), "vulnerable": False}
                    pairs.append((vul, nonvul))
        # deterministic order + dedup identical pairs
        seen = set()
        out = []
        for v, n in sorted(pairs, key=lambda p: (str(p[0]["project"]),
                                                 str(p[0]["commit_id"]),
                                                 str(p[0]["file_name"]))):
            sig = (v["project"], v["commit_id"], v["file_name"],
                   hash(v["function"]), hash(n["function"]))
            if sig in seen:
                continue
            seen.add(sig)
            out.append((v, n))
        print(f"[load_pairs] {len(out)} (vul, non-vul) pairs")
        return out

    # ------------------------------------------------------------------ #
    # stage 3: keep complete pairs with call graphs on both sides + 128K fit
    # ------------------------------------------------------------------ #
    @staticmethod
    def _has_cg(row: Dict[str, Any]) -> bool:
        repo = row.get("repository") or {}
        return bool(repo.get("callee") or repo.get("caller"))

    def filter_pairs(self, pairs):
        kept = [(v, n) for v, n in pairs if self._has_cg(v) and self._has_cg(n)]
        print(f"[filter_pairs] both-side call graphs: {len(kept)}/{len(pairs)} pairs")
        return kept

    def fit_context_budget(self, pairs):
        """Drop pairs whose rendered prompt exceeds the panel's common 128K
        budget under ANY panel tokenizer (conservative)."""
        from transformers import AutoTokenizer
        from ..prompts import PromptManager
        from ..prompts.strategies import STRATEGIES
        pm = PromptManager()
        toks, budgets = {}, {}
        for name, hf in PANEL_TOKENIZERS.items():
            tok = AutoTokenizer.from_pretrained(hf)
            sys_max = max(len(tok.encode(pm.render(file=s.system_file)))
                          for s in STRATEGIES.values())
            toks[name] = tok
            budgets[name] = MAX_MODEL_LEN - GEN_RESERVE - sys_max - MARGIN

        def prompts(r):
            language, function = r.get("language") or "", r.get("function") or ""
            repo = r.get("repository") or {}
            info = ""
            for kind, header in (("callee", "### Functions called by the Target Function (Callees):\n"),
                                 ("caller", "### Functions that call the Target Function (Callers):\n")):
                items = repo.get(kind) or []
                if items:
                    info += header
                    for c in items:
                        info += f"#### File: {c.get('file')}\n```{language}\n{c.get('function')}\n```\n"
            return [
                pm.render(file="src/prompts/detection/function.md",
                          language=language, function=function),
                pm.render(file="src/prompts/detection/file.md", language=language,
                          function=function, file_code=r.get("file") or ""),
                pm.render(file="src/prompts/detection/repository.md",
                          language=language, function=function, repository=info),
            ]

        kept = []
        for v, n in pairs:
            ok = all(len(tok.encode(p)) <= budgets[name]
                     for r in (v, n) for p in prompts(r)
                     for name, tok in toks.items())
            if ok:
                kept.append((v, n))
        print(f"[fit_context_budget] within 128K for every panel tokenizer: "
              f"{len(kept)}/{len(pairs)} pairs")
        return kept

    # ------------------------------------------------------------------ #
    # stage 3.5: retrieved examples (vul_ex / sec_ex) for the RAG strategy
    # ------------------------------------------------------------------ #
    _TOKEN_RE = re.compile(r"[A-Za-z_]\w+")

    @classmethod
    def _token_set(cls, code: str) -> frozenset:
        return frozenset(cls._TOKEN_RE.findall(code or ""))

    @staticmethod
    def _codebleu_lang(language: str) -> str:
        lang = (language or "").lower()
        return "cpp" if lang == "c++" else lang

    def build_example_corpus(self, all_pairs, exclude_cves: set) -> Dict[str, Dict[str, list]]:
        """side ('vul'|'sec') -> language -> [entry] from pairs whose CVE is
        NOT in the final dataset. Entries carry pre-tokenized sets for the
        Jaccard prefilter."""
        corpus: Dict[str, Dict[str, list]] = {"vul": collections.defaultdict(list),
                                              "sec": collections.defaultdict(list)}
        for v, n in all_pairs:
            if v.get("cve_id") in exclude_cves:
                continue
            for side, r in (("vul", v), ("sec", n)):
                corpus[side][r["language"]].append({
                    "function": r["function"],
                    "project": r["project"],
                    "cve_id": r["cve_id"],
                    "cwe_id": r["cwe_id"],
                    "tokens": self._token_set(r["function"]),
                })
        return corpus

    def _retrieve_example(self, row: Dict[str, Any], entries: list) -> Dict[str, Any] | None:
        """Most similar corpus entry: Jaccard prefilter -> CodeBLEU top-1.
        Candidates from the target's own project are excluded. Returns the
        full entry (function + provenance) so the caller can leakage-check;
        only the function text is written to the example file."""
        from codebleu import calc_codebleu
        tgt_tokens = self._token_set(row["function"])
        cands = [e for e in entries if e["project"] != row["project"]]
        if not cands:
            return None

        def jac(e):
            inter = len(tgt_tokens & e["tokens"])
            union = len(tgt_tokens | e["tokens"]) or 1
            return inter / union

        # deterministic prefilter order
        cands = sorted(cands, key=lambda e: (-jac(e), str(e["cve_id"]),
                                             str(e["project"])))[:RAG_PREFILTER_K]
        lang = self._codebleu_lang(row["language"])
        best, best_score = None, -1.0
        for e in cands:
            try:
                score = calc_codebleu([row["function"]], [e["function"]],
                                      lang=lang)["codebleu"]
            except Exception:
                score = 0.0
            if score > best_score:
                best, best_score = e, score
        return best

    def build_examples(self, test_rows: List[Dict[str, Any]],
                       all_pairs, example_path: str) -> None:
        """Retrieve one vulnerable + one secure example per test row and write
        them to `example_path` as {test_id, example, vulnerable} lines (two
        rows per test_id). Provenance (project/cve) is used only for the
        leakage assertion, not written out."""
        import logging
        logging.disable(logging.WARNING)  # codebleu dataflow warnings
        exclude = {r.get("cve_id") for r in test_rows}
        corpus = self.build_example_corpus(all_pairs, exclude)
        out_rows, missing, leak = [], 0, 0
        try:
            for i, row in enumerate(test_rows, 1):
                tid, lang = row["id"], row["language"]
                for label, side in ((True, "vul"), (False, "sec")):
                    e = self._retrieve_example(row, corpus[side].get(lang, []))
                    if e is None:
                        missing += 1
                        continue
                    # leakage guard: never the same CVE/project, never self
                    if (e["cve_id"] == row.get("cve_id")
                            or e["project"] == row.get("project")
                            or e["function"].strip() == row["function"].strip()):
                        leak += 1
                    out_rows.append({"test_id": tid, "example": e["function"],
                                     "vulnerable": label})
                if i % 50 == 0:
                    print(f"[build_examples] {i}/{len(test_rows)}")
        finally:
            logging.disable(logging.NOTSET)
        Path(example_path).parent.mkdir(parents=True, exist_ok=True)
        with open(example_path, "w", encoding="utf-8") as fout:
            for r in out_rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[build_examples] {example_path}: {len(out_rows)} rows "
              f"({len(test_rows)} test_ids x2); missing={missing}, leak_violations={leak}")

    # ------------------------------------------------------------------ #
    # stage 4: save (slim schema, sequential id, pairs adjacent)
    # ------------------------------------------------------------------ #
    def save(self, pairs) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        tags = collections.Counter()
        with open(self.output_path, "w", encoding="utf-8") as fout:
            rid = 0
            for v, n in pairs:
                for r in (v, n):
                    rid += 1
                    out = {"id": rid}
                    out.update({k: r.get(k) for k in KEEP_FIELDS})
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    tags[r.get("tag")] += 1
        print(f"[save] {self.output_path}: {rid} rows, tags={dict(tags)}")

    # ------------------------------------------------------------------ #
    def build(self) -> None:
        all_pairs = self.load_pairs()
        for i, (v, n) in enumerate(all_pairs, 1):
            self.callgraph.collect(v)
            self.callgraph.collect(n)
            if i % 25 == 0:
                print(f"[collect_callgraphs] {i}/{len(all_pairs)} pairs")
        pairs = self.filter_pairs(all_pairs)
        pairs = self.fit_context_budget(pairs)
        self.save(pairs)
        test_rows = [json.loads(l) for l in open(self.output_path, encoding="utf-8")]
        self.build_examples(test_rows, all_pairs, self.example_path)

    def build_examples_for(self, test_path: str) -> None:
        """Build the RAG example file for an already-built test set, without
        redoing call-graph collection (the corpus needs only load_pairs)."""
        test_rows = [json.loads(l) for l in open(test_path, encoding="utf-8")]
        all_pairs = self.load_pairs()
        self.build_examples(test_rows, all_pairs, self.example_path)

    # ------------------------------------------------------------------ #
    # SFT training corpus: same schema as the test set, but CVE-DISJOINT
    # from it, and NO pair-completeness / 128K requirement — every row whose
    # call graph resolves is a usable training example on its own.
    # ------------------------------------------------------------------ #
    def build_train(self, test_path: str, train_path: str,
                    resume: bool = True) -> None:
        test_cves = {json.loads(l).get("cve_id")
                     for l in open(test_path, encoding="utf-8")}
        all_pairs = self.load_pairs()
        # flatten to rows, keep only CVE-disjoint-from-test
        rows = [r for v, n in all_pairs for r in (v, n)
                if r.get("cve_id") not in test_cves]
        print(f"[build_train] {len(rows)} rows CVE-disjoint from test "
              f"(from {len(all_pairs)} pairs)")

        # resume: skip rows already written (keyed by the same triple the
        # detector resumes on), so an interrupted overnight run continues
        def rkey(r):
            return f"{r.get('project')}|{r.get('commit_id')}|{r.get('file_name')}|{r.get('vulnerable')}"
        done = {}
        if resume and Path(train_path).exists():
            for l in open(train_path, encoding="utf-8"):
                try:
                    d = json.loads(l)
                except json.JSONDecodeError:
                    continue
                done[rkey(d)] = d
            print(f"[build_train] resume: {len(done)} rows already collected")

        # rows are sorted by (project, commit, file) in load_pairs, so a
        # project's rows are contiguous: delete each project's clone the moment
        # we move to the next project, keeping peak disk to one repo at a time.
        import shutil

        def clone_dir(project, language):
            lang = LANG_NORM.get((language or "").lower())
            if not lang:
                return None
            return (self.callgraph.repos_dir / lang / "repo"
                    / (project or "").replace("/", "__"))

        def drop_clone(project, language):
            d = clone_dir(project, language)
            if d is not None and d.exists():
                shutil.rmtree(d, ignore_errors=True)

        Path(train_path).parent.mkdir(parents=True, exist_ok=True)
        n_cg = 0
        cur = None   # (project, language) currently on disk
        with open(train_path, "w", encoding="utf-8") as fout:
            rid = 0
            for i, r in enumerate(rows, 1):
                proj = (r.get("project"), r.get("language"))
                if cur is not None and proj != cur:
                    drop_clone(*cur)
                cur = proj

                rid += 1
                prev = done.get(rkey(r))
                if prev is not None:
                    prev["id"] = rid
                    out = prev
                else:
                    self.callgraph.collect(r)   # fills r['repository'] or empties it
                    out = {"id": rid}
                    out.update({k: r.get(k) for k in KEEP_FIELDS})
                if self._has_cg(out):
                    n_cg += 1
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                if i % 50 == 0:
                    print(f"[build_train] {i}/{len(rows)} (with call graph: {n_cg})")
        if cur is not None:
            drop_clone(*cur)
        print(f"[build_train] {train_path}: {rid} rows, {n_cg} with a call graph")


# ---------------------------------------------------------------------- #
# SFT training-corpus rendering (merged from build_sft_data.py)
# ---------------------------------------------------------------------- #
_SFT_SCOPES = ("function", "file", "repository")
_SFT_PROMPT_ROOT = "src/prompts"
_SFT_PROMPT_DIR = "zero"   # sft reuses zero/ templates (model variant, not a new prompt)


def _sft_repo_info(language: str, callees: list, callers: list) -> str:
    """Identical formatting to Detector._repo_info (natural order here)."""
    info = ""
    if callees:
        info += "### Functions called by the Target Function (Callees):\n"
        for c in callees:
            info += f"#### File: {c['file']}\n```{language}\n{c['function']}\n```\n"
    if callers:
        info += "### Functions that call the Target Function (Callers):\n"
        for c in callers:
            info += f"#### File: {c['file']}\n```{language}\n{c['function']}\n```\n"
    return info


def build_sft(inp: str, out: str) -> None:
    """Render (row x scope) zero-prompt/verdict pairs for SFT training."""
    from ..prompts import PromptManager
    pm = PromptManager()

    def tpl(name: str) -> str:
        return f"{_SFT_PROMPT_ROOT}/{_SFT_PROMPT_DIR}/{name}.md"

    system = pm.render(file=tpl("system"))
    rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
    n_written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            function = row["function"]
            language = row["language"]
            target = json.dumps({"vulnerable": 1 if row["vulnerable"] else 0})
            base_kw = dict(language=language, function=function)
            for scope in _SFT_SCOPES:
                if scope == "function":
                    user = pm.render(file=tpl("function"), **base_kw)
                elif scope == "file":
                    user = pm.render(file=tpl("file"),
                                     file_code=row["file"], **base_kw)
                else:
                    repo = row.get("repository") or {"callee": [], "caller": []}
                    user = pm.render(
                        file=tpl("repository"),
                        repository=_sft_repo_info(language,
                                                  repo.get("callee") or [],
                                                  repo.get("caller") or []),
                        **base_kw)
                f.write(json.dumps({
                    "id": row["id"], "scope": scope,
                    "vulnerable": bool(row["vulnerable"]),
                    "system": system, "user": user, "target": target,
                }, ensure_ascii=False) + "\n")
                n_written += 1
    print(f"wrote {n_written} sft examples ({len(rows)} rows x "
          f"{len(_SFT_SCOPES)} scopes) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="data/ReposVul.jsonl")
    ap.add_argument("-o", "--output", default="data/FuncFileRepo.test.jsonl")
    ap.add_argument("-e", "--examples", default="data/FuncFileRepo.example.jsonl")
    ap.add_argument("--budget", type=int, default=30, help="max neighbors per side")
    ap.add_argument("--examples-for", default=None, metavar="TEST_JSONL",
                    help="only build the RAG example file for an existing test "
                         "set (skips call-graph collection)")
    ap.add_argument("--build-train", default=None, metavar="TEST_JSONL",
                    help="build the SFT training corpus (CVE-disjoint from the "
                         "given test set) into --train-out; row-wise call "
                         "graphs, no pair-completeness requirement, resumable")
    ap.add_argument("--train-out", default="data/FuncFileRepo.train.jsonl")
    ap.add_argument("--build-sft", default=None, metavar="TRAIN_JSONL",
                    help="render the (row x scope) SFT prompt/verdict corpus "
                         "from an existing training split into --sft-out "
                         "(no call-graph work)")
    ap.add_argument("--sft-out", default="data/FuncFileRepo.sft.jsonl")
    args = ap.parse_args()
    if args.build_sft:
        Path(args.sft_out).parent.mkdir(parents=True, exist_ok=True)
        build_sft(args.build_sft, args.sft_out)
        return
    ds = Dataset(args.input, args.output, example_path=args.examples,
                 budget=args.budget)
    if args.build_train:
        ds.build_train(args.build_train, args.train_out)
    elif args.examples_for:
        ds.build_examples_for(args.examples_for)
    else:
        ds.build()


if __name__ == "__main__":
    main()
