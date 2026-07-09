import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from .mf_callgraph import target_func_name, _norm_lang

# separator between merged functions inside the `function` column
FUNC_SEP = "\n\n"


def _is_valid_detail_language(detail: Dict[str, Any], lang: str) -> bool:
    ext = (detail.get("file_language") or "").lower()
    if lang == "python":
        return ext == "py"
    if lang == "c++":
        return ext == "cpp"
    if lang == "c":
        return ext == "c"
    if lang == "java":
        return ext == "java"
    return False


def _collect_targets(detail: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    picked = []
    for f in (detail.get(key) or []):
        if f and isinstance(f, dict) and f.get("function"):
            if int(f.get("target", 0)) == 1:
                picked.append(f)
    return picked


def _func_name(code: str, lang_key: str | None) -> str | None:
    if not code or not lang_key:
        return None
    try:
        return target_func_name(code, lang_key)
    except Exception:
        return None


def _merge_target_functions(detail: Dict[str, Any], lang_key: str | None) -> Tuple[str | None, str | None, int, int]:
    """Build the (positive, negative) `function` content for one file.

    positive = all `target==1` functions in function_before, concatenated.
    negative = their patched versions: name-matched changed version first
               (same name, code differs), else a "changed function" from
               function_after (code absent from function_before) as fallback
               (database.py-style). Returns (pos, neg, n_targets, n_neg).
    """
    fb = detail.get("function_before") or []
    fa = detail.get("function_after") or []
    targets = [f for f in fb if isinstance(f, dict) and f.get("function") and int(f.get("target", 0)) == 1]
    # require an identified vulnerable function; files with no target==1 are
    # changed-but-not-vulnerable (tests, headers) and must not be emitted.
    if not targets:
        return None, None, 0, 0

    positive = FUNC_SEP.join(f["function"] for f in targets)

    before_codes = {f.get("function", "").strip() for f in fb if isinstance(f, dict)}
    after_by_name: Dict[str, List[str]] = {}
    for f in fa:
        if not isinstance(f, dict) or not f.get("function"):
            continue
        nm = _func_name(f["function"], lang_key)
        if nm:
            after_by_name.setdefault(nm, []).append(f["function"])
    changed_pool = [f["function"] for f in fa
                    if isinstance(f, dict) and f.get("function")
                    and f["function"].strip() not in before_codes]

    neg_funcs: List[str] = []
    used: set = set()
    for tf in targets:
        bname = _func_name(tf["function"], lang_key)
        bcode = tf["function"].strip()
        chosen = None
        for ac in (after_by_name.get(bname, []) if bname else []):
            if ac.strip() != bcode and ac.strip() not in used:
                chosen = ac
                break
        if chosen is None:
            for ac in changed_pool:
                if ac.strip() not in used:
                    chosen = ac
                    break
        if chosen is not None:
            neg_funcs.append(chosen)
            used.add(chosen.strip())

    negative = FUNC_SEP.join(neg_funcs) if neg_funcs else None
    return positive, negative, len(targets), len(neg_funcs)


def _repo_context_from_details(curr_file: str, details: List[Dict[str, Any]], exclude_code: str) -> Dict[str, List[Dict[str, str]]]:
    callees: List[Dict[str, str]] = []
    callers: List[Dict[str, str]] = []
    for d in details:
        file_name = d.get("file_name", "")
        for key in ("function_before", "function_after"):
            for f in (d.get(key) or []):
                code = f.get("function") if isinstance(f, dict) else None
                if not code or code == exclude_code:
                    continue
                entry = {"file": file_name, "function": code}
                if file_name == curr_file:
                    callees.append(entry)
                else:
                    callers.append(entry)
    return {"callee": callees, "caller": callers}


def _classify_tag(files_with_target: int, total_targets: int) -> str:
    if files_with_target <= 1 and total_targets <= 1:
        return "SFSF"  # Single-Function, Single-File
    if files_with_target <= 1 and total_targets > 1:
        return "MFSF"  # Multi-Function, Single-File
    # files_with_target > 1
    return "MFMF"      # Multi-Function, Multi-File


def build_unified(
    reposvul_path: str,
    output_path: str = "data/FuncFileRepo.jsonl",
    chunksize: int = 5000,
) -> Dict[str, int]:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # truncate
    open(output_path, "w").close()

    stats = {"samples": 0, "vulnerable": 0, "non_vulnerable": 0, "SFSF": 0, "MFSF": 0, "MFMF": 0,
             "targets_merged": 0, "neg_dropped": 0}

    for chunk in pd.read_json(reposvul_path, lines=True, chunksize=chunksize):
        out_rows: List[Dict[str, Any]] = []
        for _, row in chunk.iterrows():
            if row.get("outdated", 0) == 1:
                continue
            language = (row.get("cve_language") or "").lower()
            details = [d for d in (row.get("details") or []) if _is_valid_detail_language(d, language)]
            if not details:
                continue

            # count targets and files containing targets
            total_targets = 0
            files_with_target = 0
            for d in details:
                t = _collect_targets(d, "function_before")
                total_targets += len(t)
                if t:
                    files_with_target += 1

            tag = _classify_tag(files_with_target, total_targets)
            is_mf = (tag != "SFSF")
            is_mfi = (tag == "MFMF")

            cve_id = row.get("cve_id")
            cwe_id = tuple(row.get("cwe_id") or [])
            cvss = row.get("cvss")
            project = row.get("project")
            commit_id = row.get("commit_id")
            parents = row.get("parents") or []
            commit_id_before = parents[-1]["commit_id_before"] if parents else None
            group_id = f"{project}::{cve_id}::{commit_id or ''}"

            lang_key = _norm_lang(language)
            for d in details:
                file_name = d.get("file_name")
                code_before = d.get("code_before")
                code_after = d.get("code")
                pos_func, neg_func, n_tgt, n_neg = _merge_target_functions(d, lang_key)
                # repository is filled later by mf_callgraph (real call graph)
                empty_repo = {"callee": [], "caller": []}

                meta = {
                    "cve_id": cve_id,
                    "cwe_id": cwe_id,
                    "cvss": cvss,
                    "language": language,
                    "project": project,
                    "file_name": file_name,
                    "group_id": group_id,
                    "is_multi_function": is_mf,
                    "is_multi_file": is_mfi,
                    "tag": tag,
                }

                if pos_func and code_before:
                    out_rows.append({
                        **meta,
                        "commit_id": commit_id_before,
                        "function": pos_func,
                        "file": code_before,
                        "repository": empty_repo,
                        "vulnerable": True,
                    })
                    stats["samples"] += 1
                    stats["vulnerable"] += 1
                    stats[tag] += 1
                    stats["targets_merged"] += n_tgt

                if neg_func and code_after:
                    out_rows.append({
                        **meta,
                        "commit_id": commit_id,
                        "function": neg_func,
                        "file": code_after,
                        "repository": empty_repo,
                        "vulnerable": False,
                    })
                    stats["samples"] += 1
                    stats["non_vulnerable"] += 1
                    stats[tag] += 1
                elif pos_func and code_before:
                    # positive emitted but no usable patched negative for this file
                    stats["neg_dropped"] += 1

        if out_rows:
            with open(output_path, "a", encoding="utf-8") as fout:
                for r in out_rows:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    return stats


def main():
    p = argparse.ArgumentParser(description="Build unified FuncFileRepo.jsonl with SFSF/MFSF/MFMF tags from ReposVul JSONL")
    p.add_argument("--input", "-i", required=True, help="Path to ReposVul.jsonl")
    p.add_argument("--output", "-o", default="data/FuncFileRepo.jsonl", help="Output JSONL path")
    p.add_argument("--chunksize", type=int, default=5000, help="Read in chunks (lines)")
    args = p.parse_args()

    stats = build_unified(args.input, args.output, chunksize=args.chunksize)
    print(stats)


if __name__ == "__main__":
    main()
