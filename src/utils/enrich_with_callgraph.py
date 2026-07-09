import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .database import Database
from .funcNameParser import FuncNameParser


def run_cmd(cmd: List[str], cwd: str | None = None, timeout: int | None = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout


def extract_snippet(file_path: Path, func_name: str, pre: int = 10, post: int = 40) -> str:
    try:
        lines = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except Exception:
        return f"// {func_name}"
    # find first occurrence line
    idx = next((i for i, ln in enumerate(lines) if func_name in ln), None)
    if idx is None:
        return f"// {func_name}"
    s = max(0, idx - pre)
    e = min(len(lines), idx + post)
    return "\n".join(lines[s:e])


def cflow_edges(repo_dir: Path, rel_file: str) -> List[Tuple[str, str]]:
    """Collect edges using cflow over a small set of files (same directory as rel_file).
    This widens coverage beyond a single TU but avoids whole-repo cost.
    """
    target = (repo_dir / rel_file).resolve()
    src_dir = target.parent
    # Gather up to N C/C++ sources in the same directory
    candidates: List[str] = []
    for ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"):
        for p in list(src_dir.glob(f"*{ext}"))[:200]:
            candidates.append(str(p))
    if not candidates:
        candidates = [str(target)]
    try:
        out = run_cmd(["cflow", "-b", *candidates], cwd=str(src_dir))
    except Exception:
        return []
    edges: List[Tuple[str, str]] = []
    stack: List[str] = []
    for line in out.splitlines():
        # Heuristic: count leading spaces to get depth, capture current function token up to first '(' or end
        m = re.match(r"(\s*)([^\s(]+)", line)
        if not m:
            continue
        depth = len(m.group(1)) // 2  # 2-space indent per level (heuristic)
        func = m.group(2)
        if depth < 0:
            depth = 0
        if depth >= len(stack):
            stack.append(func)
        else:
            stack = stack[:depth] + [func]
        if depth > 0:
            caller = stack[depth - 1]
            callee = func
            edges.append((caller, callee))
    return edges


def pycg_edges(repo_dir: Path, rel_file: str) -> List[Tuple[str, str]]:
    out_json = repo_dir / ".pycg.cg.json"
    pycg_home = os.environ.get("PYCG_HOME")
    try:
        if pycg_home and Path(pycg_home).exists():
            # run as module from source tree
            run_cmd(["/home/selab/cdw/llmvd/.venv/bin/python", "-m", "pycg",
                     "--package-dir", str(repo_dir),
                     "--entry-file", str(repo_dir / rel_file),
                     "--cg", str(out_json)], cwd=pycg_home)
        else:
            run_cmd(["pycg", "--package-dir", str(repo_dir), "--entry-file", str(repo_dir / rel_file), "--cg", str(out_json)])
    except Exception:
        return []
    try:
        data = json.loads(out_json.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return []
    edges: List[Tuple[str, str]] = []
    for src, tgts in data.get("callgraph", {}).items():
        for tgt in tgts:
            edges.append((str(src), str(tgt)))
    return edges


def jacg_edges(repo_dir: Path, rel_file: str) -> List[Tuple[str, str]]:
    jar = os.getenv("JACG_JAR") or os.getenv("JACG_HOME")
    if not jar or not Path(jar).exists():
        return []
    out_txt = repo_dir / "jacg.txt"
    try:
        # tool CLIs differ; try common patterns
        try:
            run_cmd(["java", "-jar", jar, "-dir", str(repo_dir), "-o", str(out_txt)])
        except Exception:
            run_cmd(["java", "-jar", jar, str(repo_dir), str(out_txt)])
    except Exception:
        return []
    edges: List[Tuple[str, str]] = []
    for line in out_txt.read_text(encoding='utf-8', errors='ignore').splitlines():
        # Expect lines like: caller -> callee or caller===callee (vary by tool conf)
        if "->" in line:
            parts = line.split("->")
            if len(parts) == 2:
                caller = parts[0].strip()
                callee = parts[1].strip()
                edges.append((caller, callee))
    return edges


def neighbors_for_target(edges: List[Tuple[str, str]], target: str) -> Tuple[List[str], List[str]]:
    pat = re.compile(rf"(^|\W){re.escape(target)}(\W|$)")
    callers = [c for c, d in edges if pat.search(d) is not None]
    callees = [d for c, d in edges if pat.search(c) is not None]
    return callers, callees


def enrich_with_callgraph(input_jsonl: str, output_jsonl: str, limit: int, tags: List[str], languages: List[str], depth: int, budget: int) -> Dict[str, int]:
    db = Database()
    stats = {"processed": 0, "updated": 0}
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(input_jsonl, 'r', encoding='utf-8') as fin, open(output_jsonl, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if stats["processed"] >= limit:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            tag = obj.get('tag')
            lang = (obj.get('language') or '').lower()
            if tags and tag not in tags:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            if languages and lang not in languages:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            project = obj.get('project')
            commit_id = obj.get('commit_id')
            rel_file = obj.get('file_name')
            func_code = obj.get('function') or ''
            if not (project and commit_id and rel_file and func_code):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            try:
                repo_dir = db.ensure_repo(project, lang if lang != 'c++' else 'cpp')
                db.checkout_commit(repo_dir, commit_id)

                target_name = FuncNameParser.run(func_code, 'cpp' if lang in ('c++', 'cpp') else lang) or ''
                if not target_name:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    continue

                if lang in ('c', 'cpp', 'c++'):
                    edges = cflow_edges(repo_dir, rel_file)
                elif lang == 'java':
                    edges = jacg_edges(repo_dir, rel_file)
                elif lang == 'python':
                    edges = pycg_edges(repo_dir, rel_file)
                else:
                    edges = []

                callers, callees = neighbors_for_target(edges, target_name)

                repo_ctx = {"callee": [], "caller": []}
                for name in callees[:budget]:
                    # Try to find file by scanning repo for name occurrence (best-effort)
                    file_path = repo_dir / rel_file  # same file heuristic first
                    code = extract_snippet(file_path, name)
                    repo_ctx["callee"].append({"file": str(rel_file), "function": code})
                for name in callers[:budget]:
                    file_path = repo_dir / rel_file
                    code = extract_snippet(file_path, name)
                    repo_ctx["caller"].append({"file": str(rel_file), "function": code})

                if repo_ctx["callee"] or repo_ctx["caller"]:
                    obj['repository'] = repo_ctx
                    stats['updated'] += 1
            except Exception:
                pass

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats['processed'] += 1

    return stats


def main():
    ap = argparse.ArgumentParser(description='Enrich a subset using cflow / Java-ACG / PyCG call graphs')
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', default='data/FuncFileRepo.cg.jsonl')
    ap.add_argument('-n', '--limit', type=int, default=50)
    ap.add_argument('--tags', default='')
    ap.add_argument('--languages', default='c,cpp,java,python')
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--budget', type=int, default=50)
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    languages = [l.strip().lower() for l in args.languages.split(',') if l.strip()]
    stats = enrich_with_callgraph(args.input, args.output, args.limit, tags, languages, args.depth, args.budget)
    print(stats)


if __name__ == '__main__':
    main()
