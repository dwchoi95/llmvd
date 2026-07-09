import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from .database import Database
from .funcNameParser import FuncNameParser


def read_lines(path: Path, start: int, end: int) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        start_i = max(1, int(start)) - 1
        end_i = min(len(lines), int(end))
        return ''.join(lines[start_i:end_i])
    except Exception:
        return ''


def _ensure_db(db: Database, language: str, project: str, commit_id: str):
    # Prepare repo
    repo_dir = db.ensure_repo(project, language)
    db.checkout_commit(repo_dir, commit_id)

    # Paths
    codeql_lang = "c-cpp" if language in ("c", "cpp", "c++") else language
    repo_db_dir = Path(f"codeql/{codeql_lang}/{project.replace('/', '__')}")
    script_dir = repo_db_dir.parent  # codeql/{codeql_lang}

    # Build DB if missing
    if not repo_db_dir.exists() or not any(repo_db_dir.iterdir()):
        build_tpl = Path(f"codeql/{language}/build.sh").read_text()
        build_cmd = build_tpl.format(script_dir=str(script_dir.resolve()), repo=str(repo_db_dir.resolve()), repo_dir=str(repo_dir.resolve()))
        db.subprocess_run(["bash", "-lc", build_cmd])

    return repo_dir, repo_db_dir, script_dir


def _query_neighbors(db: Database, language: str, repo_dir: Path, repo_db_dir: Path, script_dir: Path,
                     file_name: str, func_name: str) -> List[Dict[str, Any]]:
    # Prepare query
    calls_ql = Path(f"codeql/{language}/template.ql").read_text()
    calls_ql = calls_ql.replace('predicate targetFileName() { result = "" }',
                                f'predicate targetFileName() {{ result = "{file_name}" }}')
    calls_ql = calls_ql.replace('predicate targetFunctionName() { result = "" }',
                                f'predicate targetFunctionName() {{ result = "{func_name}" }}')
    calls_path = script_dir / "calls.ql"
    calls_path.write_text(calls_ql)

    # Run query
    run_tpl = Path(f"codeql/{language}/run.sh").read_text()
    run_cmd = run_tpl.format(script_dir=str(script_dir.resolve()), repo=str(repo_db_dir.resolve()))
    out = db.subprocess_run(["bash", "-lc", run_cmd], cwd=str(script_dir))

    # Parse CSV
    csv_path = script_dir / "calls.csv"
    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def run_codeql_for_sample(db: Database, language: str, project: str, commit_id: str,
                          file_name: str, function_code: str,
                          depth: int = 1, budget: int = 50) -> Dict[str, List[Dict[str, str]]]:
    repo_dir, repo_db_dir, script_dir = _ensure_db(db, language, project, commit_id)

    # Resolve target function name
    func_name = FuncNameParser.run(function_code, "cpp" if language in ("c++", "cpp") else language)
    if not func_name:
        return {"callee": [], "caller": []}

    # BFS over direct neighbors using CodeQL query per node
    from collections import deque, defaultdict
    Node = tuple[str, str]  # (file, name)
    start: Node = (file_name, func_name)
    q = deque([(start, 0)])
    seen: set[Node] = {start}
    collected: Dict[str, List[Dict[str, str]]] = {"callee": [], "caller": []}

    while q and (len(collected["callee"]) + len(collected["caller"]) < budget):
        (fpath, fname), d = q.popleft()
        rows = _query_neighbors(db, language, repo_dir, repo_db_dir, script_dir, fpath, fname)
        for r in rows:
            kind = r.get('kind')
            rel = r.get('file')
            start_line = int(r.get('startLine') or 0)
            end_line = int(r.get('endLine') or 0)
            code = read_lines(repo_dir / rel, start_line, end_line)
            if not code:
                code = f"// {r.get('name', '')}"
            entry = {"file": rel, "function": code}
            if kind in collected:
                # dedup by (file, code hash)
                sig = (entry["file"], hash(entry["function"]))
                exists = any((x["file"], hash(x["function"])) == sig for x in collected[kind])
                if not exists:
                    collected[kind].append(entry)

            # enqueue neighbor for next depth
            if d + 1 <= depth:
                nname = r.get('name') or ''
                neigh: Node = (rel, nname)
                if nname and neigh not in seen:
                    seen.add(neigh)
                    q.append((neigh, d + 1))

        if d >= depth:
            continue

    return collected


def enrich_dataset(input_jsonl: str, output_jsonl: str, limit: int, tags: List[str], languages: List[str], depth: int, budget: int) -> Dict[str, int]:
    stats = {"processed": 0, "updated": 0}
    db = Database()
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(input_jsonl, 'r', encoding='utf-8') as fin, open(output_jsonl, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if stats["processed"] >= limit:
                # copy-through
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
            file_name = obj.get('file_name')
            function_code = obj.get('function') or ''
            if not (project and commit_id and file_name and function_code):
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            try:
                repo_ctx = run_codeql_for_sample(db, lang, project, commit_id, file_name, function_code, depth=depth, budget=budget)
                if (repo_ctx.get('callee') or repo_ctx.get('caller')):
                    obj['repository'] = repo_ctx
                    stats['updated'] += 1
            except Exception as e:
                # do not fail the whole pipeline; keep original
                pass

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats['processed'] += 1

    return stats


def main():
    ap = argparse.ArgumentParser(description='Enrich a subset of dataset with true caller/callee via CodeQL')
    ap.add_argument('-i', '--input', required=True, help='Input JSONL (e.g., data/FuncFileRepo.jsonl)')
    ap.add_argument('-o', '--output', default='data/FuncFileRepo.true.jsonl', help='Output JSONL path')
    ap.add_argument('-n', '--limit', type=int, default=50, help='Max samples to enrich')
    ap.add_argument('--depth', type=int, default=1, help='BFS depth for neighbors')
    ap.add_argument('--budget', type=int, default=50, help='Max total neighbors to collect')
    ap.add_argument('--tags', default='', help='Comma-separated tags (SFSF,MFSF,MFMF). Empty for all')
    ap.add_argument('--languages', default='c,cpp,c++', help='Comma-separated languages to process')
    args = ap.parse_args()

    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    languages = [l.strip().lower() for l in args.languages.split(',') if l.strip()]
    stats = enrich_dataset(args.input, args.output, args.limit, tags, languages, args.depth, args.budget)
    print(stats)


if __name__ == '__main__':
    main()
