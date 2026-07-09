"""Build real caller/callee context for MFSF/MFMF samples.

Edges (who-calls-whom) are extracted with the tools defined in the ReposVul
paper where applicable:

    * C / C++  -> GNU cflow
    * Python   -> PyCG
    * Java     -> tree-sitter call extraction (JACG/javacg require compiled
                  bytecode, which is unavailable for arbitrary commit snapshots)

The function *code* for each neighbor is resolved against a tree-sitter index of
the checked-out repository (name -> [{file, function, ...}]), so cross-file
callers/callees in MFMF samples get the correct file path and full body.

Note: the tree-sitter binding shipped via tree_sitter_language_pack in this
environment exposes node fields as *methods* (node.kind(), node.start_byte(),
node.root_node(), ...) and parses ``str`` (not bytes). The thin accessors below
tolerate both that binding and the classic attribute-based binding.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tree_sitter_language_pack import get_parser

from .database import Database

ROOT = Path(__file__).resolve().parents[2]
CFLOW_BIN = os.environ.get("CFLOW_BIN", str(ROOT / "tools" / "cflow-1.7" / "src" / "cflow"))

LANG_NORM = {"c": "c", "cpp": "cpp", "c++": "cpp", "java": "java", "python": "python"}
TS_LANG = {"c": "c", "cpp": "cpp", "java": "java", "python": "python"}

SRC_EXTS = {
    "c": (".c", ".h"),
    "cpp": (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"),
    "java": (".java",),
    "python": (".py",),
}

DEF_NODE_TYPES = {"function_definition", "method_declaration", "constructor_declaration"}

MAX_FILES = int(os.environ.get("MF_MAX_FILES", "400"))
SUBPROC_TIMEOUT = int(os.environ.get("MF_TIMEOUT", "120"))


# --------------------------------------------------------------------------- #
# tree-sitter accessors (tolerate method-style and attribute-style bindings)
# --------------------------------------------------------------------------- #
def _call(v):
    return v() if callable(v) else v


def _kind(n) -> str:
    k = getattr(n, "kind", None)
    if k is None:
        k = getattr(n, "type", None)
    return _call(k)


def _sb(n) -> int:
    return _call(getattr(n, "start_byte"))


def _eb(n) -> int:
    return _call(getattr(n, "end_byte"))


def _row(pt) -> int:
    if pt is None:
        return 0
    try:
        return pt[0]
    except (TypeError, KeyError):
        return getattr(pt, "row", 0)


def _start_row(n) -> int:
    sp = _call(getattr(n, "start_point", None))
    if sp is None:
        sp = _call(getattr(n, "start_position"))
    return _row(sp)


def _end_row(n) -> int:
    ep = _call(getattr(n, "end_point", None))
    if ep is None:
        ep = _call(getattr(n, "end_position"))
    return _row(ep)


def _field(n, name):
    return n.child_by_field_name(name)


def _named_children(n) -> List[Any]:
    cnt = _call(getattr(n, "named_child_count"))
    return [n.named_child(i) for i in range(cnt)]


def _root(tree):
    return _call(getattr(tree, "root_node"))


def _parse(parser, data: bytes):
    text = data.decode("utf-8", "ignore")
    try:
        return parser.parse(text), text.encode("utf-8")
    except TypeError:
        enc = text.encode("utf-8")
        return parser.parse(enc), enc


def _text(node, data: bytes) -> str:
    return data[_sb(node):_eb(node)].decode("utf-8", "ignore")


def _walk(node):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(_named_children(cur))


def _norm_lang(language: str) -> str | None:
    return LANG_NORM.get((language or "").lower())


# --------------------------------------------------------------------------- #
# name resolution
# --------------------------------------------------------------------------- #
def _def_name(node, data: bytes, lang_key: str) -> str | None:
    k = _kind(node)
    if k in ("method_declaration", "constructor_declaration"):
        nm = _field(node, "name")
        return _text(nm, data) if nm is not None else None
    if k == "function_definition":
        if lang_key == "python":
            nm = _field(node, "name")
            return _text(nm, data) if nm is not None else None
        # C / C++: unwrap declarator chain to the function_declarator's name
        d = _field(node, "declarator")
        guard = 0
        while d is not None and _kind(d) in ("pointer_declarator", "parenthesized_declarator", "reference_declarator") and guard < 8:
            inner = _field(d, "declarator")
            d = inner if inner is not None else (_named_children(d)[0] if _named_children(d) else None)
            guard += 1
        if d is not None and _kind(d) == "function_declarator":
            inner = _field(d, "declarator")
            if inner is not None:
                ik = _kind(inner)
                if ik in ("qualified_identifier", "field_identifier", "identifier", "destructor_name", "operator_name"):
                    # for qualified_identifier take trailing identifier
                    ids = [c for c in _walk(inner) if _kind(c) == "identifier"]
                    return _text(ids[-1], data) if ids else _text(inner, data)
        if d is not None and _kind(d) == "identifier":
            return _text(d, data)
    return None


def extract_defs(data: bytes, lang_key: str, parser) -> List[Tuple[str, Any]]:
    """Return [(name, def_node)] for every function/method definition."""
    tree, data = _parse(parser, data)
    out: List[Tuple[str, Any]] = []
    for node in _walk(_root(tree)):
        if _kind(node) in DEF_NODE_TYPES:
            name = _def_name(node, data, lang_key)
            if name:
                out.append((name, node))
    return out, data


def target_func_name(func_code: str, lang_key: str) -> str | None:
    parser = get_parser(TS_LANG[lang_key])
    raw = func_code.encode("utf-8")
    try:
        defs, data = extract_defs(raw, lang_key, parser)
    except Exception:
        return None
    if defs:
        return defs[0][0]
    # Java bare method snippet may not parse without a class wrapper
    if lang_key == "java":
        try:
            defs, data = extract_defs(b"class _W{\n" + raw + b"\n}", lang_key, parser)
            if defs:
                return defs[0][0]
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
# repository function-definition index
# --------------------------------------------------------------------------- #
def _gather_files(target_path: Path, lang_key: str) -> List[Path]:
    src_dir = target_path.parent
    exts = SRC_EXTS[lang_key]
    files: List[Path] = []
    if target_path.exists():
        files.append(target_path)
    try:
        for p in sorted(src_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts and p != target_path:
                files.append(p)
            if len(files) >= MAX_FILES:
                break
    except Exception:
        pass
    return files


def build_def_index(files: List[Path], repo_dir: Path, lang_key: str) -> Dict[str, List[Dict[str, Any]]]:
    parser = get_parser(TS_LANG[lang_key])
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fp in files:
        try:
            raw = fp.read_bytes()
            defs, data = extract_defs(raw, lang_key, parser)
        except Exception:
            continue
        try:
            rel = str(fp.relative_to(repo_dir))
        except Exception:
            rel = fp.name
        for name, node in defs:
            code = _text(node, data)
            if not code:
                continue
            index[name].append({
                "file": rel,
                "function": code,
                "start": _start_row(node) + 1,
                "end": _end_row(node) + 1,
            })
    return index


# --------------------------------------------------------------------------- #
# edge extraction
# --------------------------------------------------------------------------- #
def _run(cmd: List[str], cwd: str | None = None) -> str:
    """Run a command in its own process group; on timeout kill the whole group.

    Plain subprocess.run(timeout=) can deadlock when the child spawns
    grandchildren that keep the stdout pipe open (e.g. cflow + preprocessor),
    so we manage the process group explicitly.
    """
    p = subprocess.Popen(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=SUBPROC_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            p.kill()
        p.communicate()
        raise RuntimeError(f"timeout: {' '.join(cmd[:2])}")
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed ({p.returncode}): {' '.join(cmd[:3])}...\n{(err or '')[:300]}")
    return out


_CFLOW_LINE = re.compile(r"^(\s*)([A-Za-z_]\w*)")


def cflow_edges(files: List[Path]) -> List[Tuple[str, str]]:
    if not files:
        return []
    try:
        # --brief: do not re-expand a subtree already shown (prevents the call
        # tree from exploding combinatorially on recursive programs).
        out = _run([CFLOW_BIN, "--brief", "--omit-arguments", *[str(f) for f in files]])
    except Exception:
        return []
    edges: List[Tuple[str, str]] = []
    stack: List[Tuple[int, str]] = []
    for line in out.splitlines():
        m = _CFLOW_LINE.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        name = m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            edges.append((stack[-1][1], name))
        stack.append((indent, name))
    return edges


def pycg_edges(repo_dir: Path, entry: Path, src_dir: Path) -> List[Tuple[str, str]]:
    out_json = repo_dir / ".mf_pycg.json"
    try:
        _run([sys.executable, "-m", "pycg", "--package", str(src_dir), str(entry),
              "-o", str(out_json)])
        data = json.loads(out_json.read_text("utf-8", "ignore"))
    except Exception:
        return []
    finally:
        try:
            out_json.unlink()
        except Exception:
            pass
    edges: List[Tuple[str, str]] = []
    cg = data.get("callgraph", data) if isinstance(data, dict) else {}
    for src, tgts in cg.items():
        s = str(src).split(".")[-1]
        for t in tgts or []:
            edges.append((s, str(t).split(".")[-1]))
    return edges


def java_edges(files: List[Path]) -> List[Tuple[str, str]]:
    parser = get_parser("java")
    edges: List[Tuple[str, str]] = []
    for fp in files:
        try:
            raw = fp.read_bytes()
            tree, data = _parse(parser, raw)
        except Exception:
            continue
        for node in _walk(_root(tree)):
            if _kind(node) not in ("method_declaration", "constructor_declaration"):
                continue
            nm = _field(node, "name")
            if nm is None:
                continue
            caller = _text(nm, data)
            for desc in _walk(node):
                if _kind(desc) == "method_invocation":
                    inm = _field(desc, "name")
                    if inm is not None:
                        callee = _text(inm, data)
                        if caller and callee:
                            edges.append((caller, callee))
    return edges


# --------------------------------------------------------------------------- #
# neighbor resolution
# --------------------------------------------------------------------------- #
def _lookup(index: Dict[str, List[Dict[str, Any]]], name: str, exclude_code: str) -> Dict[str, Any] | None:
    for entry in index.get(name, []):
        if entry["function"].strip() != (exclude_code or "").strip():
            return {"file": entry["file"], "function": entry["function"]}
    return None


def resolve_neighbors(edges: List[Tuple[str, str]], index: Dict[str, List[Dict[str, Any]]],
                      targets, target_code: str, budget: int) -> Dict[str, List[Dict[str, str]]]:
    """Union the callee/caller neighbors over a set of target function names.

    `targets` may be a single name (str) or an iterable of names — the merged
    `function` column can contain several target functions, so neighbors are
    collected for all of them and deduped.
    """
    tset = {targets} if isinstance(targets, str) else set(targets)
    callee_names: List[str] = []
    caller_names: List[str] = []
    seen_ce, seen_ca = set(), set()
    for caller, callee in edges:
        if caller in tset and callee not in tset and callee not in seen_ce:
            seen_ce.add(callee)
            callee_names.append(callee)
        if callee in tset and caller not in tset and caller not in seen_ca:
            seen_ca.add(caller)
            caller_names.append(caller)

    def collect(names: List[str]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        sigs = set()
        for n in names:
            e = _lookup(index, n, target_code)
            if not e:
                continue
            sig = (e["file"], hash(e["function"]))
            if sig in sigs:
                continue
            sigs.add(sig)
            out.append(e)
            if len(out) >= budget:
                break
        return out

    return {"callee": collect(callee_names), "caller": collect(caller_names)}


# --------------------------------------------------------------------------- #
# per-row enrichment + driver
# --------------------------------------------------------------------------- #
def enrich_row(db: Database, obj: Dict[str, Any], budget: int, _cache: Dict[str, Any]) -> Tuple[bool, str]:
    lang_key = _norm_lang(obj.get("language"))
    if lang_key is None:
        return False, "lang"
    project = obj.get("project")
    commit_id = obj.get("commit_id")
    rel_file = obj.get("file_name")
    func_code = obj.get("function") or ""
    if not (project and commit_id and rel_file and func_code):
        return False, "missing-field"

    # the merged `function` column may hold several target functions — collect
    # all of their names so neighbors are unioned over the whole set.
    try:
        parser = get_parser(TS_LANG[lang_key])
        defs, _ = extract_defs(func_code.encode("utf-8"), lang_key, parser)
        target_names = {nm for nm, _node in defs if nm}
    except Exception:
        target_names = set()
    if not target_names:
        nm = target_func_name(func_code, lang_key)
        if nm:
            target_names = {nm}
    if not target_names:
        return False, "no-target-name"

    key = f"{project}@{commit_id}"
    if _cache.get("key") != key:
        repo_dir = db.ensure_repo(project, lang_key)
        db.checkout_commit(repo_dir, commit_id, with_submodules=False)
        _cache.clear()
        _cache["key"] = key
        _cache["repo_dir"] = repo_dir
    repo_dir: Path = _cache["repo_dir"]

    target_path = (repo_dir / rel_file).resolve()
    files = _gather_files(target_path, lang_key)
    if not files:
        return False, "no-files"

    index = build_def_index(files, repo_dir, lang_key)

    if lang_key in ("c", "cpp"):
        edges = cflow_edges(files)
    elif lang_key == "python":
        edges = pycg_edges(repo_dir, target_path, target_path.parent)
    else:
        edges = java_edges(files)

    repo_ctx = resolve_neighbors(edges, index, target_names, func_code, budget)
    obj["repository"] = repo_ctx
    return bool(repo_ctx["callee"] or repo_ctx["caller"]), "ok" if (repo_ctx["callee"] or repo_ctx["caller"]) else "no-neighbors"


def _row_key(r: Dict[str, Any]) -> str:
    return "|".join([
        str(r.get("project")), str(r.get("commit_id")), str(r.get("file_name")),
        str(r.get("vulnerable")), str(hash(r.get("function", ""))),
    ])


def run(input_path: str, output_path: str, tags: List[str], limit: int, budget: int,
        resume: bool = False) -> Dict[str, int]:
    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if tags and obj.get("tag") not in tags:
                continue
            rows.append(obj)
    rows.sort(key=lambda r: (r.get("project") or "", r.get("commit_id") or "", r.get("file_name") or ""))
    if limit > 0:
        rows = rows[:limit]

    done: set = set()
    if resume and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(_row_key(json.loads(line)))
                except Exception:
                    continue
        print(f"[resume] {len(done)} rows already done", flush=True)

    db = Database()
    cache: Dict[str, Any] = {}
    stats: Dict[str, int] = defaultdict(int)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    with open(output_path, mode, encoding="utf-8") as fout:
        for i, obj in enumerate(rows, 1):
            if resume and _row_key(obj) in done:
                stats["skipped"] += 1
                continue
            stats["processed"] += 1
            try:
                updated, reason = enrich_row(db, obj, budget, cache)
            except Exception as e:
                updated, reason = False, f"err:{type(e).__name__}"
            # never leave the old approximated context behind: real-or-empty only
            if reason not in ("ok", "no-neighbors"):
                obj["repository"] = {"callee": [], "caller": []}
            stats[f"reason:{reason}"] += 1
            if updated:
                stats["updated"] += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(rows)}] updated={stats['updated']} processed={stats['processed']}", flush=True)
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(description="Build real caller/callee for MFSF/MFMF via cflow/PyCG/tree-sitter")
    ap.add_argument("-i", "--input", required=True, help="Input JSONL (with tag field)")
    ap.add_argument("-o", "--output", required=True, help="Output JSONL")
    ap.add_argument("--tags", default="MFSF,MFMF", help="Comma-separated tags to process")
    ap.add_argument("-n", "--limit", type=int, default=0, help="Max rows (0 = all)")
    ap.add_argument("--budget", type=int, default=30, help="Max neighbors per side")
    ap.add_argument("--resume", action="store_true", help="Skip rows already present in output")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    stats = run(args.input, args.output, tags, args.limit, args.budget, resume=args.resume)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
