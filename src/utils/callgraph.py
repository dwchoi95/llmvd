"""Real call-graph extraction around a target function at an exact commit.

`CallGraph.collect(row)` checks out `row['project']` at `row['commit_id']`
(shallow blobless clone, LFS disabled), extracts who-calls-whom edges with the
per-language tool from the ReposVul paper where applicable —

    * C / C++  -> GNU cflow
    * Python   -> PyCG
    * Java     -> tree-sitter call extraction (JACG/javacg need compiled
                  bytecode, unavailable for arbitrary commit snapshots)

— and fills `row['repository'] = {"callee": [...], "caller": [...]}` with the
neighbors of the target function(s), their bodies resolved against a
TreeSitter definition index of the checkout. Failures always leave an EMPTY
context, never an approximation.
"""
from __future__ import annotations

import os
import re
import sys
import json
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .treesitter import TreeSitter

ROOT = Path(__file__).resolve().parents[2]
CFLOW_BIN = os.environ.get("CFLOW_BIN", str(ROOT / "tools" / "cflow-1.7" / "src" / "cflow"))

LANG_NORM = {"c": "c", "cpp": "cpp", "c++": "cpp", "java": "java", "python": "python"}
SRC_EXTS = {
    "c": (".c", ".h"),
    "cpp": (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"),
    "java": (".java",),
    "python": (".py",),
}

MAX_FILES = int(os.environ.get("MF_MAX_FILES", "400"))
SUBPROC_TIMEOUT = int(os.environ.get("MF_TIMEOUT", "120"))


def _run(cmd: List[str], cwd: str | None = None, timeout: int | None = None) -> str:
    """Run in an own process group; on timeout kill the whole group (cflow +
    preprocessor can keep the stdout pipe open through grandchildren)."""
    p = subprocess.Popen(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout or SUBPROC_TIMEOUT)
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


class CallGraph:
    """Per-row call-graph collection over exact-commit checkouts."""

    def __init__(self, repos_dir: str = "codeql", budget: int = 30):
        self.repos_dir = Path(repos_dir).resolve()
        self.budget = budget      # max neighbors per side
        self._git_env = {"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        self._checkout_cache: Dict[str, Any] = {}

    # ---- git management ------------------------------------------------ #
    def _git(self, cmd: List[str], ignore_error: bool = False,
             timeout: int | None = None) -> str:
        env = os.environ.copy()
        env.update(self._git_env)
        r = subprocess.run(cmd, text=True, capture_output=True, env=env,
                           timeout=timeout)
        if r.returncode != 0 and not ignore_error:
            raise RuntimeError(f"[git failed] {' '.join(cmd)}\n{r.stderr[:300]}")
        return r.stdout.strip()

    def ensure_repo(self, owner_repo: str, language: str) -> Path:
        repo_dir = self.repos_dir / language / "repo" / owner_repo.replace("/", "__")
        if not repo_dir.exists():
            self._git(["git", "clone", "--filter=blob:none", "--no-tags",
                       f"https://github.com/{owner_repo}.git", str(repo_dir)],
                      timeout=900)
        for k, v in (("filter.lfs.process", ""), ("filter.lfs.required", "false"),
                     ("filter.lfs.smudge", "cat"), ("lfs.fetchexclude", "*")):
            self._git(["git", "-C", str(repo_dir), "config", "--local", k, v],
                      ignore_error=True)
        return repo_dir

    def checkout(self, repo_dir: Path, commit_id: str):
        try:
            self._git(["git", "-C", str(repo_dir), "rev-parse", "--verify", commit_id])
        except RuntimeError:
            self._git(["git", "-C", str(repo_dir), "fetch", "origin", commit_id,
                       "--depth=1"], timeout=300)
        self._git(["git", "-C", str(repo_dir), "reset", "--hard"])
        self._git(["git", "-C", str(repo_dir), "clean", "-fdx"])
        self._git(["git", "-C", str(repo_dir), "checkout", "--detach", commit_id])
        self._git(["git", "-C", str(repo_dir), "reset", "--hard", commit_id])
        self._git(["git", "-C", str(repo_dir), "clean", "-fdx"])

    # ---- source gathering ----------------------------------------------- #
    @staticmethod
    def gather_files(target_path: Path, lang_key: str) -> List[Path]:
        """The target file first, then its directory siblings (bounded)."""
        exts = SRC_EXTS[lang_key]
        files: List[Path] = []
        if target_path.exists():
            files.append(target_path)
        try:
            for p in sorted(target_path.parent.iterdir()):
                if p.is_file() and p.suffix.lower() in exts and p != target_path:
                    files.append(p)
                if len(files) >= MAX_FILES:
                    break
        except Exception:
            pass
        return files

    # ---- edge extraction ------------------------------------------------ #
    _CFLOW_LINE = re.compile(r"^(\s*)([A-Za-z_]\w*)")

    @classmethod
    def cflow_edges(cls, files: List[Path]) -> List[Tuple[str, str]]:
        if not files:
            return []
        try:
            # --brief prevents the call tree exploding on recursive programs
            out = _run([CFLOW_BIN, "--brief", "--omit-arguments",
                        *[str(f) for f in files]])
        except Exception:
            return []
        edges: List[Tuple[str, str]] = []
        stack: List[Tuple[int, str]] = []
        for line in out.splitlines():
            m = cls._CFLOW_LINE.match(line)
            if not m:
                continue
            indent, name = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if stack:
                edges.append((stack[-1][1], name))
            stack.append((indent, name))
        return edges

    @staticmethod
    def pycg_edges(repo_dir: Path, entry: Path, src_dir: Path) -> List[Tuple[str, str]]:
        out_json = repo_dir / ".pycg.json"
        try:
            _run([sys.executable, "-m", "pycg", "--package", str(src_dir),
                  str(entry), "-o", str(out_json)])
            data = json.loads(out_json.read_text("utf-8", "ignore"))
        except Exception:
            return []
        finally:
            try:
                out_json.unlink()
            except Exception:
                pass
        edges = []
        cg = data.get("callgraph", data) if isinstance(data, dict) else {}
        for src, tgts in cg.items():
            s = str(src).split(".")[-1]
            edges.extend((s, str(t).split(".")[-1]) for t in tgts or [])
        return edges

    @staticmethod
    def java_edges(files: List[Path]) -> List[Tuple[str, str]]:
        ts = TreeSitter
        parser = ts.parser("java")
        edges: List[Tuple[str, str]] = []
        for fp in files:
            try:
                tree, data = ts.parse(parser, fp.read_bytes())
            except Exception:
                continue
            for node in ts.walk(ts.root(tree)):
                if ts.kind(node) not in ("method_declaration", "constructor_declaration"):
                    continue
                nm = ts.field(node, "name")
                if nm is None:
                    continue
                caller = ts.text(nm, data)
                for desc in ts.walk(node):
                    if ts.kind(desc) == "method_invocation":
                        inm = ts.field(desc, "name")
                        if inm is not None and caller:
                            callee = ts.text(inm, data)
                            if callee:
                                edges.append((caller, callee))
        return edges

    # ---- neighbor resolution -------------------------------------------- #
    def resolve_neighbors(self, edges, index, targets: set, target_code: str):
        callee_names, caller_names = [], []
        seen_ce, seen_ca = set(), set()
        for caller, callee in edges:
            if caller in targets and callee not in targets and callee not in seen_ce:
                seen_ce.add(callee)
                callee_names.append(callee)
            if callee in targets and caller not in targets and caller not in seen_ca:
                seen_ca.add(caller)
                caller_names.append(caller)

        def collect(names):
            out, sigs = [], set()
            for n in names:
                for e in index.get(n, []):
                    if e["function"].strip() == (target_code or "").strip():
                        continue
                    sig = (e["file"], hash(e["function"]))
                    if sig not in sigs:
                        sigs.add(sig)
                        out.append({"file": e["file"], "function": e["function"]})
                    break
                if len(out) >= self.budget:
                    break
            return out

        return {"callee": collect(callee_names), "caller": collect(caller_names)}

    # ---- per-row driver --------------------------------------------------- #
    def collect(self, row: Dict[str, Any]) -> bool:
        """Fill row['repository'] with the extracted call graph; True if
        non-empty. Failures leave an EMPTY context (never an approximation)."""
        lang_key = LANG_NORM.get((row.get("language") or "").lower())
        if lang_key is None:
            return False
        project, commit_id = row.get("project"), row.get("commit_id")
        rel_file, func_code = row.get("file_name"), row.get("function") or ""
        if not (project and commit_id and rel_file and func_code):
            return False
        try:
            targets = TreeSitter.function_names(func_code, lang_key)
            if not targets:
                return False

            key = f"{project}@{commit_id}"
            if self._checkout_cache.get("key") != key:
                repo_dir = self.ensure_repo(project, lang_key)
                self.checkout(repo_dir, commit_id)
                self._checkout_cache = {"key": key, "repo_dir": repo_dir}
            repo_dir = self._checkout_cache["repo_dir"]

            target_path = (repo_dir / rel_file).resolve()
            files = self.gather_files(target_path, lang_key)
            if not files:
                return False
            index = TreeSitter.build_def_index(files, repo_dir, lang_key)
            if lang_key in ("c", "cpp"):
                edges = self.cflow_edges(files)
            elif lang_key == "python":
                edges = self.pycg_edges(repo_dir, target_path, target_path.parent)
            else:
                edges = self.java_edges(files)
            row["repository"] = self.resolve_neighbors(edges, index, targets, func_code)
        except Exception:
            row["repository"] = {"callee": [], "caller": []}
            return False
        return bool(row["repository"]["callee"] or row["repository"]["caller"])
