import os
import json
import logging
import asyncio
from pathlib import Path
from time import perf_counter
import pandas as pd
from codebleu import calc_codebleu
from tqdm.asyncio import tqdm as tqdm_async
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    matthews_corrcoef, balanced_accuracy_score,
)
import numpy as np
from prettytable import PrettyTable

from .evaluator import Evaluator
from ..llms import GPT, CLAUDE, GEMINI, OLLAMA
from ..prompts import PromptManager, DEFAULT_STRATEGIES, get_strategy


class Detector:
    """Runs the scope x strategy grid for ONE served model.

    The experiment crosses three input scopes (function/file/repository) with
    the prompting strategies given (`zero`/`rag`/`sft`). `reasoning` marks
    whether the served model emits a thinking trace before its verdict — a
    MODEL property, so it is set per Detector, not per strategy: reasoning
    models get a large generation budget and free-form output (verdict parsed
    after the thinking trace), direct models get schema-forced JSON.
    """

    _SCOPES = ["function", "file", "repository"]

    def __init__(
        self,
        llm: str,
        temperature: float | None,
        dataset_path: str,
        save_dir: str = "results",
        async_limit: int = 32,
        strategies: list[str] | None = None,
        reasoning: bool = False,
        example_path: str | None = None,
        prompt_root: str = "src/prompts",
        control_token: str | None = None,
        label: str | None = None,
    ):
        self.dataset_path = dataset_path
        self.benchmark = Path(dataset_path).stem
        # `label` names the results subdir (default = served model id). A hybrid
        # model served once but run in two modes writes to two labels so the
        # non-reasoning and reasoning results do not collide.
        self.results_dir = Path(save_dir) / (label or llm)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.__scopes = self._SCOPES
        self.strategies = [get_strategy(s) for s in (strategies or DEFAULT_STRATEGIES)]
        self.reasoning = reasoning
        # hybrid models toggle thinking via a control token prepended to the
        # system prompt (e.g. NVIDIA Nemotron: "/no_think" off, "/think" on).
        # Vendor-split checkpoints (Qwen Instruct/Thinking, Mistral Small/
        # Magistral) need none.
        self.control_token = control_token
        self.prompt_root = prompt_root
        # thinking models need room to reason before the verdict; direct models
        # emit only the short JSON object.
        self.max_tokens = 4096 if reasoning else 16
        self.__rows = ["Trial", "Strategy", "Scope", "Accuracy", "Precision",
                       "Recall", "F1-score", "MCC", "Pos.Rate",
                       "AVG Tokens", "AVG Time (sec)"]
        self.model = self._select_model(llm, temperature)
        # only the local (open-weight) client exposes token logprobs -> scores
        self.__scored = hasattr(self.model, "run_scored")
        self.pm = PromptManager()
        # per-strategy system prompts (from that strategy's prompt folder),
        # with the hybrid control token prepended when set
        prefix = f"{self.control_token}\n" if self.control_token else ""
        self.strategy_systems = {
            s.name: prefix + self.pm.render(file=self._tpl(s.prompt_dir, "system"))
            for s in self.strategies
        }
        self.dataset_df = self._load_data(dataset_path)
        self.examples = self._load_examples(example_path)
        self.evaluator = Evaluator(llm)
        self.async_limit = async_limit

    # ------------------------------------------------------------------ #
    def _tpl(self, prompt_dir: str, name: str) -> str:
        return f"{self.prompt_root}/{prompt_dir}/{name}.md"

    def _select_model(self, llm: str, temperature: float):
        if llm.startswith("gpt"):
            return GPT(llm, temperature)
        elif llm.startswith("claude"):
            return CLAUDE(llm, temperature)
        elif llm.startswith("gemini"):
            return GEMINI(llm, temperature)
        else:
            return OLLAMA(llm, temperature)

    def _load_data(self, dataset_path: str) -> pd.DataFrame:
        return pd.read_json(dataset_path, lines=True)

    def _load_examples(self, example_path: str | None) -> dict:
        """test_id -> {'vul': code, 'sec': code} for the RAG strategy.

        Auto-discovers <benchmark>.example.jsonl next to the dataset when no
        explicit path is given; required only when a `rag` strategy is run.
        """
        need_rag = any(s.uses_examples for s in self.strategies)
        if example_path is None:
            # e.g. data/FuncFileRepo.test.jsonl -> data/FuncFileRepo.example.jsonl
            stem = self.benchmark.rsplit(".", 1)[0] if "." in self.benchmark \
                else self.benchmark
            cand = Path(self.dataset_path).parent / f"{stem}.example.jsonl"
            example_path = str(cand) if cand.exists() else None
        ex: dict = {}
        if example_path and os.path.exists(example_path):
            for line in open(example_path, encoding="utf-8"):
                r = json.loads(line)
                tid = r.get("test_id")
                side = "vul" if r.get("vulnerable") else "sec"
                ex.setdefault(tid, {})[side] = r.get("example", "")
        if need_rag and not ex:
            raise ValueError(
                "rag strategy requested but no example file found; pass "
                "example_path=<benchmark>.example.jsonl")
        return ex

    # ------------------------------------------------------------------ #
    # repository-scope context ranking (CodeBLEU similarity to the target)
    # ------------------------------------------------------------------ #
    def _code_bleu(self, function: str, call: str, language: str) -> float:
        language = language.lower()
        if language == "c++":
            language = "cpp"
        logging.disable(logging.WARNING)
        try:
            return calc_codebleu([function], [call], lang=language)["codebleu"]
        except Exception:
            pass
        finally:
            logging.disable(logging.NOTSET)
        return 0.0

    def priority(self, function: str, calls: list[dict], language: str) -> list[dict]:
        scored = [(c, self._code_bleu(function, c["function"], language)) for c in calls]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    # ------------------------------------------------------------------ #
    # client-side context budget (see class comment on truncation)
    # ------------------------------------------------------------------ #
    MAX_MODEL_LEN = 131072
    GEN_RESERVE = 1024 + 256   # must match src/utils/dataset.py budget

    def _user_budget(self) -> int:
        budget = getattr(self, "_user_budget_cache", None)
        if budget is None:
            sys_max = max(self.evaluator.token_count(s)
                          for s in self.strategy_systems.values())
            budget = self.MAX_MODEL_LEN - self.GEN_RESERVE - sys_max
            self._user_budget_cache = budget
        return budget

    _TRUNC_MARK = "\n... [truncated to fit the context budget] ...\n"

    def _repo_info(self, language, callees, callers) -> str:
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

    def _render(self, strategy, scope, **kw) -> str:
        return self.pm.render(file=self._tpl(strategy.prompt_dir, scope), **kw)

    def _make_user_prompt(self, row: pd.Series, scope: str, strategy):
        """Render the user prompt for one (row, scope, strategy); returns
        (prompt, truncated). Deterministic given the row.

        RAG prepends the two retrieved examples (`vul_example`/`sec_example`)
        via the rag/<scope>.md template; zero/sft use the zero/<scope>.md
        template with the same scope inputs.
        """
        function = row["function"]
        language = row["language"]
        budget = self._user_budget()
        truncated = False

        base_kw = dict(language=language, function=function)
        if strategy.uses_examples:
            ex = self.examples.get(row.get("id"), {})
            base_kw["vul_example"] = ex.get("vul", "")
            base_kw["sec_example"] = ex.get("sec", "")

        if scope == "function":
            user = self._render(strategy, "function", **base_kw)
        elif scope == "file":
            file_code = row["file"]
            user = self._render(strategy, "file", file_code=file_code, **base_kw)
            if self.evaluator.token_count(user) > budget:
                truncated = True
                lo, hi = 0, len(file_code)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    cand = self._render(strategy, "file",
                                        file_code=file_code[:mid] + self._TRUNC_MARK,
                                        **base_kw)
                    if self.evaluator.token_count(cand) <= budget:
                        lo = mid
                    else:
                        hi = mid - 1
                user = self._render(strategy, "file",
                                    file_code=file_code[:lo] + self._TRUNC_MARK,
                                    **base_kw)
        elif scope == "repository":
            repo = row.get("repository", {"callee": [], "caller": []})
            callees = self.priority(function, repo.get("callee") or [], language)
            callers = self.priority(function, repo.get("caller") or [], language)
            user = self._render(strategy, "repository",
                                repository=self._repo_info(language, callees, callers),
                                **base_kw)
            while (self.evaluator.token_count(user) > budget
                   and (callees or callers)):
                truncated = True
                if len(callers) >= len(callees):
                    callers = callers[:-1]
                else:
                    callees = callees[:-1]
                user = self._render(strategy, "repository",
                                    repository=self._repo_info(language, callees, callers),
                                    **base_kw)
        return user, truncated

    # ------------------------------------------------------------------ #
    # the paper's prediction is the TYPED verdict digit parsed from the raw
    # completion (not a logprob threshold); see \S Evaluation Metrics.
    import re as _re
    _DIGIT = _re.compile(r'vulnerable"?\s*:?\s*([01])')
    _ANY01 = _re.compile(r"\b([01])\b")

    @classmethod
    def _typed(cls, raw) -> "bool | None":
        if raw is None:
            return None
        # take the LAST match: free-form (CoT) outputs end with the verdict,
        # and structured outputs contain exactly one match either way
        ms = list(cls._DIGIT.finditer(str(raw))) or list(cls._ANY01.finditer(str(raw)))
        return bool(int(ms[-1].group(1))) if ms else None

    async def __task(self, idx: int, row: pd.Series, scope: str, strategy):
        row_dict = row.to_dict()
        system = self.strategy_systems[strategy.name]
        # prompt building runs CodeBLEU ranking (CPU-bound) for repo scope;
        # keep it off the event loop so it can't stall in-flight requests
        user, truncated = await asyncio.to_thread(
            self._make_user_prompt, row, scope, strategy)
        start = perf_counter()
        if self.__scored:
            # free-form strategies (cot) need an in-channel scratchpad even on
            # non-reasoning models: no schema forcing, reasoning-sized budget
            free_form = self.reasoning or getattr(strategy, "free_form", False)
            mt = 4096 if free_form else self.max_tokens
            label, score, raw = await self.model.run_scored(
                system, user, max_tokens=mt,
                reasoning=free_form)
            predict = self._typed(raw)
        else:
            label = await self.model.run(system, user)
            predict = bool(label) if label is not None else None
        duration = perf_counter() - start
        # slim per-row record — matches results/{model}/{strategy}/*.jsonl
        slim = {
            "id": row_dict.get("id"),
            "language": row_dict.get("language"),
            "scope": scope,
            "prompts": strategy.name,
            "complexity": row_dict.get("tag"),
            "label": bool(row_dict.get("vulnerable")),
            "predict": predict,
            "tokens": self.evaluator.token_count(f"{system}\n\n{user}"),
            "time(s)": duration,
        }
        return idx, slim

    @staticmethod
    def _json_default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    def _row_key(self, r: dict, scope: str = None, strategy: str = None) -> str:
        s = scope if scope is not None else r.get('scope')
        st = strategy if strategy is not None else r.get('prompts')
        return f"{int(r.get('id'))}|{s}|{st}"

    async def _detect(self, todo: list, fouts: dict) -> None:
        # streaming scheduler: keep exactly async_limit requests in flight,
        # refilling each slot the moment its call returns. Each finished row is
        # routed to its strategy's own file (fouts[strategy]).
        pbar = tqdm_async(total=len(todo), desc=self.benchmark)
        sem = asyncio.Semaphore(self.async_limit)

        async def bounded(idx, row, scope, strategy):
            async with sem:
                return await self.__task(idx, row, scope, strategy)

        tasks = [asyncio.create_task(bounded(idx, row, scope, strategy))
                 for row, scope, strategy, idx in todo]
        for completed in asyncio.as_completed(tasks):
            idx, slim = await completed
            fout = fouts[slim["prompts"]]
            fout.write(json.dumps(slim, ensure_ascii=False, default=self._json_default) + "\n")
            fout.flush()
            pbar.update(1)
        pbar.close()

    def _strat_path(self, strategy: str, trial: int) -> Path:
        # results/{label or model}/{strategy}/{benchmark}_{trial}.jsonl
        return self.results_dir / strategy / f"{self.benchmark}_{trial}.jsonl"

    async def _async_run(self, trial: int, reset: bool = False) -> pd.DataFrame:
        # resume: pull already-scored rows from each strategy's own file
        done: dict = {}
        if not reset:
            for s in self.strategies:
                p = self._strat_path(s.name, trial)
                if os.path.exists(p):
                    prev = pd.read_json(p, lines=True)
                    for _, r in prev.iterrows():
                        rd = r.to_dict()
                        if pd.notnull(rd.get('predict')):
                            done[self._row_key(rd)] = rd

        idx = 0
        todo: list = []
        done_rows: list = []
        for _, row in self.dataset_df.iterrows():
            base = row.to_dict()
            for scope in self.__scopes:
                for strategy in self.strategies:
                    idx += 1
                    key = self._row_key(base, scope, strategy.name)
                    if key in done:
                        done_rows.append(done[key])
                    else:
                        todo.append((row, scope, strategy, idx))

        # one open file handle per strategy; rewrite resumed rows then stream new
        fouts: dict = {}
        for s in self.strategies:
            p = self._strat_path(s.name, trial)
            p.parent.mkdir(parents=True, exist_ok=True)
            fouts[s.name] = open(p, "w", encoding="utf-8")
        try:
            for dr in done_rows:
                f = fouts[dr["prompts"]]
                f.write(json.dumps(dr, ensure_ascii=False, default=self._json_default) + "\n")
            for f in fouts.values():
                f.flush()
            if todo:
                await self._detect(todo, fouts)
        finally:
            for f in fouts.values():
                f.close()

        dfs = [pd.read_json(self._strat_path(s.name, trial), lines=True)
               for s in self.strategies
               if os.path.exists(self._strat_path(s.name, trial))]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def convert_to_bool(self, val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            if val == 1 or val == 1.0:
                return True
            elif val == 0 or val == 0.0:
                return False
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ('true', '1'):
                return True
            elif s in ('false', '0'):
                return False
        return None

    def summary(self, results_df: pd.DataFrame, trial: int) -> pd.DataFrame:
        data = []
        table = PrettyTable(self.__rows)
        strat_names = [s.name for s in self.strategies]
        if 'prompts' not in results_df.columns:
            results_df = results_df.assign(prompts=strat_names[0])
        for strategy in strat_names:
            for scope in self.__scopes:
                scope_df = results_df[(results_df['scope'] == scope)
                                      & (results_df['prompts'] == strategy)].copy()
                if scope_df.empty:
                    continue
                scope_df['predict'] = scope_df['predict'].apply(self.convert_to_bool)
                scope_df = scope_df.dropna(subset=['predict', 'label'])
                if scope_df.empty:
                    continue
                y_trues = scope_df['label'].tolist()
                y_preds = scope_df['predict'].tolist()
                f1 = f1_score(y_trues, y_preds, zero_division=0)
                pre = precision_score(y_trues, y_preds, zero_division=0)
                rec = recall_score(y_trues, y_preds, zero_division=0)
                acc = accuracy_score(y_trues, y_preds)
                both = len(set(y_trues)) > 1
                mcc = matthews_corrcoef(y_trues, y_preds) if both else 0.0
                ppr = float(np.mean([bool(p) for p in y_preds]))
                tok = scope_df['tokens'].mean()
                time = scope_df['time(s)'].mean()
                table.add_row([trial, strategy, scope.capitalize(),
                    f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}",
                    f"{mcc:.3f}", f"{ppr:.3f}", f"{tok:.0f}", f"{time:.2f}"])
                data.append({
                    'Trial': trial, 'Strategy': strategy, 'Scope': scope,
                    'Accuracy': acc, 'Precision': pre, 'Recall': rec,
                    'F1-score': f1, 'MCC': mcc, 'Pos.Rate': ppr,
                    'AVG Tokens': tok, 'AVG Time (sec)': time,
                })
        print(table)
        return pd.DataFrame(data)

    async def _run_trials(self, executions: int, reset: bool) -> None:
        overall = []
        for trial in range(1, executions + 1):
            results_df = await self._async_run(trial, reset)
            summary_df = self.summary(results_df, trial)
            overall.append(summary_df)

        overall_df = pd.concat(overall, ignore_index=True)
        overall_df.to_csv(self.results_dir / "result.csv", index=False)

        table = PrettyTable(self.__rows[1:])
        grouped = overall_df.groupby(['Strategy', 'Scope']).agg({
            'Accuracy': ['mean', 'std'], 'Precision': ['mean', 'std'],
            'Recall': ['mean', 'std'], 'F1-score': ['mean', 'std'],
            'MCC': ['mean', 'std'], 'Pos.Rate': ['mean', 'std'],
            'AVG Tokens': ['mean', 'std'], 'AVG Time (sec)': ['mean', 'std']
        })
        for strategy in [s.name for s in self.strategies]:
            for scope in self.__scopes:
                if (strategy, scope) not in grouped.index:
                    continue
                g = grouped.loc[(strategy, scope)]
                table.add_row([
                    strategy, scope.capitalize(),
                    f"{g[('Accuracy', 'mean')]:.3f} ± {g[('Accuracy', 'std')]:.3f}",
                    f"{g[('Precision', 'mean')]:.3f} ± {g[('Precision', 'std')]:.3f}",
                    f"{g[('Recall', 'mean')]:.3f} ± {g[('Recall', 'std')]:.3f}",
                    f"{g[('F1-score', 'mean')]:.3f} ± {g[('F1-score', 'std')]:.3f}",
                    f"{g[('MCC', 'mean')]:.3f} ± {g[('MCC', 'std')]:.3f}",
                    f"{g[('Pos.Rate', 'mean')]:.3f} ± {g[('Pos.Rate', 'std')]:.3f}",
                    f"{g[('AVG Tokens', 'mean')]:.0f} ± {g[('AVG Tokens', 'std')]:.0f}",
                    f"{g[('AVG Time (sec)', 'mean')]:.2f} ± {g[('AVG Time (sec)', 'std')]:.2f}"
                ])
        print(table)

    def run(self, executions: int, reset: bool = False) -> pd.DataFrame:
        asyncio.run(self._run_trials(executions, reset))
