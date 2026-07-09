import os
import re
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
    def __init__(
        self,
        llm:str,
        temperature:float,
        dataset_path:str,
        save_dir:str="results",
        async_limit:int=100,
        strategies:list[str]|None=None,
        system_prompt_file:str="src/prompts/detection/system.md",
        func_prompt_file:str="src/prompts/detection/function.md",
        file_prompt_file:str="src/prompts/detection/file.md",
        repo_prompt_file:str="src/prompts/detection/repository.md"
    ):
        self.benchmark = Path(dataset_path).stem
        self.results_dir = Path(save_dir) / llm
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.save_file = self.results_dir / f"{self.benchmark}"

        self.__scopes = ["function", "file", "repository"]
        # prompting strategies are the second experimental axis (scope x strategy)
        self.strategies = [get_strategy(s) for s in (strategies or DEFAULT_STRATEGIES)]
        self.__rows = ["Trial", "Strategy", "Scope", "Accuracy", "Precision",
                       "Recall", "F1-score", "MCC", "Pos.Rate",
                       "AVG Tokens", "AVG Time (sec)"]
        self.model = self._select_model(llm, temperature)
        # only the local (open-weight) client exposes token logprobs -> scores
        self.__scored = hasattr(self.model, "run_scored")
        self.pm = PromptManager()
        # legacy single system prompt (used only by the label-only fallback path)
        self.system = self.pm.render(file=system_prompt_file)
        # per-strategy system prompts, rendered once
        self.strategy_systems = {
            s.name: self.pm.render(file=s.system_file) for s in self.strategies
        }
        self.func_prompt_file = func_prompt_file
        self.file_prompt_file = file_prompt_file
        self.repo_prompt_file = repo_prompt_file
        self.dataset_df = self._load_data(dataset_path)
        self.columns = self.dataset_df.columns.tolist()
        self.evaluator = Evaluator(llm)

        self.async_limit = async_limit

    def _select_model(self, llm:str, temperature:float):
        if llm.startswith("gpt"):
            return GPT(llm, temperature)
        elif llm.startswith("claude"):
            return CLAUDE(llm, temperature)
        elif llm.startswith("gemini"):
            return GEMINI(llm, temperature)
        else:
            return OLLAMA(llm, temperature)
        raise ValueError(f"Unsupported model: {llm}")

    def _load_data(self, dataset_path: str) -> pd.DataFrame:
        return pd.read_json(dataset_path, lines=True)

    def _code_bleu(self, function:str, call:str, language:str) -> float:
        language = language.lower()
        if language == "c++": language = "cpp"
        logging.disable(logging.WARNING)
        try:
            return calc_codebleu(
                [function],
                [call],
                lang=language
            )["codebleu"]
        except Exception as e:
            pass
        finally:
            logging.disable(logging.NOTSET)
        return 0.0

    def priority(self, function:str, calls:list[dict], language:str) -> list[str]:
        scored_calls = []
        for call in calls:
            score = self._code_bleu(function, call['function'], language)
            scored_calls.append((call, score))
        scored_calls.sort(key=lambda x: x[1], reverse=True)
        return [call for call, score in scored_calls]

    # remove any trailing "# Output Format" block from a scope template so the
    # strategy's system prompt is the sole authority on the answer format
    _OUTFMT_RE = re.compile(r"\n#+\s*Output Format.*\Z", re.DOTALL | re.IGNORECASE)

    def _strip_output_format(self, text:str) -> str:
        return self._OUTFMT_RE.sub("", text).rstrip() + "\n"

    def _make_user_prompt(self, row:pd.Series, scope:str) -> str:
        function = row["function"]
        language = row["language"]

        if scope == "function":
            user = self.pm.render(
                file=self.func_prompt_file,
                language=language,
                function=function,
            )
        elif scope == "file":
            user = self.pm.render(
                file=self.file_prompt_file,
                language=language,
                function=function,
                file_code=row["file"],
            )
        elif scope == "repository":
            repository = row.get("repository", {"callee": [], "caller": []})
            callees = self.priority(function, repository['callee'], language)
            callers = self.priority(function, repository['caller'], language)
            repo_info = ""
            if callees:
                repo_info += "### Functions called by the Target Function (Callees):\n"
                for callee in callees:
                    repo_info += f"#### File: {callee['file']}\n```{language}\n{callee['function']}\n```\n"
            if callers:
                repo_info += "### Functions that call the Target Function (Callers):\n"
                for caller in callers:
                    repo_info += f"#### File: {caller['file']}\n```{language}\n{caller['function']}\n```\n"
            user = self.pm.render(
                file=self.repo_prompt_file,
                language=language,
                function=function,
                repository=repo_info,
            )
        return self._strip_output_format(user)

    async def __task(self, idx:int, row:pd.Series, scope:str, strategy):
        row_dict = row.to_dict()
        system = self.strategy_systems[strategy.name]
        user = self._make_user_prompt(row, scope)
        start = perf_counter()
        if self.__scored:
            label, score, raw = await self.model.run_scored(
                system, user, max_tokens=strategy.max_tokens,
                reasoning=strategy.reasoning)
        else:
            label = await self.model.run(system, user)
            score, raw = None, None
        duration = perf_counter() - start
        prompt = f"{system}\n\n{user}"
        return idx, row_dict, scope, strategy.name, label, score, raw, prompt, duration

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

    def _row_key(self, r:dict, scope:str=None, strategy:str=None) -> str:
        s = scope if scope is not None else r.get('scope')
        st = strategy if strategy is not None else r.get('strategy')
        return f"{r.get('group_id')}|{r.get('file_name')}|{r.get('vulnerable')}|{s}|{st}"

    async def __run(self, batch:list, pbar:tqdm_async, fout) -> None:
        for completed in asyncio.as_completed(batch):
            idx, row_dict, scope, strategy, label, score, raw, prompt, duration = await completed
            row_dict['index'] = idx
            row_dict["predict"] = label
            row_dict["score"] = score
            row_dict["raw"] = raw
            row_dict["scope"] = scope
            row_dict["strategy"] = strategy
            row_dict["prompt"] = prompt
            row_dict["tokens"] = self.evaluator.token_count(prompt)
            row_dict["Time (sec)"] = duration
            # incremental save: persist each result the moment it completes
            fout.write(json.dumps(row_dict, ensure_ascii=False, default=self._json_default) + "\n")
            fout.flush()
            pbar.update(1)

    async def _detect(self, todo:list, fout) -> None:
        pbar = tqdm_async(total=len(todo), desc=self.benchmark)
        batch:list[asyncio.Task] = []
        for row, scope, strategy, idx in todo:
            batch.append(asyncio.create_task(self.__task(idx, row, scope, strategy)))
            if len(batch) >= self.async_limit:
                await self.__run(batch, pbar, fout)
                batch.clear()
        if batch:
            await self.__run(batch, pbar, fout)
        pbar.close()

    async def _async_run(self, save_path:str, reset:bool=False) -> pd.DataFrame:
        # already-completed (non-null predict) results, keyed by (sample, scope, strategy)
        done:dict = {}
        if not reset and os.path.exists(save_path):
            prev = pd.read_json(save_path, lines=True)
            for _, r in prev.iterrows():
                rd = r.to_dict()
                if pd.notnull(rd.get('predict')):
                    done[self._row_key(rd)] = rd

        # split every (row, scope, strategy) into done (keep) vs todo (request)
        idx = 0
        todo:list = []
        done_rows:list = []
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

        # rewrite file: keep done rows, then stream new results row-by-row
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fout:
            for dr in done_rows:
                fout.write(json.dumps(dr, ensure_ascii=False, default=self._json_default) + "\n")
            fout.flush()
            if todo:
                await self._detect(todo, fout)
        return pd.read_json(save_path, lines=True)

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

    def summary(self, results_df:pd.DataFrame, trial:int) -> pd.DataFrame:
        data = []
        table = PrettyTable(self.__rows)
        strat_names = [s.name for s in self.strategies]
        # tolerate legacy files without a strategy column
        if 'strategy' not in results_df.columns:
            results_df = results_df.assign(strategy=strat_names[0])
        for strategy in strat_names:
            for scope in self.__scopes:
                scope_df = results_df[(results_df['scope'] == scope)
                                      & (results_df['strategy'] == strategy)].copy()
                if scope_df.empty:
                    continue
                scope_df['predict'] = scope_df['predict'].apply(self.convert_to_bool)
                scope_df = scope_df.dropna(subset=['predict', 'vulnerable'])
                if scope_df.empty:
                    continue

                y_trues = scope_df['vulnerable'].tolist()
                y_preds = scope_df['predict'].tolist()

                f1 = f1_score(y_trues, y_preds, zero_division=0)
                pre = precision_score(y_trues, y_preds, zero_division=0)
                rec = recall_score(y_trues, y_preds, zero_division=0)
                acc = accuracy_score(y_trues, y_preds)
                both = len(set(y_trues)) > 1
                mcc = matthews_corrcoef(y_trues, y_preds) if both else 0.0
                ppr = float(np.mean([bool(p) for p in y_preds]))
                tok = scope_df['tokens'].mean()
                time = scope_df['Time (sec)'].mean()
                table.add_row([trial, strategy, scope.capitalize(),
                    f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}",
                    f"{mcc:.3f}", f"{ppr:.3f}",
                    f"{tok:.0f}", f"{time:.2f}"])
                data.append({
                    'Trial': trial, 'Strategy': strategy, 'Scope': scope,
                    'Accuracy': acc, 'Precision': pre, 'Recall': rec,
                    'F1-score': f1, 'MCC': mcc, 'Pos.Rate': ppr,
                    'AVG Tokens': tok, 'AVG Time (sec)': time,
                })

        print(table)
        return pd.DataFrame(data)

    async def _run_trials(self, executions:int, reset:bool) -> None:
        overall = []
        for trial in range(1, executions+1):
            save_path = str(self.save_file) + f"_{trial}.jsonl" \
            if executions > 1 else str(self.save_file) + ".jsonl"
            results_df = await self._async_run(save_path, reset)
            summary_df = self.summary(results_df, trial)
            overall.append(summary_df)

        overall_df = pd.concat(overall, ignore_index=True)
        overall_save_path = self.results_dir / "result.csv"
        overall_df.to_csv(overall_save_path, index=False)

        table = PrettyTable(self.__rows[1:])
        grouped = overall_df.groupby(['Strategy', 'Scope']).agg({
            'Accuracy': ['mean', 'std'],
            'Precision': ['mean', 'std'],
            'Recall': ['mean', 'std'],
            'F1-score': ['mean', 'std'],
            'MCC': ['mean', 'std'],
            'Pos.Rate': ['mean', 'std'],
            'AVG Tokens': ['mean', 'std'],
            'AVG Time (sec)': ['mean', 'std']
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

    def run(self, executions:int, reset:bool=False) -> pd.DataFrame:
        asyncio.run(self._run_trials(executions, reset))
