import os
import logging
import asyncio
from pathlib import Path
from time import perf_counter
import pandas as pd
from codebleu import calc_codebleu
from tqdm.asyncio import tqdm as tqdm_async
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
from prettytable import PrettyTable

from .evaluator import Evaluator
from ..llms import GPT, CLAUDE, GEMINI, OLLAMA
from ..prompts import PromptManager


class Detector:
    def __init__(
        self, 
        llm:str, 
        temperature:float,
        dataset_path:str, 
        save_dir:str="results", 
        async_limit:int=100,
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
        self.__rows = ["Trial", "Scope", "Accuracy", "Precision", "Recall", "F1-score", "AVG Tokens", "AVG Time (sec)"]
        self.model = self._select_model(llm, temperature)
        self.pm = PromptManager()
        self.system = self.pm.render(file=system_prompt_file)
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
        return user
        
    async def __task(self, idx:int, row:pd.Series, scope:str): 
        row_dict = row.to_dict()
        user = self._make_user_prompt(row, scope)
        start = perf_counter()
        is_vulnerable = await self.model.run(self.system, user)
        duration = perf_counter() - start
        prompt = f"{self.system}\n\n{user}"
        return idx, row_dict, scope, is_vulnerable, prompt, duration

    async def __run(self, 
        batch:list[asyncio.Task], 
        results:list[dict],
        pbar:tqdm_async
    ) -> None:
        for completed in asyncio.as_completed(batch):
            idx, row_dict, scope, is_vulnerable, prompt, duration = await completed
            row_dict['index'] = idx
            row_dict["predict"] = is_vulnerable
            row_dict["scope"] = scope
            row_dict["prompt"] = prompt
            row_dict["tokens"] = self.evaluator.token_count(prompt)
            row_dict["Time (sec)"] = duration
            results.append(row_dict)
            pbar.update(1)

    async def _detect(self, total_task:int, dataset_df:pd.DataFrame) -> list:
        results = []
        idx = 0
        pbar = tqdm_async(total=total_task, desc=self.benchmark)
        batch:list[asyncio.Task] = []
        for _, row in dataset_df.iterrows():
            if 'index' in row:
                idx = int(row['index'])
                if pd.notnull(row['predict']) or pd.notna(row['predict']):
                    results.append(row.to_dict())
                    continue
                scope = row['scope']
                batch.append(asyncio.create_task(self.__task(idx, row, scope)))
            else:
                for scope in self.__scopes:
                    idx += 1
                    batch.append(asyncio.create_task(self.__task(idx, row, scope)))
            if len(batch) >= self.async_limit:
                await self.__run(batch, results, pbar)
                batch.clear()
        if batch:
            await self.__run(batch, results, pbar)
        pbar.close()
        return results
      
    async def _async_run(self, save_path:str, reset:bool=False) -> pd.DataFrame:
        results_df = self.dataset_df.copy()
        total_tasks = len(results_df) * len(self.__scopes)
        if not reset and os.path.exists(save_path):
            results_df = pd.read_json(save_path, lines=True)
            nan_rows = results_df[results_df['predict'].isna()]
            total_tasks = len(nan_rows)
            
        if total_tasks > 0:
            # Detection
            results = await self._detect(total_tasks, results_df)
            # Save Repair Results
            results_df = pd.DataFrame(results)
            results_df.to_json(save_path, orient='records', lines=True)
        return results_df
    
    def convert_to_bool(self, val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            if val == 1 or val == 1.0:
                return True
            elif val == 0 or val == 0.0:
                return False
        if isinstance(val, str):
            if val.lower() == 'true':
                return True
            elif val.lower() == 'false':
                return False
        return None

    def summary(self, results_df:pd.DataFrame, trial:int) -> pd.DataFrame:
        data = []
        table = PrettyTable(self.__rows)
        for scope in self.__scopes:
            scope_df = results_df[results_df['scope'] == scope]
            scope_df = scope_df.copy()
            scope_df['predict'] = scope_df['predict'].apply(self.convert_to_bool)
            scope_df = scope_df.dropna(subset=['predict', 'vulnerable'])

            y_trues = scope_df['vulnerable'].tolist()
            y_preds = scope_df['predict'].tolist()
            
            f1 = f1_score(y_trues, y_preds, zero_division=0)
            pre = precision_score(y_trues, y_preds, zero_division=0)
            rec = recall_score(y_trues, y_preds, zero_division=0)
            acc = accuracy_score(y_trues, y_preds)
            tok = scope_df['tokens'].mean()
            time = scope_df['Time (sec)'].mean()
            table.add_row([trial, scope.capitalize(), 
                f"{acc:.3f}", f"{pre:.3f}", f"{rec:.3f}", f"{f1:.3f}", 
                f"{tok:.0f}", f"{time:.2f}"])
            data.append({
                'Trial': trial,
                'Scope': scope,
                'Accuracy': acc,
                'Precision': pre,
                'Recall': rec,
                'F1-score': f1,
                'AVG Tokens': tok,
                'AVG Time (sec)': time
            })
            
        print(table)
        df = pd.DataFrame(data)
        return df
    
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
        grouped = overall_df.groupby('Scope').agg({
            'Accuracy': ['mean', 'std'],
            'Precision': ['mean', 'std'],
            'Recall': ['mean', 'std'],
            'F1-score': ['mean', 'std'],
            'AVG Tokens': ['mean', 'std'],
            'AVG Time (sec)': ['mean', 'std']
        })
        for scope in self.__scopes:
            table.add_row([
                scope.capitalize(),
                f"{grouped.loc[scope, ('Accuracy', 'mean')]:.3f} ± {grouped.loc[scope, ('Accuracy', 'std')]:.3f}",
                f"{grouped.loc[scope, ('Precision', 'mean')]:.3f} ± {grouped.loc[scope, ('Precision', 'std')]:.3f}",
                f"{grouped.loc[scope, ('Recall', 'mean')]:.3f} ± {grouped.loc[scope, ('Recall', 'std')]:.3f}",
                f"{grouped.loc[scope, ('F1-score', 'mean')]:.3f} ± {grouped.loc[scope, ('F1-score', 'std')]:.3f}",
                f"{grouped.loc[scope, ('AVG Tokens', 'mean')]:.0f} ± {grouped.loc[scope, ('AVG Tokens', 'std')]:.0f}",
                f"{grouped.loc[scope, ('AVG Time (sec)', 'mean')]:.2f} ± {grouped.loc[scope, ('AVG Time (sec)', 'std')]:.2f}"
            ])
            
        print(table)
            
    def run(self, executions:int, reset:bool=False) -> pd.DataFrame:
        asyncio.run(self._run_trials(executions, reset))
        