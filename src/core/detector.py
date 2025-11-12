import os
import ast
import asyncio
from pathlib import Path
from time import perf_counter
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async

from .evaluation import Evaluation
from ..llms import GPT, CLAUDE, GEMINI, LLAMA
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
        results_dir = Path(save_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.save_file = results_dir / f"{self.benchmark}_{llm}"
        
        self.__scopes = ["function", "file", "repository"]
        self.model = self._select_model(llm, temperature)
        self.pm = PromptManager()
        self.system = self.pm.render(file=system_prompt_file)
        self.func_prompt_file = func_prompt_file
        self.file_prompt_file = file_prompt_file
        self.repo_prompt_file = repo_prompt_file
        self.dataset_df = self._load_data(dataset_path)
        self.columns = self.dataset_df.columns.tolist()
        self.evaluation = Evaluation()
        
        self.async_limit = async_limit

    def _select_model(self, llm:str, temperature:float):
        if llm.startswith("gpt"):
            return GPT(llm, temperature)
        if llm.startswith("claude"):
            return CLAUDE(llm, temperature)        
        if llm.startswith("gemini"):
            return GEMINI(llm, temperature)
        if llm.startswith("llama3:8b"):
            return LLAMA(llm, temperature)
        raise ValueError(f"Unsupported model: {llm}")

    def _load_data(self, dataset_path: str) -> pd.DataFrame:
        return pd.read_json(dataset_path, lines=True)
    
    def _safe_eval(self, data:str) -> dict:
        try: data = ast.literal_eval(data)
        except: data = {"callee": [], "caller": []}
        return data
            
    def _make_user_prompt(self, row:pd.Series, scope:str) -> str:
        if scope == "function":
            user = self.pm.render(
                file=self.func_prompt_file,
                language=row["language"],
                function=row["function"],
            ) 
        elif scope == "file":
            user = self.pm.render(
                file=self.file_prompt_file,
                language=row["language"],
                function=row["function"],
                file_code=row["file"],
            )
        elif scope == "repository":
            repository = row.get("repository", {"callee": [], "caller": []})
            calls = self._safe_eval(repository)
            callees = calls['callee']
            callers = calls['caller']
            repo_info = ""
            if callees:
                repo_info += "### Callee Functions:\n"
                for callee in callees:
                    repo_info += f"#### File: {callee['file']}\n```{row['language']}\n{callee['function']}\n```\n"
            if callers:
                repo_info += "### Caller Functions:\n"
                for caller in callers:
                    repo_info += f"#### File: {caller['file']}\n```{row['language']}\n{caller['function']}\n```\n"
            user = self.pm.render(
                file=self.repo_prompt_file,
                language=row["language"],
                function=row["function"],
                repository=repo_info,
            )
        return user
        
    async def __task(self, row:pd.Series, scope:str): 
        row_dict = row.to_dict()
        user = self._make_user_prompt(row, scope)
        start = perf_counter()
        is_vulnerable = await self.model.run(self.system, user)
        duration = perf_counter() - start
        prompt = f"{self.system}\n\n{user}"
        return row_dict, scope, is_vulnerable, prompt, duration

    async def __run(self, 
        batch:list[asyncio.Task], 
        results:list[dict],
        pbar:tqdm_async
    ) -> None:
        for completed in asyncio.as_completed(batch):
            row_dict, scope, is_vulnerable, prompt, duration = await completed
            row_dict["predict"] = is_vulnerable
            row_dict["scope"] = scope
            row_dict["prompt"] = prompt
            row_dict["tokens"] = self.evaluation.token_count(prompt)
            row_dict["Time (sec)"] = duration
            results.append(row_dict)
            pbar.update(1)

    async def _detect(self) -> list:
        results = []
        pbar = tqdm_async(total=len(self.dataset_df), desc=self.benchmark)
        batch:list[asyncio.Task] = []
        for _, row in self.dataset_df.iterrows():
            for scope in self.__scopes:
                batch.append(asyncio.create_task(self.__task(row, scope)))
            if len(batch) >= self.async_limit:
                await self.__run(batch, results, pbar)
                batch.clear()
        if batch:
            await self.__run(batch, results, pbar)
        pbar.close()
        return results
      
    async def _async_run(self, save_path:str, reset:bool=False) -> pd.DataFrame:
        results_df = pd.DataFrame()
        if not reset and os.path.exists(save_path):
            results_df = pd.read_csv(save_path)
        
        if results_df.empty:
            # Detection
            results = await self._detect()
            # Save Repair Results
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
        return results_df

    def run(self, executions:int, reset:bool=False) -> pd.DataFrame:
        for trial in range(1, executions+1):
            save_path = str(self.save_file) + f"_{trial}.csv" if executions > 1 else str(self.save_file) + ".csv"
            asyncio.run(self._async_run(save_path, reset))
