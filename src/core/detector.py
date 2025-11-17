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
        results_dir = Path(save_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.save_file = results_dir / llm / f"{self.benchmark}"
        
        self.__scopes = ["function", "file", "repository"]
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
                repo_info += "### Callee Functions:\n"
                for callee in callees:
                    repo_info += f"#### File: {callee['file']}\n```{language}\n{callee['function']}\n```\n"
            if callers:
                repo_info += "### Caller Functions:\n"
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
    
    def summary(self, executions):
        """n개의 실행 결과를 평가"""
        
        # 각 실행별 결과를 저장할 리스트
        all_metrics = []
        all_tokens = []
        all_times = []
        
        # 전체 데이터를 모을 리스트
        all_results = []
        
        for trial in range(executions):
            save_path = str(self.save_file) + f"_{trial+1}.jsonl" \
                if executions > 1 else str(self.save_file) + ".jsonl"
            results_df = pd.read_json(save_path, lines=True)
            
            # 1. 분류 성능 지표 계산
            y_true = results_df['vulnerable'].values
            y_pred = results_df['predict'].values
            
            # NaN/None 값 확인 및 처리
            # pd.isna()는 None과 NaN 모두 처리
            valid_mask = ~pd.isna(y_pred)
            n_total = len(y_pred)
            n_valid = valid_mask.sum()
            n_invalid = n_total - n_valid
            
            # 유효한 예측만 필터링
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            if n_valid > 0:
                metrics = {
                    'trial': trial + 1,
                    'accuracy': accuracy_score(y_true_valid, y_pred_valid),
                    'precision': precision_score(y_true_valid, y_pred_valid, zero_division=0),
                    'recall': recall_score(y_true_valid, y_pred_valid, zero_division=0),
                    'f1_score': f1_score(y_true_valid, y_pred_valid, zero_division=0),
                    'n_total': n_total,
                    'n_valid': n_valid,
                    'n_invalid': n_invalid,
                    'valid_rate': n_valid / n_total if n_total > 0 else 0
                }
            else:
                # 모든 예측이 실패한 경우
                metrics = {
                    'trial': trial + 1,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'n_total': n_total,
                    'n_valid': 0,
                    'n_invalid': n_invalid,
                    'valid_rate': 0.0
                }
            
            all_metrics.append(metrics)
            
            # 2. Tokens 통계
            all_tokens.extend(results_df['tokens'].values)
            
            # 3. Time 통계
            all_times.extend(results_df['Time (sec)'].values)
            
            # 전체 데이터 저장
            all_results.append(results_df)
            
            print(f"\n{'='*50}")
            print(f"Trial {trial + 1} Results:")
            print(f"{'='*50}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        # 종합 통계 계산
        metrics_df = pd.DataFrame(all_metrics)
        
        print(f"\n{'='*50}")
        print(f"SUMMARY ACROSS {executions} TRIALS")
        print(f"{'='*50}")
        
        # 성능 지표 요약
        print("\n[Classification Metrics]")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            print(f"{metric.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Tokens 통계
        print("\n[Token Usage]")
        print(f"Mean:   {np.mean(all_tokens):.2f} ± {np.std(all_tokens):.2f}")
        print(f"Median: {np.median(all_tokens):.2f}")
        print(f"Min:    {np.min(all_tokens)}")
        print(f"Max:    {np.max(all_tokens)}")
        
        # Time 통계
        print("\n[Execution Time (sec)]")
        print(f"Mean:   {np.mean(all_times):.2f} ± {np.std(all_times):.2f}")
        print(f"Median: {np.median(all_times):.2f}")
        print(f"Min:    {np.min(all_times):.2f}")
        print(f"Max:    {np.max(all_times):.2f}")
        print(f"Total:  {np.sum(all_times):.2f}")
        
        # Optional: 전체 데이터 종합 분석
        if executions > 1:
            combined_df = pd.concat(all_results, ignore_index=True)
            y_true_combined = combined_df['vulnerable'].values
            y_pred_combined = combined_df['predict'].values
            
            print(f"\n[Overall Performance (All Trials Combined)]")
            print(f"Total samples: {len(combined_df)}")
            print(f"Accuracy:  {accuracy_score(y_true_combined, y_pred_combined):.4f}")
            print(f"Precision: {precision_score(y_true_combined, y_pred_combined, zero_division=0):.4f}")
            print(f"Recall:    {recall_score(y_true_combined, y_pred_combined, zero_division=0):.4f}")
            print(f"F1-Score:  {f1_score(y_true_combined, y_pred_combined, zero_division=0):.4f}")
            
            # # 클래스 분포 확인
            # print(f"\n[Class Distribution]")
            # print(f"True - Vulnerable: {y_true_combined.sum()} ({y_true_combined.sum()/len(y_true_combined)*100:.1f}%)")
            # print(f"True - Not Vulnerable: {(~y_true_combined).sum()} ({(~y_true_combined).sum()/len(y_true_combined)*100:.1f}%)")
            # print(f"Pred - Vulnerable: {y_pred_combined.sum()} ({y_pred_combined.sum()/len(y_pred_combined)*100:.1f}%)")
            # print(f"Pred - Not Vulnerable: {(~y_pred_combined).sum()} ({(~y_pred_combined).sum()/len(y_pred_combined)*100:.1f}%)")
            
            # # Confusion Matrix
            # print("\n[Confusion Matrix]")
            # cm = confusion_matrix(y_true_combined, y_pred_combined)
            # print("                  Predicted")
            # print("                  Not Vuln  Vulnerable")
            # print(f"Actual Not Vuln   {cm[0, 0]:8d}  {cm[0, 1]:10d}")
            # print(f"Actual Vulnerable {cm[1, 0]:8d}  {cm[1, 1]:10d}")
            
            # # Classification Report (수정된 부분)
            # print("\n[Classification Report]")
            # # 실제 데이터에 존재하는 클래스 확인
            # unique_classes = np.unique(np.concatenate([y_true_combined, y_pred_combined]))
            
            # if len(unique_classes) == 2:
            #     # 두 클래스가 모두 존재할 때
            #     print(classification_report(
            #         y_true_combined, 
            #         y_pred_combined, 
            #         labels=[False, True],
            #         target_names=['Not Vulnerable', 'Vulnerable'],
            #         zero_division=0
            #     ))
            # else:
            #     # 한 클래스만 존재할 때
            #     print(f"Warning: Only one class present in data: {unique_classes}")
            #     print(classification_report(
            #         y_true_combined, 
            #         y_pred_combined,
            #         zero_division=0
            #     ))
        
        # 결과를 DataFrame으로 반환
        summary_dict = {
            'metrics_per_trial': metrics_df,
            'token_stats': {
                'mean': np.mean(all_tokens),
                'std': np.std(all_tokens),
                'median': np.median(all_tokens),
                'min': np.min(all_tokens),
                'max': np.max(all_tokens)
            },
            'time_stats': {
                'mean': np.mean(all_times),
                'std': np.std(all_times),
                'median': np.median(all_times),
                'min': np.min(all_times),
                'max': np.max(all_times),
                'total': np.sum(all_times)
            }
        }
        
        return summary_dict
                    
    async def _run_trials(self, executions:int, reset:bool) -> None:
        for trial in range(1, executions+1):
            save_path = str(self.save_file) + f"_{trial}.jsonl" \
            if executions > 1 else str(self.save_file) + ".jsonl"
            await self._async_run(save_path, reset)
            
    def run(self, executions:int, reset:bool=False) -> pd.DataFrame:
        asyncio.run(self._run_trials(executions, reset))
        
        # summary_dict = self.summary(executions)
        # print(summary_dict)