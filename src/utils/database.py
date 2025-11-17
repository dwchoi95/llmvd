import os
import time
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from .funcNameParser import FuncNameParser

class Database:
    def __init__(self):
        self.BASE_URL = "https://github.com"
        self.BASE_DIR = Path("codeql").resolve()
        self.DEFAULT_ENV = {
            "GIT_LFS_SKIP_SMUDGE": "1",   # 체크아웃/리셋 시 LFS 객체 다운로드 금지
            "GIT_TERMINAL_PROMPT": "0",   # 인증 프롬프트 방지
        }
    
    def subprocess_run(self, cmd, cwd=None, ignore_error=False, env=None, timeout=None):
        merged_env = os.environ.copy()
        merged_env.update(self.DEFAULT_ENV)
        if env:
            merged_env.update(env)
        r = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            env=merged_env,
            timeout=timeout,
        )
        if r.returncode != 0 and not ignore_error:
            details = "[cmd failed] {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                " ".join(cmd),
                r.stdout,
                r.stderr,
            )
            raise RuntimeError(details)
        return r.stdout.strip()

    
    def safe_name(self, name: str) -> str:
        return name.replace("/", "__")
    
    def disable_lfs(self, repo_dir: Path):
        self.subprocess_run(["git", "-C", str(repo_dir), "config", "--local", "filter.lfs.process", ""], ignore_error=True)
        self.subprocess_run(["git", "-C", str(repo_dir), "config", "--local", "filter.lfs.required", "false"], ignore_error=True)
        self.subprocess_run(["git", "-C", str(repo_dir), "config", "--local", "filter.lfs.smudge", "cat"], ignore_error=True)
        self.subprocess_run(["git", "-C", str(repo_dir), "config", "--local", "lfs.fetchexclude", "*"], ignore_error=True)

    def ensure_repo(self, owner_repo: str, language: str) -> Path:
        repo_dir = self.BASE_DIR / language / "repo" / self.safe_name(owner_repo)
        if not repo_dir.exists():
            repo_url = f"{self.BASE_URL.rstrip('/')}/{owner_repo}.git"
            self.subprocess_run(["git", "clone", "--filter=blob:none", "--no-tags", repo_url, str(repo_dir)])
        self.disable_lfs(repo_dir)
        return repo_dir
    
    def ensure_commit_fetched(self, repo_dir: Path, commit_id: str):
        try:
            self.subprocess_run(["git", "-C", str(repo_dir), "rev-parse", "--verify", commit_id])
        except RuntimeError:
            self.subprocess_run(["git", "-C", str(repo_dir), "fetch", "origin", commit_id, "--depth=1"])
            self.subprocess_run(["git", "-C", str(repo_dir), "rev-parse", "--verify", commit_id])
            
    def checkout_commit(self, repo_dir: Path, commit_id: str):
        self.ensure_commit_fetched(repo_dir, commit_id)
        self.subprocess_run(["git", "-C", str(repo_dir), "reset", "--hard"])
        self.subprocess_run(["git", "-C", str(repo_dir), "clean", "-fdx"])
        self.subprocess_run(["git", "-C", str(repo_dir), "checkout", "--detach", commit_id])
        self.subprocess_run(["git", "-C", str(repo_dir), "reset", "--hard", commit_id])
        self.subprocess_run(["git", "-C", str(repo_dir), "clean", "-fdx"])
        self.subprocess_run(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"], ignore_error=True)

    def filter(self, df:pd.DataFrame, save_path:str) -> pd.DataFrame:
        new_df = []
        for index, row in df.iterrows():
            # skip outdated records (이 커밋 이후에도 취약점 수정이 이루어진 경우 스킵)
            outdated = row['outdated']
            if outdated == 1: continue
            
            cve_id = row['cve_id']
            cwe_id = row['cwe_id']
            cvss = row['cvss']
            language = row['cve_language'].lower()
            project = row['project']
            commit_id = row['commit_id']
            parents = row['parents']
            commit_id_before = parents[-1]['commit_id_before']
            details = row['details']
            
            # filter if multiple files are changed in a single commit
            # if len(details) > 1: continue
            
            data = []
            for detail in details:
                # CVE 언어와 파일 언어가 다른 경우 스킵
                if "file_language" not in detail:
                    file_ext = detail["file_language"].lower()
                    if language == "python" :
                        if file_ext != "py":
                            continue
                    elif language == "c++":
                        if file_ext != "cpp":
                            continue
                    elif language == "c":
                        if file_ext != "c":
                            continue
                    elif language == "java":
                        if file_ext != "java":
                            continue
                    continue
                
                # filter only single function changes
                function_before = detail.get("function_before", [])
                vul_functions = []
                for func in function_before:
                    if func["target"] == 1:
                        vul_functions.append(func)
                if len(vul_functions) != 1: continue
                
                fix_functions = []
                function_after = detail.get("function_after", [])
                for func in function_after:
                    if func["target"] == 1:
                        continue
                    has_same_func = False
                    func_after = func["function"]
                    for func_b in function_before:
                        if func_b["function"] == func_after:
                            has_same_func = True
                            break
                    if not has_same_func:
                        fix_functions.append(func)
                if len(fix_functions) != 1: continue
                
                # Add single file with single vulnerable function changes to new_df
                vul_func = vul_functions[0]
                data.append({
                    'cve_id': cve_id,
                    'cwe_id': tuple(cwe_id),
                    'csvvs': cvss,
                    'language': language,
                    'project': project,
                    'commit_id': commit_id_before,
                    'file_name': detail['file_name'],
                    'line': vul_func['line'],
                    'function': vul_func["function"],
                    'file': detail["code_before"],
                    'repository': None,
                    'vulnerable': True
                })
                
                non_vul_func = fix_functions[0]
                data.append({
                    'cve_id': cve_id,
                    'cwe_id': tuple(cwe_id),
                    'csvvs': cvss,
                    'language': language,
                    'project': project,
                    'commit_id': commit_id,
                    'file_name': detail['file_name'],
                    'line': None,
                    'function': non_vul_func["function"],
                    'file': detail["code"],
                    'repository': None,
                    'vulnerable': False
                })
            
            if len(data) == 2:
                new_df.extend(data)
        new_df = pd.DataFrame(new_df)

        # project별로 정렬하고 같은 project, commit_id 묶어서 처리 속도 향상
        new_df = new_df.sort_values(by=['project', 'commit_id']).reset_index(drop=True)
        new_df = new_df.drop_duplicates().reset_index(drop=True)

        # new_df 저장
        new_df.to_json(save_path, orient='records', lines=True)
        return new_df

    def summary(self, df:pd.DataFrame):
        samples = df.shape[0]
        projects = df['project'].nunique().sum()
        cves = df['cve_id'].nunique().sum()
        cwes = set()
        for cwe_ids in df['cwe_id']:
            for cwe_id in cwe_ids:
                cwes.add(cwe_id)
        cwes = len(cwes)
        languages = df['language'].nunique().sum()
        vulnerable = df[df['vulnerable'] == True].shape[0]
        non_vulnerable = df[df['vulnerable'] == False].shape[0]
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Datasets", "Counts"]
        table.add_column("Samples", samples)
        table.add_column("Projects", projects)
        table.add_column("CVEs", cves)
        table.add_column("CWEs", cwes)
        table.add_column("Languages", languages)
        table.add_column("Vulnerable", vulnerable)
        table.add_column("Non-Vulnerable", non_vulnerable)
        print(table)
    
    def build(self, 
              db_path:str="data/ReposVul.jsonl", 
              save_path:str="data/FileFuncRepoVul.jsonl") -> pd.DataFrame:
        df = pd.read_json(db_path, lines=True)
        if os.path.exists(save_path):
            new_df = pd.read_json(save_path, lines=True)
        else:
            new_df = self.filter(df, save_path)
        
        before_projects = None
        before_commit = None
        
        pbar = tqdm(total=len(new_df))
        for index, row in new_df.iterrows():
            project = row["project"]
            language = row["language"]
            if language == "c++": language = "cpp"
            commit_id = row["commit_id"]
            file_name = row["file_name"]
            function = row["function"]
            func_name = FuncNameParser.run(function, language)
            
            pbar.set_description(f"{project}@{index}/{time.strftime('%H:%M')}")
            pbar.update(1)
            
            logs = []
            
            if func_name is None or \
                func_name.strip() == "" or \
                "\n" in func_name or \
                " " in func_name:
                logs.append(f"[skip] Cannot found function name: {project}@{commit_id[:12]} - {file_name}")
                continue
            
            try:
                if language in ["c", "cpp"]: codeql_lang = "c-cpp"
                else: codeql_lang = language
                repo = f"codeql/{codeql_lang}/{project.replace('/', '__')}"
                script_dir = os.path.abspath(os.path.dirname(repo))
                
                if before_projects != project or before_commit != commit_id:
                    repo_dir = self.ensure_repo(project, language)
                    self.checkout_commit(repo_dir, commit_id)
                    logs.append(f"[ready] {project}@{commit_id[:12]} -> {repo_dir}")
                    
                    build_shell = open(os.path.join("codeql", language, "build.sh"), "r").read()
                    build_script = build_shell.format(script_dir=script_dir, repo=repo)
                    subprocess.run(
                        ["bash", "-lc", build_script],
                        text=True, capture_output=True, check=True
                        )
                    logs.append("[build] Database created.")
                
                calls_ql = open(os.path.join("codeql", language, "template.ql"), "r").read()
                calls_ql = calls_ql.replace('string targetFileName()      { result = "" }',
                                            "string targetFileName()      { result = \"" + file_name + "\" }")
                calls_ql = calls_ql.replace('string targetFunctionName()  { result = "" }',
                                            "string targetFunctionName()  { result = \"" + func_name + "\" }")
                with open(os.path.join(script_dir, "calls.ql"), "w") as f:
                    f.write(calls_ql)
                logs.append("[prep] Query prepared.")
                
                run_shell = open(os.path.join("codeql", language, "run.sh"), "r").read()
                run_script = run_shell.format(script_dir=script_dir)
                calls = subprocess.run(
                    ["bash", "-lc", run_script], 
                    text=True, capture_output=True, check=True,
                    cwd=script_dir)
                logs.append("[run] Query executed.")
                new_df.at[index, "repository"] = calls.stdout.strip()
            except Exception as e:
                print("\n".join(logs))
                print(file_name)
                print(func_name)
                print(function)
                print(f"[error] {project}@{commit_id[:12]}: {e}")
                # stderr도 출력하면 디버깅에 도움됨
                if hasattr(e, 'stderr'):
                    print(f"[stderr] {e.stderr}")
                break
                pass
            
            before_projects, before_commit = project, commit_id
            new_df.to_json(save_path, orient='records', lines=True)
        pbar.close()
        return new_df
