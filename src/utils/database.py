import os
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from .funcNameParser import FuncNameParser

class Database:
    def __init__(self):
        self.BASE_URL = "https://github.com"
        self.BASE_DIR = Path("codeql").resolve()
        self.configure_git_https_rewrite()
    
    def subprocess_run(self, cmd, cwd=None, ignore_error=False):
        r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
        if r.returncode != 0:
            if ignore_error:
                return r.stdout
            details = "[cmd failed] {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                " ".join(cmd),
                r.stdout,
                r.stderr
            )
            raise RuntimeError(details)
        return r.stdout

    def configure_git_https_rewrite(self):
        """
        Ensure git@github.com style URLs fallback to HTTPS so submodules clone without SSH keys.
        """
        self.subprocess_run(
            ["git", "config", "--global", "url.https://github.com/.insteadOf", "git@github.com:"],
            ignore_error=True,
        )
        self.subprocess_run(
            ["git", "config", "--global", "url.https://github.com/.insteadOf", "ssh://git@github.com/"],
            ignore_error=True,
        )
    
    def safe_name(self, name: str) -> str:
        return name.replace("/", "__")
    
    def ensure_repo(self, repo: str, lang: str) -> Path:
        repo_dir = self.BASE_DIR / lang / self.safe_name(repo)
        if not repo_dir.exists():
            repo_url = f"{self.BASE_URL.rstrip('/')}/{repo}.git"
            # print(f"[clone] {repo_url} -> {repo_dir}")
            self.subprocess_run(["git", "clone", "--filter=blob:none", "--no-tags", repo_url, str(repo_dir)])
        # else:
            # print(f"[reuse] {repo_dir}")
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

    def build(self, db_path:str, save_path:str="data/FileFuncRepoVul.csv") -> pd.DataFrame:
        df = pd.read_json(db_path, lines=True)
        
        new_df = []
        for index, row in df.iterrows():
            cve_id = row['cve_id']
            cwe_id = row['cwe_id']
            language = row['cve_language'].lower()
            project = row['project']
            commit_id = row['commit_id']
            parents = row['parents']
            commit_id_before = parents[-1]['commit_id_before']
            details = row['details']
            
            for detail in details:
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
                if "target" not in detail or detail["target"] != 1:
                    continue
                function_before = detail.get("function_before", [])
                function_after = detail.get("function_after", [])
                code_before = detail.get("code_before", "")
                code_after = detail.get("code", "")
                
                for fb, fa in zip(function_before, function_after):
                    if fb["target"] != 1 or fa["target"] == 1:
                        continue
                        
                    vul_row = {
                        'cve_id': cve_id,
                        'cwe_id': tuple(cwe_id),
                        'language': language,
                        'project': project,
                        'commit_id': commit_id_before,
                        'file_name': detail['file_name'],
                        'function': fb["function"],
                        'file': code_before,
                        'repository': None,
                        'vulnerable': True
                    }
                    non_vul_row = {
                        'cve_id': cve_id,
                        'cwe_id': tuple(cwe_id),
                        'language': language,
                        'project': project,
                        'commit_id': commit_id,
                        'file_name': detail['file_name'],
                        'function': fa["function"],
                        'file': code_after,
                        'repository': None,
                        'vulnerable': False
                    }
                    new_df.append(vul_row)
                    new_df.append(non_vul_row)
        new_df = pd.DataFrame(new_df)
        
        new_df = new_df.sort_values(by=['project', 'commit_id']).reset_index(drop=True)
        new_df = new_df.drop_duplicates().reset_index(drop=True)
        
        before_projects = None
        before_commit = None
        
        for index, row in tqdm(new_df.iterrows(), total=len(new_df), desc="Building"):
            project = row["project"]
            language = row["language"]
            if language == "c++": language = "cpp"
            commit_id = row["commit_id"]
            file_name = row["file_name"]
            function = row["function"]
            func_name = FuncNameParser.run(function, language)
            
            try:
                if before_projects != project or before_commit != commit_id:
                    repo_dir = self.ensure_repo(project, language)
                    self.checkout_commit(repo_dir, commit_id)
                    # print(f"[ready] {project}@{commit_id[:12]} -> {repo_dir}")
                    
                    repo = f"codeql/{language}/{self.safe_name(project)}"
                    script_dir = os.path.abspath(os.path.dirname(repo))
                    
                    build_shell = open(os.path.join("codeql", language, "build.sh"), "r").read()
                    build_script = build_shell.format(
                        script_dir=script_dir, repo=repo
                    )
                    self.subprocess_run(["bash", "-lc", build_script])
                    # print("[build] Database created.")
                
                calls_ql = open(os.path.join("codeql", language, "template.ql"), "r").read()
                calls_ql = calls_ql.replace('string targetFileName()      { result = "" }',
                                            "string targetFileName()      { result = \"" + file_name + "\" }")
                calls_ql = calls_ql.replace('string targetFunctionName()  { result = "" }',
                                            "string targetFunctionName()  { result = \"" + func_name + "\" }")
                with open(os.path.join(script_dir, "calls.ql"), "w") as f:
                    f.write(calls_ql)
                
                run_shell = open(os.path.join("codeql", language, "run.sh"), "r").read()
                run_script = run_shell.format(
                    script_dir=script_dir,
                )
                calls = self.subprocess_run(["bash", "-lc", run_script], cwd=script_dir)
                # print(f"[run] {file_name}@{func_name} Query executed.")
                new_df.at[index, "repository"] = calls
            except Exception as e:
                # print(f"[error] {project}@{commit_id[:12]}: {e}")
                # if hasattr(e, 'stderr'):
                #     print(f"[stderr] {e.stderr}")
                pass
            
            before_projects = project
            before_commit = commit_id
        new_df.to_csv(save_path, index=False)
        return new_df
