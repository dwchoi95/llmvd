import time
import os
import pandas as pd
import subprocess
from pathlib import Path
from src.utils import FuncNameParser
from tqdm import tqdm

# ====== 설정 ======
BASE_URL = "https://github.com"
BASE_DIR = Path("codeql").resolve()
DEFAULT_ENV = {
    "GIT_LFS_SKIP_SMUDGE": "1",   # 체크아웃/리셋 시 LFS 객체 다운로드 금지
    "GIT_TERMINAL_PROMPT": "0",   # 인증 프롬프트 방지
}
# ==================

def run(cmd, cwd=None, ignore_error=False, env=None, timeout=None):
    merged_env = os.environ.copy()
    merged_env.update(DEFAULT_ENV)
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

def safe_name(owner_repo: str) -> str:
    return owner_repo.replace("/", "__")

def disable_lfs(repo_dir: Path):
    """
    (최대한 안전하게) 현재 저장소에서 LFS 필터 자체를 비활성화합니다.
    git-lfs 미설치/오류 상황에서도 체크아웃이 계속되도록 합니다.
    """
    # filter.lfs.process를 비워서 외부 필터 호출 자체를 막음
    run(["git", "-C", str(repo_dir), "config", "--local", "filter.lfs.process", ""], ignore_error=True)
    # 필수 아님으로 지정
    run(["git", "-C", str(repo_dir), "config", "--local", "filter.lfs.required", "false"], ignore_error=True)
    # smudge 단계 대체(다운로드 대신 통과); 일부 git 버전에서 필요
    run(["git", "-C", str(repo_dir), "config", "--local", "filter.lfs.smudge", "cat"], ignore_error=True)
    # LFS fetch도 전부 제외
    run(["git", "-C", str(repo_dir), "config", "--local", "lfs.fetchexclude", "*"], ignore_error=True)

def ensure_repo(owner_repo: str, language: str) -> Path:
    """
    프로젝트 작업 디렉토리가 없으면 최초 1회만 클론합니다.
    이후에는 같은 디렉토리를 재사용합니다.
    """
    repo_dir = BASE_DIR / language / safe_name(owner_repo)
    if not repo_dir.exists():
        repo_url = f"{BASE_URL.rstrip('/')}/{owner_repo}.git"
        # print(f"[clone] {repo_url} -> {repo_dir}")
        # 부분 클론으로 최초 트래픽/공간 최소화 (필요한 blob은 체크아웃 시점에 on-demand로 받음)
        run(["git", "clone", "--filter=blob:none", "--no-tags", repo_url, str(repo_dir)])
        # 기본 브랜치를 알 수 없으니 일단 패치 대상 커밋별로 가져와서 체크아웃할 예정
    # else:
    #     print(f"[reuse] {repo_dir}")
    disable_lfs(repo_dir)
    return repo_dir

def checkout_commit(repo_dir: Path, commit_id: str):
    """
    작업 디렉토리를 해당 커밋 상태로 깔끔하게 맞춥니다.
    (detached HEAD + 하드 리셋 + 쓰레기 파일 제거)
    """
    disable_lfs(repo_dir)
    ensure_commit_fetched(repo_dir, commit_id)

    # 워킹트리 변경사항/빌드 산출물 제거
    run(["git", "-C", str(repo_dir), "reset", "--hard"])
    run(["git", "-C", str(repo_dir), "clean", "-fdx"])

    # 커밋으로 이동(detached HEAD)
    run(["git", "-C", str(repo_dir), "checkout", "--force", "--detach", commit_id])
    run(["git", "-C", str(repo_dir), "reset", "--hard", commit_id])
    run(["git", "-C", str(repo_dir), "clean", "-fdx"])

    # 서브모듈 쓰는 레포 대비
    run(["git", "-C", str(repo_dir), "submodule", "update", "--init", "--recursive"], ignore_error=True)

def ensure_commit_fetched(repo_dir: Path, commit_id: str):
    """
    해당 커밋이 로컬에 없으면 최소한의 깊이로 정확히 그 커밋만 fetch 합니다.
    """
    try:
        run(["git", "-C", str(repo_dir), "rev-parse", "--verify", commit_id])
    except RuntimeError:
        # 지정 커밋만 얕게 가져와서 히스토리 팽창을 막습니다.
        run(["git", "-C", str(repo_dir), "fetch", "origin", commit_id, "--depth=1"])
        # 검증 재시도
        run(["git", "-C", str(repo_dir), "rev-parse", "--verify", commit_id])


# ================= 사용 예시 =================
save_path = "data/FuncFileRepo2.jsonl"
before_projects = None
before_commit = None

if os.path.exists(save_path):
    new_df = pd.read_json(save_path, lines=True)

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
    
    if project in ["mnbikeways/database"]:
        continue
    
    logs = []
    
    if func_name is None or \
        func_name.strip() == "" or \
        "\n" in func_name or \
        " " in func_name:
        logs.append(f"[skip] 함수 이름을 추출할 수 없습니다: {project}@{commit_id[:12]} - {file_name}")
        continue
    
    if row["repository"] is not None and row["repository"] != "{'callee': [], 'caller': []}":
        continue  # 이미 처리된 항목은 건너뜀
    try:
        repo = f"codeql/{language}/{project.replace('/', '__')}"
        script_dir = os.path.abspath(os.path.dirname(repo))
        
        # if project == "openbsd/src" and commit_id == "f748277ed1fc7065ae8998d61ed78b9ab1e55fae":
        #     print("debug")
        #     pass
        # else:
        #     continue
        
        if before_projects != project or before_commit != commit_id:
            repo_dir = ensure_repo(project, language)
            checkout_commit(repo_dir, commit_id)
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
        # pass
    
    before_projects, before_commit = project, commit_id
    new_df.to_json(save_path, orient='records', lines=True)

pbar.close()