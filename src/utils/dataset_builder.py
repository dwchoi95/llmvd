import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd


def _is_valid_detail_language(detail: Dict[str, Any], lang: str) -> bool:
    ext = (detail.get("file_language") or "").lower()
    if lang == "python":
        return ext == "py"
    if lang == "c++":
        return ext == "cpp"
    if lang == "c":
        return ext == "c"
    if lang == "java":
        return ext == "java"
    return False


def _collect_functions(detail: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    fs = []
    for f in detail.get(key, []) or []:
        func_code = f.get("function")
        line = f.get("line")
        target = f.get("target", 0)
        if func_code and isinstance(func_code, str):
            fs.append({"function": func_code, "line": line, "target": target})
    return fs


def _choose_primary(functions: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    # 1) 우선 target=1 중 첫 번째를 선택, 2) 없으면 가장 긴 함수 선택
    for i, f in enumerate(functions):
        if f.get("target", 0) == 1:
            return i, f
    if not functions:
        return -1, {}
    lens = [len(f.get("function", "")) for f in functions]
    idx = int(max(range(len(lens)), key=lambda i: lens[i]))
    return idx, functions[idx]


def _related_repo_payload(
    this_file: str,
    all_details: List[Dict[str, Any]],
    exclude_code: str,
) -> Dict[str, List[Dict[str, str]]]:
    # 호출관계가 없으므로 근사치: 동일 파일 내는 callee, 다른 파일은 caller로 분류
    callees: List[Dict[str, str]] = []
    callers: List[Dict[str, str]] = []
    for d in all_details:
        file_name = d.get("file_name", "")
        for key in ("function_before", "function_after"):
            for f in d.get(key, []) or []:
                code = f.get("function")
                if not code or code == exclude_code:
                    continue
                item = {"file": file_name, "function": code}
                if file_name == this_file:
                    callees.append(item)
                else:
                    callers.append(item)
    return {"callee": callees, "caller": callers}


def build_multifunc_multifile(
    reposvul_path: str,
    save_path: str = "data/FuncFileRepo-MF.jsonl",
    chunksize: int | None = None,
) -> pd.DataFrame:
    """
    ReposVul 유사 스키마(JSONL)를 입력으로 받아, 다함수/다파일 사례를 포함한
    Func/File/Repo 실험용 JSONL을 생성합니다.

    출력 스키마(Detector 호환):
    - language: str        (python|c|c++|java)
    - function: str        (타깃 함수 코드; 다함수의 경우 대표 타깃)
    - file: str            (전체 파일 코드)
    - repository: {callee: [{file, function}], caller: [{file, function}]}
    - vulnerable: bool
    - 메타: cve_id, project, file_name, group_id, is_multi_function, is_multi_file, targets_count, files_count
    """

    def process_row(row, sink: List[Dict[str, Any]]):
        if row.get("outdated", 0) == 1:
            return

        cve_id = row.get("cve_id")
        cwe_id = tuple(row.get("cwe_id") or [])
        cvss = row.get("cvss")
        language = (row.get("cve_language") or "").lower()
        project = row.get("project")
        commit_id = row.get("commit_id")
        parents = row.get("parents") or []
        commit_id_before = parents[-1]["commit_id_before"] if parents else None
        details = row.get("details") or []

        # 파일 단위 수
        valid_details = [d for d in details if _is_valid_detail_language(d, language)]
        files_count = len(valid_details)
        is_multi_file = files_count > 1

        # 함수 수 카운트(변경 전 기준)
        before_funcs_total = 0
        for d in valid_details:
            before_funcs_total += len(d.get("function_before", []) or [])
        is_multi_function = before_funcs_total > 1

        # 그룹 식별자(취약/패치 페어 연결용)
        group_id = f"{project}::{cve_id}::{commit_id or ''}"

        for d in valid_details:
            file_name = d.get("file_name")
            code_before = d.get("code_before")
            code_after = d.get("code")
            fun_before = _collect_functions(d, "function_before")
            fun_after = _collect_functions(d, "function_after")

            # 대표 타깃 선택(변경 전)
            idx_b, prim_before = _choose_primary(fun_before)
            # 변경 후에서 타깃에 가장 유사한(동일 인덱스 우선) 함수 선택
            prim_after = {}
            if 0 <= idx_b < len(fun_after):
                prim_after = fun_after[idx_b]
            elif fun_after:
                prim_after = fun_after[0]

            # 저장소 컨텍스트 근사(호출관계 미사용)
            repo_ctx_before = _related_repo_payload(file_name, valid_details, prim_before.get("function", ""))
            repo_ctx_after = _related_repo_payload(file_name, valid_details, prim_after.get("function", ""))

            meta = {
                "cve_id": cve_id,
                "cwe_id": cwe_id,
                "cvss": cvss,
                "language": language,
                "project": project,
                "file_name": file_name,
                "group_id": group_id,
                "is_multi_function": is_multi_function,
                "is_multi_file": is_multi_file,
                "targets_count": before_funcs_total,
                "files_count": files_count,
            }

            # 취약(before)
            if prim_before and code_before:
                rec_vul = {
                    **meta,
                    "commit_id": commit_id_before,
                    "function": prim_before.get("function", ""),
                    "file": code_before,
                    "repository": repo_ctx_before,
                    "vulnerable": True,
                }
                sink.append(rec_vul)

            # 패치(after)
            if prim_after and code_after:
                rec_fix = {
                    **meta,
                    "commit_id": commit_id,
                    "function": prim_after.get("function", ""),
                    "file": code_after,
                    "repository": repo_ctx_after,
                    "vulnerable": False,
                }
                sink.append(rec_fix)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # 임시 누적 저장 방식: 청크 단위로 파일에 append
    if chunksize and chunksize > 0:
        # 초기화
        open(save_path, "w").close()
        agg = []
        for chunk in pd.read_json(reposvul_path, lines=True, chunksize=chunksize):
            agg.clear()
            for _, row in chunk.iterrows():
                process_row(row, agg)
            if agg:
                df_part = pd.DataFrame(agg)
                df_part.to_json(save_path, orient="records", lines=True, mode="a")
        # 최종 로드(정렬/중복제거)
        out_df = pd.read_json(save_path, lines=True)
        # 문자열 키로 중복 제거
        def _mk_key(r):
            return (
                r.get("project"), r.get("commit_id"), r.get("file_name"),
                r.get("vulnerable"), hash(r.get("function", "")), hash(r.get("file", "")),
            )
        out_df["__key__"] = out_df.apply(_mk_key, axis=1)
        out_df = out_df.sort_values(by=["project", "commit_id", "file_name"]).drop_duplicates(subset=["__key__"]).reset_index(drop=True)
        out_df = out_df.drop(columns=["__key__"], errors="ignore")
        out_df.to_json(save_path, orient="records", lines=True)
        return out_df
    else:
        df = pd.read_json(reposvul_path, lines=True)
        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            process_row(row, records)
        out_df = pd.DataFrame(records)
        if not out_df.empty:
            def _mk_key2(r):
                return (
                    r.get("project"), r.get("commit_id"), r.get("file_name"),
                    r.get("vulnerable"), hash(r.get("function", "")), hash(r.get("file", "")),
                )
            out_df["__key__"] = out_df.apply(_mk_key2, axis=1)
            out_df = out_df.sort_values(by=["project", "commit_id", "file_name"]).drop_duplicates(subset=["__key__"]).reset_index(drop=True)
            out_df = out_df.drop(columns=["__key__"], errors="ignore")
            out_df.to_json(save_path, orient="records", lines=True)
        return out_df


def main():
    import argparse

    p = argparse.ArgumentParser(description="Build multi-function/multi-file dataset (ReposVul JSONL → FuncFileRepo-MF JSONL)")
    p.add_argument("--input", "-i", required=True, help="Path to ReposVul-style JSONL")
    p.add_argument("--output", "-o", default="data/FuncFileRepo-MF.jsonl", help="Output JSONL path")
    p.add_argument("--chunksize", type=int, default=5000, help="Read JSONL in chunks (lines). Use 0 to disable.")
    args = p.parse_args()

    df = build_multifunc_multifile(args.input, args.output, chunksize=(args.chunksize or None))
    print({
        "samples": int(df.shape[0]) if not df.empty else 0,
        "vulnerable": int((df["vulnerable"] == True).sum()) if not df.empty else 0,
        "non_vulnerable": int((df["vulnerable"] == False).sum()) if not df.empty else 0,
        "multi_function_frac": float(df["is_multi_function"].mean()) if not df.empty else 0.0,
        "multi_file_frac": float(df["is_multi_file"].mean()) if not df.empty else 0.0,
    })


if __name__ == "__main__":
    main()
