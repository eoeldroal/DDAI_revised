#!/usr/bin/env python3
"""Async GTI validation filter using OpenAI-compatible client (image + query + answer)."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI


SYSTEM_PROMPT = (
    "You are a strict verifier. The provided answer is NOT a hint. "
    "First decide if the question is answerable from the image alone (visual reasoning allowed). "
    "Then verify whether the provided answer is explicitly supported by the image. "
    "If either fails, verdict must be 'no'. If unsure, respond 'no'. "
    "Return JSON only."
)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "gti_check_simple",
        "schema": {
            "type": "object",
            "properties": {
                "rationale_short": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "verdict": {"type": "string", "enum": ["yes", "no"]},
            },
            "required": ["rationale_short", "confidence", "verdict"],
            "additionalProperties": False,
        },
    },
}


async def call_llm(
    client: AsyncOpenAI,
    model: str,
    query: str,
    answer: str,
    image_path: Path,
    temperature: float,
    max_tokens: int,
    parse_retries: int = 2,
) -> dict:
    """이미지 1장 + (query, answer)를 넣어 멀티모달 호출을 수행한다.

    이 함수는 다음 순서로 동작한다:
    1) 이미지 파일을 읽어 base64 data URL로 변환한다.
    2) system/user 메시지를 구성해 OpenAI-compatible API로 요청한다.
    3) 응답 본문을 JSON으로 파싱하고, 실패 시 정해진 횟수만큼 재시도한다.

    반환:
      - 성공 시 파싱된 dict를 반환한다.
      - 모든 재시도가 실패하면 마지막 예외를 raise한다.
    """
    last_err: Exception | None = None
    for _ in range(parse_retries + 1):
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        user_text = f"Question: {query}\nProposed Answer: {answer}\n"
        resp = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=RESPONSE_SCHEMA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        )
        content = resp.choices[0].message.content
        if content is None:
            last_err = ValueError("empty content")
            continue
        try:
            return json.loads(content)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def load_completed_runs(paths: list[Path], runs: int) -> tuple[dict[str, set[int]], set[str]]:
    """JSONL 로그를 읽어 완료된 run_idx를 복원한다.

    각 JSONL 레코드를 읽어 id와 run_idx를 누적하며,
    특정 id가 runs 횟수만큼 수행되었는지 판단한다.

    반환값:
      - completed: id -> 완료된 run_idx 집합 (부분 완료 포함)
      - completed_full: 모든 run이 끝난 id 집합
    """
    completed: dict[str, set[int]] = {}
    full_set = set(range(runs))
    if not paths:
        return completed, set()

    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        rid = str(rec["id"])
                    except Exception:
                        continue
                    run_idx = rec.get("run_idx")
                    if run_idx is None:
                        completed[rid] = set(full_set)
                    else:
                        try:
                            completed.setdefault(rid, set()).add(int(run_idx))
                        except Exception:
                            continue
        except FileNotFoundError:
            continue
    completed_full = {rid for rid, idxs in completed.items() if len(idxs) >= runs}
    return completed, completed_full


def build_corpus_index(index_dir: Path) -> dict[str, dict[str, str]]:
    """코퍼스 인덱스에서 doc_id -> 이미지 경로 매핑을 로드한다.

    - OpenDocVQA / SlideVQA / VDR_ibm 각각의 corpus_index parquet를 읽는다.
    - 각 데이터셋에 대해 doc_id -> path 사전을 만든다.
    - 이후 GTI doc_id를 실제 이미지 파일 경로로 변환할 때 사용한다.
    """
    idx_maps: dict[str, dict[str, str]] = {}
    for name in ["OpenDocVQA", "SlideVQA", "VDR_ibm"]:
        df = pd.read_parquet(index_dir / f"{name}.parquet", columns=["doc_id", "path"])
        idx_maps[name] = dict(zip(df["doc_id"], df["path"]))
    return idx_maps


def resolve_idx_key(dataset_name: str, doc_id: str) -> str:
    """dataset_name이 custom일 때 doc_id 접두사로 인덱스 키를 결정한다."""
    if dataset_name.startswith("SlideVQA") or doc_id.startswith("slidevqa/"):
        return "SlideVQA"
    if doc_id.startswith("vdr/"):
        return "VDR_ibm"
    return "OpenDocVQA"


async def run_dataset(
    client: AsyncOpenAI,
    dataset_name: str,
    df: pd.DataFrame,
    out_path: Path,
    model: str,
    idx_maps: dict[str, dict[str, str]],
    corpus_root: Path,
    runs: int,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    log_every: int,
    counter: dict,
    counter_lock: asyncio.Lock,
    completed_runs: dict[str, set[int]] | None,
    completed_full: set[str] | None,
) -> None:
    """하나의 데이터셋 split을 처리하고 결과를 JSONL로 저장한다.

    주요 흐름:
    - 이미 완료된 run_idx는 건너뛴다(resume 지원).
    - run 단위 작업을 생성하고, as_completed로 완료되는 즉시 처리한다.
    - 세마포어로 동시에 실행되는 API 호출 수를 제한한다.
    - 결과는 JSONL 한 줄로 append하며, 전역 진행률을 갱신한다.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_runs or {}
    completed_full = completed_full or set()
    file_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    idx_key = "SlideVQA" if dataset_name.startswith("SlideVQA") else dataset_name

    remaining_runs: dict[str, int] = {}
    tasks: list[asyncio.Task] = []

    async def run_one(rid: str, query: str, answer: str, gti_list: list[str], idx: int) -> tuple[str, dict]:
        async with sem:
            per_gti = []
            for gti_idx, doc_id in enumerate(gti_list):
                key = idx_key if idx_key != "custom" else resolve_idx_key(dataset_name, doc_id)
                rel_path = idx_maps[key][doc_id]
                img_path = corpus_root / rel_path
                res = await call_llm(
                    client,
                    model,
                    query,
                    answer,
                    img_path,
                    temperature,
                    max_tokens,
                )
                per_gti.append(
                    {
                        "gti_idx": gti_idx,
                        "doc_id": doc_id,
                        "rationale_short": res.get("rationale_short"),
                        "confidence": res.get("confidence"),
                        "verdict": res.get("verdict"),
                    }
                )
            return rid, {
                "dataset": dataset_name,
                "id": rid,
                "query": query,
                "answer": answer,
                "run_idx": idx,
                "gti_results": per_gti,
            }

    for row in df.itertuples(index=False):
        rid = str(row.id)
        if completed_full and rid in completed_full:
            continue
        done_runs = completed.get(rid, set())
        missing = [idx for idx in range(runs) if idx not in done_runs]
        if not missing:
            continue
        remaining_runs[rid] = len(missing)
        query = str(row.query)
        answer = "" if pd.isna(row.answer) else str(row.answer)
        gti_list = list(row.gti)
        for idx in missing:
            tasks.append(asyncio.create_task(run_one(rid, query, answer, gti_list, idx)))

    for fut in asyncio.as_completed(tasks):
        try:
            rid, run_rec = await fut
        except Exception as exc:
            print(f"[error] dataset={dataset_name} error={exc}")
            continue
        async with file_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(run_rec, ensure_ascii=True) + "\n")
        async with counter_lock:
            remaining_runs[rid] -= 1
            if remaining_runs[rid] == 0:
                counter["done"] += 1
                if log_every > 0 and counter["done"] % log_every == 0:
                    print(
                        f"[progress] {counter['done']}/{counter['total']} "
                        f"total (last dataset={dataset_name})"
                    )
    print(f"[dataset] {dataset_name} done {len(df)}")


async def run(args) -> int:
    """상위 실행 함수: 데이터셋 로딩, resume 상태 복원, split별 실행.

    - 입력 경로(전체 또는 단일 parquet)에 따라 데이터셋을 준비한다.
    - --resume 옵션이 있으면 기존 JSONL로부터 완료 상태를 복원한다.
    - 각 데이터셋 split을 순차적으로 처리한다.
    """
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    counter_lock = asyncio.Lock()

    datasets = {}
    if args.input:
        df = pd.read_parquet(args.input, columns=["id", "query", "answer", "gti"])
        if args.max_rows > 0:
            df = df.head(args.max_rows)
        datasets["custom"] = df
    else:
        root = Path(args.root) / args.qa_root
        datasets["OpenDocVQA"] = pd.read_parquet(
            root / "OpenDocVQA/opendocvqa_train.parquet", columns=["id", "query", "answer", "gti"]
        )
        datasets["VDR_ibm"] = pd.read_parquet(
            root / "VDR_ibm/vdr_ibm_train.parquet", columns=["id", "query", "answer", "gti"]
        )
        for split in ["train", "val", "test"]:
            datasets[f"SlideVQA_{split}"] = pd.read_parquet(
                root / f"SlideVQA/slidevqa_{split}.parquet", columns=["id", "query", "answer", "gti"]
            )
        if args.max_rows > 0:
            for k in list(datasets.keys()):
                datasets[k] = datasets[k].head(args.max_rows)

    idx_maps = build_corpus_index(Path(args.index_dir))
    corpus_root = Path(args.corpus_root)

    resume_state: dict[str, tuple[dict[str, set[int]], set[str]]] = {}
    total = 0
    for name, df in datasets.items():
        if args.resume:
            resume_files = sorted(Path(args.out_dir).glob(f"{name}_*.jsonl"))
            completed_runs, completed_full = load_completed_runs(resume_files, args.runs)
            resume_state[name] = (completed_runs, completed_full)
            remaining = len(df) - len(completed_full)
            print(f"[resume] {name}: skip {len(completed_full)}/{len(df)} rows")
            total += max(remaining, 0)
        else:
            resume_state[name] = ({}, set())
            total += len(df)
    counter = {"done": 0, "total": total}
    print(f"[start] total rows={total}, datasets={list(datasets.keys())}")

    ts = args.timestamp
    for name, df in datasets.items():
        out_path = Path(args.out_dir) / f"{name}_{ts}.jsonl"
        completed_runs, completed_full = resume_state.get(name, ({}, set()))
        await run_dataset(
            client,
            name,
            df,
            out_path,
            args.model,
            idx_maps,
            corpus_root,
            args.runs,
            args.temperature,
            args.max_tokens,
            args.concurrency,
            args.log_every,
            counter,
            counter_lock,
            completed_runs,
            completed_full,
        )
    print(f"[done] total rows={counter['done']}/{counter['total']}")
    return 0


def main() -> int:
    """CLI 엔트리포인트: 인자 파싱 후 비동기 파이프라인 실행.

    - 기본값과 사용자 옵션을 결합해 실행 파라미터를 구성한다.
    - timestamp가 없으면 현재 시각으로 자동 생성한다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Single parquet with columns: id, query, answer, gti")
    parser.add_argument("--root", default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2")
    parser.add_argument("--qa-root", default="qa_doc_independent_filtered")
    parser.add_argument("--index-dir", default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1/corpus_index")
    parser.add_argument("--corpus-root", default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1")
    parser.add_argument("--out-dir", default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/filters/gti_validation")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in output JSONL")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=1200.0, help="Total request timeout (seconds)")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--timestamp", default="")
    args = parser.parse_args()

    if not args.timestamp:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
