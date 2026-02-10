#!/usr/bin/env python3
"""Async doc-independent filter using OpenAI-compatible client."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import httpx
from openai import AsyncOpenAI


SYSTEM_PROMPT = (
    "You are a strict judge. Decide if the question can be answered from general knowledge "
    "alone, without any document, chart, table, figure, page, section, or specific record.\n"
    "Say 'yes' ONLY for universally known facts (e.g., basic geography, common definitions) "
    "that do not depend on a particular source or named document.\n"
    "If the question involves a specific person/organization, addresses, contract numbers, "
    "dates/years, numeric values, chart/axis/table details, or document sections, answer 'no'.\n"
    "If unsure, answer 'no'. When doc_independent='no', predicted_answer must be null."
)

USER_TEMPLATE = "Question: {query}"

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "doc_independent_check",
        "schema": {
            "type": "object",
            "properties": {
                "doc_independent": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "predicted_answer": {"type": ["string", "null"]},
                "rationale_short": {"type": "string"},
            },
            "required": ["doc_independent", "confidence", "predicted_answer", "rationale_short"],
            "additionalProperties": False,
        },
    },
}


async def call_llm(client: AsyncOpenAI, model: str, query: str, parse_retries: int = 2) -> dict:
    last_err: Exception | None = None
    for _ in range(parse_retries + 1):
        resp = await client.chat.completions.create(
            model=model,
            temperature=0.7,
            response_format=RESPONSE_SCHEMA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(query=query)},
            ],
        )
        content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except Exception as e:
            last_err = e
            continue
    assert last_err is not None
    raise last_err


def iter_resume_files(out_dir: Path, dataset_name: str) -> list[Path]:
    return sorted(out_dir.glob(f"{dataset_name}_*.jsonl"))


def load_completed_runs(paths: list[Path], runs: int) -> tuple[dict[str, set[int]], set[str]]:
    completed: dict[str, set[int]] = {}
    if not paths:
        return completed, set()
    full_set = set(range(runs))
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = str(rec.get("id"))
                run_idx = rec.get("run_idx", None)
                if run_idx is None:
                    completed[rid] = set(full_set)
                    continue
                try:
                    idx = int(run_idx)
                except Exception:
                    continue
                completed.setdefault(rid, set()).add(idx)
    completed_full = {rid for rid, idxs in completed.items() if len(idxs) >= runs}
    return completed, completed_full


async def run_dataset(
    client: AsyncOpenAI,
    dataset_name: str,
    df: pd.DataFrame,
    out_path: Path,
    model: str,
    runs: int,
    concurrency: int,
    log_every: int,
    counter: dict,
    counter_lock: asyncio.Lock,
    resume: bool,
    completed_runs: dict[str, set[int]] | None,
    completed_full: set[str] | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)
    completed = completed_runs or {}
    completed_full = completed_full or set()
    file_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    queue: asyncio.Queue = asyncio.Queue()
    pending: dict[str, int] = {}
    for row in df.itertuples(index=False):
        rid = str(row.id)
        if completed_full and rid in completed_full:
            continue
        done_runs = completed.get(rid, set()) if completed else set()
        missing = [idx for idx in range(runs) if idx not in done_runs]
        if not missing:
            continue
        pending[rid] = len(missing)
        query = str(row.query)
        answer = None if pd.isna(row.answer) else str(row.answer)
        for idx in missing:
            queue.put_nowait((rid, query, answer, idx))
    for _ in range(concurrency):
        queue.put_nowait(None)

    async def worker():
        while True:
            row = await queue.get()
            if row is None:
                queue.task_done()
                return
            rid, query, answer, idx = row
            try:
                async with sem:
                    res = await call_llm(client, model, query)
                run_rec = {
                    "dataset": dataset_name,
                    "id": rid,
                    "query": query,
                    "answer": answer,
                    "run_idx": idx,
                    "doc_independent": res.get("doc_independent"),
                    "confidence": res.get("confidence"),
                    "predicted_answer": res.get("predicted_answer"),
                    "rationale_short": res.get("rationale_short"),
                }
                async with file_lock:
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(run_rec, ensure_ascii=True) + "\n")
                async with counter_lock:
                    pending[rid] -= 1
                    if pending[rid] == 0:
                        counter["done"] += 1
                        if log_every > 0 and counter["done"] % log_every == 0:
                            print(
                                f"[progress] {counter['done']}/{counter['total']} "
                                f"total (last dataset={dataset_name})"
                            )
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await queue.join()
    for w in workers:
        await w
    print(f"[dataset] {dataset_name} done {len(df)}")


def build_http_client(timeout: httpx.Timeout, max_connections: int, max_keepalive: int) -> httpx.AsyncClient:
    connect_timeout = timeout.connect if timeout.connect is not None else 10.0
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
        keepalive_expiry=connect_timeout,
    )
    return httpx.AsyncClient(timeout=timeout, limits=limits)


async def run(args) -> int:
    timeout = httpx.Timeout(
        connect=args.connect_timeout,
        read=args.read_timeout,
        write=args.write_timeout,
        pool=args.pool_timeout,
    )
    http_client = build_http_client(timeout, args.max_connections, args.max_keepalive)
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=timeout,
        max_retries=args.max_retries,
        http_client=http_client,
    )
    counter_lock = asyncio.Lock()

    datasets = {}
    if args.input:
        df = pd.read_parquet(args.input, columns=["id", "query", "answer"])
        if args.max_rows > 0:
            df = df.head(args.max_rows)
        datasets["custom"] = df
    else:
        root = Path(args.root) / args.qa_root
        datasets["OpenDocVQA"] = pd.read_parquet(
            root / "OpenDocVQA/opendocvqa_train.parquet", columns=["id", "query", "answer"]
        )
        datasets["VDR_ibm"] = pd.read_parquet(
            root / "VDR_ibm/vdr_ibm_train.parquet", columns=["id", "query", "answer"]
        )
        for split in ["train", "val", "test"]:
            datasets[f"SlideVQA_{split}"] = pd.read_parquet(
                root / f"SlideVQA/slidevqa_{split}.parquet", columns=["id", "query", "answer"]
            )
        if args.max_rows > 0:
            for k in list(datasets.keys()):
                datasets[k] = datasets[k].head(args.max_rows)

    resume_state: dict[str, tuple[dict[str, set[int]], set[str]]] = {}
    total = 0
    for name, df in datasets.items():
        if args.resume:
            resume_files = iter_resume_files(Path(args.out_dir), name)
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
            args.runs,
            args.concurrency,
            args.log_every,
            counter,
            counter_lock,
            args.resume,
            completed_runs,
            completed_full,
        )
    print(f"[done] total rows={counter['done']}/{counter['total']}")
    await http_client.aclose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="Single parquet with columns: id, query, answer")
    parser.add_argument("--root", default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1")
    parser.add_argument("--qa-root", default="qa_query_dedup")
    parser.add_argument("--out-dir", default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_1/filters/doc_independent")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Skip IDs already present in output JSONL")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=1200.0)
    parser.add_argument("--write-timeout", type=float, default=1200.0)
    parser.add_argument("--pool-timeout", type=float, default=60.0)
    parser.add_argument("--max-connections", type=int, default=1024)
    parser.add_argument("--max-keepalive", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--timestamp", default="")
    args = parser.parse_args()

    if not args.timestamp:
        args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
