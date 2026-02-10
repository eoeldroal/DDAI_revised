#!/usr/bin/env python3
"""Compute SHA1 for corpus images and build duplicate groups per dataset.

This does NOT delete anything. It produces a report and progress logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def sha1_file(path: Path, chunk_size: int) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _worker(task: tuple[str, str, int]) -> tuple[str, str]:
    rel, abs_path, chunk_size = task
    digest = sha1_file(Path(abs_path), chunk_size=chunk_size)
    return rel, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, e.g. data/VDR_processed_filtered_1")
    parser.add_argument("--dataset", choices=["OpenDocVQA", "SlideVQA", "VDR_ibm", "all"], default="all")
    parser.add_argument("--out", default="filters/sha1_dedup")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--map-chunksize", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=5000)
    args = parser.parse_args()

    root = Path(args.root)

    datasets = ["OpenDocVQA", "SlideVQA", "VDR_ibm"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        idx_path = root / f"corpus_index/{dataset}.parquet"
        if not idx_path.exists():
            print(f"[WARN] missing index: {idx_path}")
            continue
        df = pd.read_parquet(idx_path, columns=["doc_id", "path"])

        out_dir = root / args.out / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        doc_id_map = dict(zip(df["path"].astype(str), df["doc_id"].astype(str)))
        tasks = []
        for rel in doc_id_map.keys():
            img_path = root / rel
            if img_path.exists():
                tasks.append((rel, str(img_path), args.chunk_bytes))

        total = len(tasks)
        sha_map: dict[str, list[str]] = {}

        print(f"dataset={dataset} total_images={total}")
        print(f"workers={args.workers} chunk_bytes={args.chunk_bytes} map_chunksize={args.map_chunksize}")
        start = time.time()

        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rel, digest in ex.map(_worker, tasks, chunksize=args.map_chunksize):
                doc_id = doc_id_map.get(rel)
                if doc_id is None:
                    continue
                sha_map.setdefault(digest, []).append(doc_id)
                done += 1
                if done % args.log_every == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0.0
                    print(f"progress: {done}/{total} ({rate:.1f} imgs/s)")

        dup_groups = {k: v for k, v in sha_map.items() if len(v) > 1}

        (out_dir / "sha1_groups.json").write_text(json.dumps(sha_map, indent=2))
        (out_dir / "sha1_duplicates.json").write_text(json.dumps(dup_groups, indent=2))

        elapsed = time.time() - start
        print(f"total images: {len(df)}")
        print(f"duplicate groups: {len(dup_groups)}")
        print(f"elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
