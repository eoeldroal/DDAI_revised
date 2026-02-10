#!/usr/bin/env python3
"""Apply SHA1 dedup remapping: keep smallest doc_id, drop duplicates, remap GTI.

This updates:
- corpus_index/<dataset>.parquet
- qa/<dataset>/*.parquet (GTI remap)
- corpus images (removes non-canonical duplicates)

Input required:
filters/sha1_dedup/<dataset>/sha1_duplicates.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def pick_canonical(ids: Iterable[str]) -> str:
    return sorted(ids)[0]


def remap_list(values: list[str], remap: dict[str, str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        nv = remap.get(v, v)
        if nv not in seen:
            seen.add(nv)
            out.append(nv)
    return out


def apply_to_dataset(root: Path, dataset: str) -> None:
    dup_path = root / f"filters/sha1_dedup/{dataset}/sha1_duplicates.json"
    if not dup_path.exists():
        print(f"[WARN] missing duplicates file: {dup_path}")
        return

    with dup_path.open("r") as f:
        dup_groups: dict[str, list[str]] = json.load(f)

    remap: dict[str, str] = {}
    to_remove: set[str] = set()
    for _, ids in dup_groups.items():
        if len(ids) < 2:
            continue
        canon = pick_canonical(ids)
        for doc_id in ids:
            if doc_id == canon:
                continue
            remap[doc_id] = canon
            to_remove.add(doc_id)

    print(f"{dataset}: duplicate_groups={len(dup_groups)} remap_entries={len(remap)}")

    # Update corpus index
    idx_path = root / f"corpus_index/{dataset}.parquet"
    idx = pd.read_parquet(idx_path)
    orig_idx = len(idx)
    idx = idx[~idx["doc_id"].astype(str).isin(to_remove)].copy()
    idx.to_parquet(idx_path, index=False)
    print(f"{dataset}: index {orig_idx} -> {len(idx)}")

    # Remove images for non-canonical doc_ids
    img_root = root / f"corpus/{dataset}/images"
    removed_files = 0
    for doc_id in to_remove:
        img_path = img_root / Path(doc_id)
        if img_path.exists():
            img_path.unlink()
            removed_files += 1
    print(f"{dataset}: removed_images={removed_files}")

    # Remap GTI in QA parquet(s)
    qa_dir = root / f"qa/{dataset}"
    if qa_dir.exists():
        for qa_path in qa_dir.glob("*.parquet"):
            qa = pd.read_parquet(qa_path)
            if "gti" not in qa.columns:
                continue
            orig_rows = len(qa)
            qa["gti"] = qa["gti"].apply(
                lambda g: remap_list(g, remap) if isinstance(g, list) else g
            )
            qa.to_parquet(qa_path, index=False)
            print(f"{dataset}: remap GTI in {qa_path.name} rows={orig_rows}")

    # Save remap table
    remap_path = root / f"filters/sha1_dedup/{dataset}/remap.json"
    remap_path.write_text(json.dumps(remap, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, e.g. data/VDR_processed_filtered_1")
    parser.add_argument("--dataset", choices=["OpenDocVQA", "SlideVQA", "VDR_ibm", "all"], default="all")
    args = parser.parse_args()

    root = Path(args.root)
    datasets = ["OpenDocVQA", "SlideVQA", "VDR_ibm"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        apply_to_dataset(root, dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
