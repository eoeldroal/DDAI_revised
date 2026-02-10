#!/usr/bin/env python3
"""Apply SigLIP ANN near-duplicate dedup: remove images, update index, remap GTI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    if not path.exists():
        return pairs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def build_remap(pairs: list[dict]) -> tuple[dict[str, str], dict[str, set[str]]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for rec in pairs:
        union(rec["doc_id_a"], rec["doc_id_b"])

    groups: dict[str, set[str]] = {}
    for x in list(parent.keys()):
        r = find(x)
        groups.setdefault(r, set()).add(x)

    remap: dict[str, str] = {}
    for rep, members in groups.items():
        rep_id = min(members)
        for m in members:
            if m != rep_id:
                remap[m] = rep_id
    return remap, groups


def remap_gti(gti, remap: dict[str, str]):
    if gti is None:
        return gti
    if isinstance(gti, np.ndarray):
        gti = gti.tolist()
    if isinstance(gti, (list, tuple, set)):
        return [remap.get(x, x) for x in gti]
    return remap.get(gti, gti)


def apply_dataset(root: Path, dataset: str, cand_dir: str) -> None:
    pairs_path = root / cand_dir / f"{dataset}.jsonl"
    pairs = load_pairs(pairs_path)
    remap, groups = build_remap(pairs)
    print(f"{dataset}: pairs={len(pairs)} groups={len(groups)} remap={len(remap)}")

    # Update corpus index + delete images
    idx_path = root / f"corpus_index/{dataset}.parquet"
    idx = pd.read_parquet(idx_path)
    idx["doc_id"] = idx["doc_id"].astype(str)
    remove_ids = set(remap.keys())
    if remove_ids:
        # Delete images
        for rel in idx[idx["doc_id"].isin(remove_ids)]["path"].astype(str):
            p = root / rel
            if p.exists():
                p.unlink()
        # Filter index
        idx = idx[~idx["doc_id"].isin(remove_ids)].copy()
        idx.to_parquet(idx_path, index=False)

    # Remap GTI in QA
    if dataset == "OpenDocVQA":
        qa_path = root / "qa/OpenDocVQA/opendocvqa_train.parquet"
    elif dataset == "VDR_ibm":
        qa_path = root / "qa/VDR_ibm/vdr_ibm_train.parquet"
    else:
        return

    if remap:
        qa = pd.read_parquet(qa_path)
        qa["gti"] = qa["gti"].apply(lambda g: remap_gti(g, remap))
        qa.to_parquet(qa_path, index=False)

    # Save remap for auditing
    out_remap = root / cand_dir / f"{dataset}_remap.json"
    out_remap.write_text(json.dumps(remap, ensure_ascii=True, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", choices=["OpenDocVQA", "VDR_ibm", "all"], required=True)
    parser.add_argument("--cand-dir", default="filters/siglip_ann_candidates")
    args = parser.parse_args()

    datasets = ["OpenDocVQA", "VDR_ibm"] if args.dataset == "all" else [args.dataset]
    root = Path(args.root)
    for ds in datasets:
        apply_dataset(root, ds, args.cand_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
