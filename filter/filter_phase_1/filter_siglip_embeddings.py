#!/usr/bin/env python3
"""Filter SigLIP embeddings to match current corpus_index (doc_id allowlist)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def load_doc_ids(path: Path) -> list[str]:
    text = path.read_text().strip()
    return text.splitlines() if text else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", choices=["OpenDocVQA", "SlideVQA", "VDR_ibm"], required=True)
    parser.add_argument("--emb-dir", default="filters/siglip_embeddings")
    parser.add_argument("--out-dir", default="filters/siglip_embeddings_filtered")
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    idx_path = root / f"corpus_index/{args.dataset}.parquet"
    idx = pd.read_parquet(idx_path, columns=["doc_id"])
    allow = set(idx["doc_id"].astype(str).tolist())

    emb_base = root / args.emb_dir / args.dataset
    out_base = emb_base if args.inplace else (root / args.out_dir / args.dataset)
    out_base.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0

    for ids_path in sorted(emb_base.glob("doc_ids_rank*.txt")):
        rank = ids_path.stem.replace("doc_ids_", "")
        pooled_path = emb_base / f"pooled_{rank}.pt"
        if not pooled_path.exists():
            continue

        doc_ids = load_doc_ids(ids_path)
        total_in += len(doc_ids)

        pooled = torch.load(pooled_path, map_location="cpu")
        if isinstance(pooled, torch.Tensor):
            emb = pooled
        else:
            emb = torch.cat(pooled, dim=0)
        if len(doc_ids) != emb.shape[0]:
            raise SystemExit(f"Length mismatch in {rank}: ids={len(doc_ids)} emb={emb.shape[0]}")

        keep_mask = [d in allow for d in doc_ids]
        keep_ids = [d for d, k in zip(doc_ids, keep_mask) if k]
        keep_idx = torch.tensor([i for i, k in enumerate(keep_mask) if k], dtype=torch.long)
        emb_f = emb.index_select(0, keep_idx)

        out_ids = out_base / ids_path.name
        out_pool = out_base / pooled_path.name
        out_ids.write_text("\n".join(keep_ids))
        torch.save(emb_f, out_pool)

        total_out += len(keep_ids)

    print(f"{args.dataset}: filtered embeddings {total_in} -> {total_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
