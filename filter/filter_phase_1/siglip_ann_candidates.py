#!/usr/bin/env python3
"""SigLIP embedding ANN candidate generator for near-duplicate images.

This builds a FAISS index per dataset and emits candidate pairs above a
similarity threshold. Designed for high-recall filtering (top-k + strict
threshold).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def _require_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "faiss is required. Install with: pip install faiss-cpu or faiss-gpu"
        ) from exc
    return faiss


def load_embeddings(root: Path, dataset: str, emb_dir: str) -> tuple[np.ndarray, list[str]]:
    base = root / emb_dir / dataset
    pooled = []
    for p in sorted(base.glob("pooled_rank*.pt")):
        batches = torch.load(p, map_location="cpu")
        if isinstance(batches, torch.Tensor):
            pooled.append(batches)
        else:
            pooled.append(torch.cat(batches, dim=0))
    if not pooled:
        raise SystemExit(f"No pooled_rank*.pt found under {base}")
    vecs = torch.cat(pooled, dim=0).float().numpy()

    doc_ids: list[str] = []
    for p in sorted(base.glob("doc_ids_rank*.txt")):
        text = p.read_text().strip()
        if text:
            doc_ids.extend(text.splitlines())
    if len(doc_ids) != len(vecs):
        raise SystemExit(f"doc_ids ({len(doc_ids)}) != embeddings ({len(vecs)}) for {dataset}")
    return vecs, doc_ids


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def build_index(
    faiss,
    vecs: np.ndarray,
    index_type: str,
    nlist: int,
    nprobe: int,
    hnsw_m: int,
    hnsw_efc: int,
    hnsw_efs: int,
):
    dim = vecs.shape[1]
    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_efc
        index.hnsw.efSearch = hnsw_efs
        index.add(vecs)
        return index

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = nprobe
    return index


def iter_chunks(n: int, chunk: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", choices=["OpenDocVQA", "SlideVQA", "VDR_ibm", "all"], required=True)
    parser.add_argument("--emb-dir", default="filters/siglip_embeddings")
    parser.add_argument("--out-dir", default="filters/siglip_ann_candidates")
    parser.add_argument("--index", choices=["hnsw", "ivf"], default="hnsw")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--sim-threshold", type=float, default=0.985)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-efc", type=int, default=200)
    parser.add_argument("--hnsw-efs", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    faiss = _require_faiss()

    datasets = ["OpenDocVQA", "SlideVQA", "VDR_ibm"] if args.dataset == "all" else [args.dataset]
    root = Path(args.root)
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        vecs, doc_ids = load_embeddings(root, dataset, args.emb_dir)
        vecs = normalize(vecs).astype(np.float32, copy=False)
        total = len(vecs)

        if args.index == "ivf" and args.nlist > total:
            args.nlist = max(256, int(math.sqrt(total)))

        index = build_index(
            faiss,
            vecs,
            args.index,
            args.nlist,
            args.nprobe,
            args.hnsw_m,
            args.hnsw_efc,
            args.hnsw_efs,
        )
        if args.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, args.gpu_id, index)

        out_path = out_root / f"{dataset}.jsonl"
        kept = 0
        seen = 0
        with out_path.open("w", encoding="utf-8") as f:
            for step, (start, end) in enumerate(iter_chunks(total, args.chunk_size), start=1):
                q = vecs[start:end]
                scores, indices = index.search(q, args.topk + 1)
                for i in range(end - start):
                    qidx = start + i
                    for j, score in zip(indices[i], scores[i]):
                        if j < 0:
                            continue
                        if j == qidx:
                            continue
                        if j < qidx:
                            continue
                        if score < args.sim_threshold:
                            continue
                        rec = {
                            "doc_id_a": doc_ids[qidx],
                            "doc_id_b": doc_ids[j],
                            "score": float(score),
                        }
                        f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                        kept += 1
                        if args.max_pairs and kept >= args.max_pairs:
                            break
                    if args.max_pairs and kept >= args.max_pairs:
                        break
                seen += (end - start)
                if step % args.log_every == 0:
                    pct = (seen / total) * 100.0
                    print(
                        f"{dataset}: {seen}/{total} ({pct:.1f}%) pairs={kept}",
                        flush=True,
                    )
                if args.max_pairs and kept >= args.max_pairs:
                    break
        print(f"{dataset}: candidates written to {out_path} (pairs={kept})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
