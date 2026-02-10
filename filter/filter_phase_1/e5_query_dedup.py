#!/usr/bin/env python3
"""Query dedup candidates with multilingual-e5-large-instruct.

Pipeline:
1) Load QA (id, query, gti) from qa_root
2) Embed queries with E5 (query: prefix)
3) Build FAISS index (cosine via normalized vectors)
4) Emit candidate pairs above threshold (for GTI merge)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel


def require_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("faiss is required. Install faiss-cpu or faiss-gpu.") from exc
    return faiss


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def iter_chunks(n: int, chunk: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def load_qa(qa_root: Path) -> pd.DataFrame:
    files = [
        qa_root / "OpenDocVQA/opendocvqa_train.parquet",
        qa_root / "SlideVQA/slidevqa_train.parquet",
        qa_root / "SlideVQA/slidevqa_val.parquet",
        qa_root / "SlideVQA/slidevqa_test.parquet",
        qa_root / "VDR_ibm/vdr_ibm_train.parquet",
    ]
    frames = []
    for p in files:
        df = pd.read_parquet(p, columns=["id", "query", "gti"])
        df["query"] = df["query"].fillna("")
        df["source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def embed_queries(
    model_name: str,
    queries: list[str],
    device: str,
    batch_size: int,
    prompt_prefix: str,
    trust_remote_code: bool,
    use_cache: bool,
    attn_impl: str,
) -> np.ndarray:
    device = device.strip()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else None,
    ).to(device).eval()
    if hasattr(model, "config"):
        model.config.use_cache = use_cache

    vecs = []
    for start, end in iter_chunks(len(queries), batch_size):
        batch = [prompt_prefix + q for q in queries[start:end]]
        toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.inference_mode():
            out = model(**toks, use_cache=use_cache)
            pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vecs.append(pooled.cpu())
    return torch.cat(vecs, dim=0).numpy()


def build_index(faiss, vecs: np.ndarray, index_type: str, nlist: int, nprobe: int):
    dim = vecs.shape[1]
    if index_type == "flat":
        return faiss.IndexFlatIP(dim)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = nprobe
    return index


def _embed_shard(
    rank: int,
    world_size: int,
    model_name: str,
    queries: list[str],
    ids: list[str],
    out_dir: Path,
    batch_size: int,
    prompt_prefix: str,
    trust_remote_code: bool,
    use_cache: bool,
    attn_impl: str,
):
    device = f"cuda:{rank}"
    shard_q = queries[rank::world_size]
    shard_ids = ids[rank::world_size]
    vecs = embed_queries(
        model_name,
        shard_q,
        device,
        batch_size,
        prompt_prefix,
        trust_remote_code,
        use_cache,
        attn_impl,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"emb_rank{rank:02d}.npy", vecs)
    (out_dir / f"ids_rank{rank:02d}.txt").write_text("\n".join(shard_ids))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--qa-root", default="qa_filtered_1")
    parser.add_argument("--model", default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--out-dir", default="filters/query_dedup_e5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-prefix", default="query: ")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", default=True)
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--index", choices=["flat", "ivf"], default="ivf")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root)
    qa_root = root / args.qa_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_qa(qa_root)
    queries = df["query"].astype(str).tolist()
    ids = df["id"].astype(str).tolist()

    if args.gpus > 1:
        shard_dir = out_dir / "emb_shards"
        mp.spawn(
            _embed_shard,
            args=(
                args.gpus,
                args.model,
                queries,
                ids,
                shard_dir,
                args.batch_size,
                args.prompt_prefix,
                args.trust_remote_code,
                args.use_cache,
                args.attn_impl,
            ),
            nprocs=args.gpus,
            join=True,
        )
        shard_vecs = []
        shard_ids = []
        for p in sorted(shard_dir.glob("emb_rank*.npy")):
            shard_vecs.append(np.load(p))
        for p in sorted(shard_dir.glob("ids_rank*.txt")):
            text = p.read_text().strip()
            if text:
                shard_ids.extend(text.splitlines())
        vecs = np.vstack(shard_vecs)
        ids = shard_ids
    else:
        vecs = embed_queries(
            args.model,
            queries,
            args.device,
            args.batch_size,
            args.prompt_prefix,
            args.trust_remote_code,
            args.use_cache,
            args.attn_impl,
        )
    vecs = vecs.astype(np.float32, copy=False)

    faiss = require_faiss()
    if args.index == "ivf" and args.nlist > len(vecs):
        args.nlist = max(256, int(np.sqrt(len(vecs))))

    index = build_index(faiss, vecs, args.index, args.nlist, args.nprobe)
    if args.index == "flat":
        index.add(vecs)

    out_pairs = out_dir / "query_candidates.jsonl"
    kept = 0
    with out_pairs.open("w", encoding="utf-8") as f:
        for step, (start, end) in enumerate(iter_chunks(len(vecs), args.chunk_size), start=1):
            q = vecs[start:end]
            scores, indices = index.search(q, args.topk + 1)
            for i in range(end - start):
                qidx = start + i
                for j, score in zip(indices[i], scores[i]):
                    if j < 0 or j == qidx or j < qidx:
                        continue
                    if score < args.threshold:
                        continue
                    rec = {"id_a": ids[qidx], "id_b": ids[j], "score": float(score)}
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    kept += 1
            if step % args.log_every == 0:
                pct = (end / len(vecs)) * 100.0
                print(f"queries {end}/{len(vecs)} ({pct:.1f}%) pairs={kept}", flush=True)

    print(f"wrote {out_pairs} (pairs={kept})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
