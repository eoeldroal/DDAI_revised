#!/usr/bin/env python3
"""
ColQwen2 기반 덱 재구성 (샘플 실험용)

기능
 - QA 파켓(필드 id, query, gti, answer)에서 샘플링
 - ColQwen2 사전계산 멀티벡터 임베딩(이미지) 로드
 - 멀티벡터 평균(pool)으로 FAISS HNSW 인덱스 구축 후 후보 검색
 - ColBERT-style late interaction으로 재랭크 (쿼리=원문+리라이팅 중 최고 점수)
 - GTI 필수 포함 + hard/semi/fallback 규칙으로 덱 20장 구성
 - 덱, 로그, 통계 산출

주의
 - 러닝타임/메모리 비용 높음. colqwen 가상환경에서 실행 권장.
 - 현재 스크립트는 실험/프로토타입용. 전체 데이터 실행 전 소규모 드라이런 필요.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import ColQwen2ForRetrieval, ColQwen2Processor


@dataclass
class EmbShard:
    doc_ids: List[str]
    embeddings: List[torch.Tensor]  # list [T,128] bf16
    paths: List[str]
    split: List[str]


def load_shards(emb_root: Path, dataset: str) -> Tuple[List[str], np.ndarray]:
    """Load pooled embeddings (mean over tokens) for ANN."""
    doc_ids, pooled = [], []
    shards = sorted((emb_root / dataset).glob("shard_*.pt"))
    for fn in shards:
        obj = torch.load(fn, map_location="cpu")
        for emb, doc_id in zip(obj["embeddings"], obj["doc_ids"]):
            pooled.append(emb.mean(0).float().numpy())
            doc_ids.append(doc_id)
    pooled_np = np.vstack(pooled)
    return doc_ids, pooled_np


def build_hnsw(x: np.ndarray, m: int = 32, efc: int = 200) -> faiss.Index:
    dim = x.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = efc
    index.add(x)
    return index


def load_query_rewrites(path: Path) -> Dict[str, List[str]]:
    rewrites = defaultdict(list)
    if not path.exists():
        return rewrites
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            qid = rec.get("id") or rec.get("qa_id") or rec.get("dataset_id")
            if not qid:
                continue
            # format A: {"rewrites": [str,...]}  (most common)
            if isinstance(rec.get("rewrites"), list):
                for txt in rec["rewrites"]:
                    if isinstance(txt, str) and txt.strip():
                        rewrites[qid].append(txt.strip())
                continue
            # format B: {"runs": [ {text:..}, ... ]} or ["..."]
            runs = rec.get("runs") or []
            if isinstance(runs, list):
                for r in runs:
                    if isinstance(r, str):
                        if r.strip():
                            rewrites[qid].append(r.strip())
                    elif isinstance(r, dict):
                        txt = r.get("text") or r.get("rewrite") or r.get("query")
                        if txt:
                            rewrites[qid].append(str(txt).strip())
    return rewrites


def colbert_score(q: torch.Tensor, d: torch.Tensor) -> float:
    """
    q: [Tq, 128], d: [Td, 128]
    score = sum over q tokens of max(dot(q_i, d_j))
    """
    # dot: [Tq, Td]
    dot = torch.matmul(q, d.t())
    return torch.max(dot, dim=1).values.sum().item()


def query_embeddings(model, proc, query: str, rewrites: List[str], device: torch.device) -> List[torch.Tensor]:
    texts = [query] + rewrites
    embs = []
    for t in texts:
        toks = proc(text=t, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**toks).embeddings  # [1, T, 128]
        embs.append(out[0].cpu().float())
    return embs


def rerank_with_qembs(q_embs, cand_doc_ids, doc_map):
    scores = []
    for doc_id in cand_doc_ids:
        d_emb = doc_map[doc_id]
        # max score across rewrites
        best = None
        for q in q_embs:
            sc = colbert_score(q, d_emb)
            if best is None or sc > best:
                best = sc
        scores.append(best)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-path", required=True, help="parquet with id, query, gti, answer")
    ap.add_argument("--dataset", required=True, choices=["OpenDocVQA", "VDR_ibm"])
    ap.add_argument("--emb-root", default="data/VDR_processed_filtered_2/embeddings/colqwen2")
    ap.add_argument("--rewrites", required=True, help="jsonl rewrites file")
    ap.add_argument("--out-dir", default="data/VDR_processed_filtered_2/Deck_processing")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-pool", type=int, default=500)
    ap.add_argument("--top-rerank", type=int, default=100)
    ap.add_argument("--hard-k", type=int, default=9)
    ap.add_argument("--semi-k", type=int, default=6)
    ap.add_argument("--hard-from-gtif", action="store_true", help="hard negatives from GTI similarity only")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    qa = pd.read_parquet(args.qa_path, columns=["id", "query", "gti", "answer"])
    if args.sample > 0:
        qa = qa.sample(n=min(args.sample, len(qa)), random_state=args.seed)

    emb_root = Path(args.emb_root)
    doc_ids, pooled = load_shards(emb_root, args.dataset)
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    print(f"loaded pooled {len(doc_ids)} vectors for {args.dataset}")

    index = build_hnsw(pooled, m=32, efc=200)
    index.hnsw.efSearch = 200

    # map doc_id -> multi-vector
    doc_map = {}
    for fn in sorted((emb_root / args.dataset).glob("shard_*.pt")):
        obj = torch.load(fn, map_location="cpu")
        for doc_id, emb in zip(obj["doc_ids"], obj["embeddings"]):
            doc_map[doc_id] = emb.float()  # to FP32 for scoring

    rewrites = load_query_rewrites(Path(args.rewrites))
    device = torch.device(args.device)
    model = ColQwen2ForRetrieval.from_pretrained("vidore/colqwen2-v1.0-hf", torch_dtype=torch.bfloat16, device_map=device)
    proc = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-hf")

    out_dir = Path(args.out_dir)
    decks_dir = out_dir / "decks_colqwen_rebuilt" / args.dataset
    decks_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"deck_build_log_{args.dataset}.jsonl"
    stats = []
    gti_rank_mins = []

    for _, row in qa.iterrows():
        qid = row["id"]
        query = row["query"]
        gti_list = row["gti"]
        if isinstance(gti_list, np.ndarray):
            gti_list = gti_list.tolist()
        if not isinstance(gti_list, list):
            gti_list = [gti_list]
        rew = rewrites.get(qid, [])[:7]
        q_embs = query_embeddings(model, proc, query, rew, device)
        # pooled query vector for ANN (mean of token means across rewrites)
        q_pool = torch.stack([q.mean(0) for q in q_embs], dim=0).mean(0).numpy()

        # pooled search per gti
        cand_ids = set()
        for gti in gti_list:
            if gti not in doc_map or gti not in id_to_idx:
                continue
            idx = id_to_idx[gti]
            D, I = index.search(pooled[idx][None, :], args.top_pool)
            for j in I[0]:
                if j < len(doc_ids):
                    cand_ids.add(doc_ids[j])
        # pooled search per query (query-anchored hard negatives)
        if not args.hard_from_gtif:
            Dq, Iq = index.search(q_pool[None, :], args.top_pool)
            for j in Iq[0]:
                if j < len(doc_ids):
                    cand_ids.add(doc_ids[j])
        # remove gti self
        cand_ids = [c for c in cand_ids if c not in gti_list]
        # rerank
        if len(cand_ids) == 0:
            cand_ids = random.sample(doc_ids, k=min(args.top_rerank, len(doc_ids)))
        cand_ids = cand_ids[: args.top_rerank]
        scores = rerank_with_qembs(q_embs, cand_ids, doc_map)
        reranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)

        deck_ids = []
        deck_scores = []
        rank_type = []
        # GTI first
        for g in gti_list:
            if g in doc_map:
                deck_ids.append(g)
                deck_scores.append(None)
                rank_type.append("gti")
        # hard
        for doc_id, sc in reranked:
            if len(deck_ids) >= 20:
                break
            if doc_id in deck_ids:
                continue
            if len([r for r in rank_type if r == "hard"]) < args.hard_k:
                deck_ids.append(doc_id)
                deck_scores.append(sc)
                rank_type.append("hard")
        # semi: pick remaining reranked
        for doc_id, sc in reranked:
            if len(deck_ids) >= 20:
                break
            if doc_id in deck_ids:
                continue
            if len([r for r in rank_type if r == "semi"]) < args.semi_k:
                deck_ids.append(doc_id)
                deck_scores.append(sc)
                rank_type.append("semi")
        # fallback
        idx_pool = 0
        while len(deck_ids) < 20 and idx_pool < len(doc_ids):
            cand = doc_ids[idx_pool]
            idx_pool += 1
            if cand in deck_ids:
                continue
            deck_ids.append(cand)
            deck_scores.append(None)
            rank_type.append("fallback")

        # gti rank stats within deck
        gti_scores = {}
        for g in gti_list:
            if g in doc_map:
                best = None
                for q in q_embs:
                    sc = colbert_score(q, doc_map[g])
                    if best is None or sc > best:
                        best = sc
                gti_scores[g] = best
        # compute ranks in deck
        deck_score_map = {}
        for d, sc in zip(deck_ids, deck_scores):
            if sc is None and d in gti_scores:
                deck_score_map[d] = gti_scores[d]
            elif sc is not None:
                deck_score_map[d] = sc
        sorted_deck = sorted(deck_score_map.items(), key=lambda x: x[1], reverse=True)
        rank_map = {doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted_deck)}
        gti_ranks = [rank_map[g] for g in gti_list if g in rank_map]

        torch.save(
            {
                "qa_id": qid,
                "query": query,
                "gti": gti_list,
                "doc_ids": deck_ids,
                "scores": deck_scores,
                "rank_type": rank_type,
                "source_model": "vidore/colqwen2-v1.0-hf",
                "rewrite_used": rew,
                "gti_ranks": gti_ranks,
            },
            decks_dir / f"{qid}.pt",
        )
        stats.append(len(deck_ids))
        if gti_ranks:
            gti_rank_mins.append(min(gti_ranks))
        with log_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "qa_id": qid,
                        "deck_len": len(deck_ids),
                        "gti": list(gti_list),
                        "hard": [d for d, t in zip(deck_ids, rank_type) if t == "hard"],
                        "semi": [d for d, t in zip(deck_ids, rank_type) if t == "semi"],
                        "gti_ranks": gti_ranks,
                    }
                )
                + "\n"
            )

    stat_path = out_dir / f"deck_stats_{args.dataset}.json"
    with stat_path.open("w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "count": len(stats),
                "deck_len_mean": float(np.mean(stats)),
                "deck_len_min": int(np.min(stats)),
                "deck_len_max": int(np.max(stats)),
                "gti_rank_min_mean": float(np.mean(gti_rank_mins)) if gti_rank_mins else None,
                "gti_rank_min_median": float(np.median(gti_rank_mins)) if gti_rank_mins else None,
                "gti_rank_min_p1": sum(1 for r in gti_rank_mins if r == 1),
                "gti_rank_min_p3": sum(1 for r in gti_rank_mins if r <= 3),
            },
            f,
            indent=2,
        )
    print(f"done {args.dataset} QA={len(stats)}; stats saved to {stat_path}")


if __name__ == "__main__":
    main()
