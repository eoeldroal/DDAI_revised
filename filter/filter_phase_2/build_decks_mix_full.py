#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import ColQwen2ForRetrieval, ColQwen2Processor


@dataclass
class Setting:
    name: str
    top_pool: int
    top_rerank: int
    deep_rerank_cap: int
    hard_k: int
    semi_k: int
    hard_from_gtif: bool


SETTING2 = Setting(
    "setting2", top_pool=2048, top_rerank=1000, deep_rerank_cap=256, hard_k=18, semi_k=1, hard_from_gtif=True
)
SETTING1 = Setting(
    "setting1", top_pool=2000, top_rerank=500, deep_rerank_cap=192, hard_k=15, semi_k=2, hard_from_gtif=False
)


def shard_ok(qid: str, rank: int, world: int) -> bool:
    h = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16)
    return (h % world) == rank


def load_query_rewrites(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = rec.get("id") or rec.get("qa_id") or rec.get("dataset_id")
            if not qid:
                continue
            rewrites: List[str] = []
            if isinstance(rec.get("rewrites"), list):
                rewrites = [str(x).strip() for x in rec["rewrites"] if isinstance(x, str) and x.strip()]
            elif isinstance(rec.get("runs"), list):
                for r in rec["runs"]:
                    if isinstance(r, str) and r.strip():
                        rewrites.append(r.strip())
                    elif isinstance(r, dict):
                        t = r.get("text") or r.get("rewrite") or r.get("query")
                        if t:
                            rewrites.append(str(t).strip())
            if rewrites:
                out[str(qid)] = rewrites
    return out


def cache_paths(emb_root: Path, dataset: str) -> Dict[str, Path]:
    cdir = emb_root / dataset / "_deck_cache"
    return {
        "dir": cdir,
        "pooled": cdir / "pooled_f32.npy",
        "shard_ord": cdir / "shard_ord_i32.npy",
        "local_idx": cdir / "local_idx_i32.npy",
        "doc_ids": cdir / "doc_ids.txt",
        "shards": cdir / "shards.json",
    }


def index_cache_path(cache_dir: Path, args: argparse.Namespace) -> Path:
    if args.index == "ivf":
        name = f"faiss_ivf_flat_ip_nlist{args.nlist}_train{args.ivf_train_size}.index"
    else:
        name = f"faiss_hnsw_flat_ip_m{args.hnsw_m}_efc{args.hnsw_efc}.index"
    return cache_dir / name


def ensure_dataset_cache(emb_root: Path, dataset: str, log_shard_every: int = 16, force: bool = False) -> None:
    p = cache_paths(emb_root, dataset)
    required = [p["pooled"], p["shard_ord"], p["local_idx"], p["doc_ids"], p["shards"]]
    if (not force) and all(x.exists() for x in required):
        print(f"[cache] dataset={dataset} already exists: {p['dir']}", flush=True)
        return

    p["dir"].mkdir(parents=True, exist_ok=True)

    shards = sorted((emb_root / dataset).glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {(emb_root / dataset)}")

    doc_ids: List[str] = []
    pooled_list: List[np.ndarray] = []
    shard_ord: List[int] = []
    local_idx: List[int] = []

    t0 = time.time()
    for s_ord, fn in enumerate(shards):
        obj = torch.load(fn, map_location="cpu")
        ids = [str(x) for x in obj["doc_ids"]]
        embs = obj["embeddings"]
        for li, (doc_id, emb) in enumerate(zip(ids, embs)):
            e = emb if torch.is_tensor(emb) else torch.tensor(emb)
            doc_ids.append(doc_id)
            pooled_list.append(e.mean(0).float().numpy())
            shard_ord.append(s_ord)
            local_idx.append(li)

        if (s_ord + 1) % log_shard_every == 0 or (s_ord + 1) == len(shards):
            print(
                f"[cache] dataset={dataset} shard={s_ord+1}/{len(shards)} docs={len(doc_ids)} elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    pooled = np.ascontiguousarray(np.vstack(pooled_list).astype(np.float32))
    shard_ord_np = np.asarray(shard_ord, dtype=np.int32)
    local_idx_np = np.asarray(local_idx, dtype=np.int32)

    np.save(p["pooled"], pooled)
    np.save(p["shard_ord"], shard_ord_np)
    np.save(p["local_idx"], local_idx_np)
    with p["doc_ids"].open("w", encoding="utf-8") as f:
        for d in doc_ids:
            f.write(d + "\n")
    with p["shards"].open("w", encoding="utf-8") as f:
        json.dump([str(x) for x in shards], f)

    print(
        f"[cache] dataset={dataset} built docs={len(doc_ids)} shape={pooled.shape} at={p['dir']}",
        flush=True,
    )


def load_dataset_cache(emb_root: Path, dataset: str):
    p = cache_paths(emb_root, dataset)
    with p["doc_ids"].open("r", encoding="utf-8") as f:
        doc_ids = [line.rstrip("\n") for line in f]
    with p["shards"].open("r", encoding="utf-8") as f:
        shard_files = json.load(f)

    pooled = np.load(p["pooled"], mmap_mode="r")
    shard_ord = np.load(p["shard_ord"], mmap_mode="r")
    local_idx = np.load(p["local_idx"], mmap_mode="r")

    id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    return doc_ids, pooled, shard_ord, local_idx, shard_files, id_to_idx


class ShardStore:
    def __init__(self, shard_files: List[str], max_cache_shards: int = 8):
        self.shard_files = shard_files
        self.max_cache_shards = max_cache_shards
        self.cache: "OrderedDict[int, List[torch.Tensor]]" = OrderedDict()

    def _load_shard(self, s_ord: int) -> List[torch.Tensor]:
        obj = torch.load(self.shard_files[s_ord], map_location="cpu", weights_only=False)
        return obj["embeddings"]

    def get_emb(self, s_ord: int, local_idx: int) -> torch.Tensor:
        if s_ord not in self.cache:
            emb_list = self._load_shard(s_ord)
            self.cache[s_ord] = emb_list
            self.cache.move_to_end(s_ord)
            while len(self.cache) > self.max_cache_shards:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(s_ord)
        return self.cache[s_ord][local_idx]


def build_faiss_index(
    pooled: np.ndarray,
    index_type: str,
    nlist: int,
    nprobe: int,
    hnsw_m: int,
    hnsw_efc: int,
    hnsw_efs: int,
    train_size: int,
) -> Tuple[faiss.Index, bool]:
    x = np.array(pooled, dtype=np.float32, copy=True)
    faiss.normalize_L2(x)

    dim = x.shape[1]
    if index_type == "ivf":
        quant = faiss.IndexFlatIP(dim)
        index_cpu = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        if not index_cpu.is_trained:
            if len(x) <= train_size:
                train_x = x
            else:
                sel = np.random.choice(len(x), size=train_size, replace=False)
                train_x = x[sel]
            index_cpu.train(train_x)

        index_cpu.add(x)
        index_cpu.nprobe = nprobe

        return index_cpu, True

    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_efc
        index.hnsw.efSearch = hnsw_efs
        index.add(x)
        return index, True

    raise ValueError(f"Unknown index_type: {index_type}")


def ensure_faiss_index_cache(
    pooled: np.ndarray,
    idx_path: Path,
    args: argparse.Namespace,
) -> None:
    if idx_path.exists() and not args.rebuild_index:
        print(f"[index] reuse dataset index: {idx_path}", flush=True)
        return

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    index_cpu, _ = build_faiss_index(
        pooled=pooled,
        index_type=args.index,
        nlist=args.nlist,
        nprobe=args.nprobe,
        hnsw_m=args.hnsw_m,
        hnsw_efc=args.hnsw_efc,
        hnsw_efs=args.hnsw_efs,
        train_size=args.ivf_train_size,
    )
    faiss.write_index(index_cpu, str(idx_path))
    print(
        f"[index] built and saved: {idx_path} ntotal={index_cpu.ntotal} elapsed={time.time()-t0:.1f}s",
        flush=True,
    )


def wait_for_file(path: Path, timeout_sec: int, poll_sec: float = 2.0) -> bool:
    t0 = time.time()
    while (time.time() - t0) < timeout_sec:
        if path.exists():
            return True
        time.sleep(poll_sec)
    return path.exists()


def pass_threshold(best_rank: int | None, median_rank: int | None, args: argparse.Namespace) -> bool:
    if best_rank is None or median_rank is None:
        return False
    return best_rank <= args.best_rank_threshold and median_rank <= args.median_rank_threshold


def search_index(index: faiss.Index, vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    q = vec.astype(np.float32, copy=True).reshape(1, -1)
    faiss.normalize_L2(q)
    k = max(1, int(topk))
    ntotal = int(getattr(index, "ntotal", 0))
    if ntotal > 0:
        k = min(k, ntotal)
    d, i = index.search(q, k)
    return d[0], i[0]


@torch.no_grad()
def colbert_score_tensor(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    dot = torch.matmul(q, d.t())
    return torch.max(dot, dim=1).values.sum()


@torch.no_grad()
def query_embeddings(model, processor, query: str, rewrites: List[str], device: torch.device) -> List[torch.Tensor]:
    texts = [query] + rewrites
    toks = processor(text=texts, return_tensors="pt", padding=True).to(device)
    out = model(**toks).embeddings
    # Keep query embeddings on GPU for rerank speed.
    return [out[i].detach().to(dtype=torch.bfloat16) for i in range(out.shape[0])]


def rerank_candidates(
    q_embs: List[torch.Tensor],
    cand_idx: List[int],
    shard_store: ShardStore,
    shard_ord_arr: np.ndarray,
    local_idx_arr: np.ndarray,
    device: torch.device,
    limit: int,
    batch_size: int,
) -> List[Tuple[int, float]]:
    if limit > 0 and len(cand_idx) > limit:
        cand_idx = cand_idx[:limit]

    scored: List[Tuple[int, float]] = []
    q_gpu = [q.to(device=device, dtype=torch.bfloat16, non_blocking=True) for q in q_embs]
    if batch_size <= 0:
        batch_size = 16

    for off in range(0, len(cand_idx), batch_size):
        batch_idx = cand_idx[off : off + batch_size]
        docs_cpu: List[torch.Tensor] = []
        lengths: List[int] = []
        for idx in batch_idx:
            s_ord = int(shard_ord_arr[idx])
            l_idx = int(local_idx_arr[idx])
            d = shard_store.get_emb(s_ord, l_idx).to(dtype=torch.bfloat16)
            docs_cpu.append(d)
            lengths.append(int(d.shape[0]))

        max_len = max(lengths)
        emb_dim = int(docs_cpu[0].shape[1])
        docs_pad = torch.zeros((len(batch_idx), max_len, emb_dim), dtype=torch.bfloat16)
        for i, d in enumerate(docs_cpu):
            docs_pad[i, : d.shape[0], :] = d
        docs_gpu = docs_pad.to(device=device, non_blocking=True)
        len_t = torch.tensor(lengths, device=device, dtype=torch.long)
        pos = torch.arange(max_len, device=device).view(1, 1, -1)
        valid_mask = pos < len_t.view(-1, 1, 1)

        per_query_scores: List[torch.Tensor] = []
        for q in q_gpu:
            # [B, Tq, Td] = [Tq,D] x [B,Td,D]
            dots = torch.einsum("qd,bkd->bqk", q, docs_gpu)
            dots = dots.masked_fill(~valid_mask, -1e4)
            score_q = dots.max(dim=2).values.sum(dim=1)  # [B]
            per_query_scores.append(score_q)

        best_scores = torch.stack(per_query_scores, dim=0).max(dim=0).values.float().cpu().tolist()
        scored.extend(zip(batch_idx, best_scores))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def compute_best_median_rank(
    q_embs: List[torch.Tensor],
    deck_idx: List[int],
    gti_idx: set,
    shard_store: ShardStore,
    shard_ord_arr: np.ndarray,
    local_idx_arr: np.ndarray,
    device: torch.device,
) -> Tuple[int | None, int | None]:
    ranks = []
    deck_embs: Dict[int, torch.Tensor] = {}
    for idx in deck_idx:
        s_ord = int(shard_ord_arr[idx])
        l_idx = int(local_idx_arr[idx])
        deck_embs[idx] = shard_store.get_emb(s_ord, l_idx).to(device=device, dtype=torch.bfloat16, non_blocking=True)

    q_gpu = [q.to(device=device, dtype=torch.bfloat16, non_blocking=True) for q in q_embs]
    for q in q_gpu:
        valid_idx = [idx for idx in deck_idx if idx in deck_embs]
        score_vec = torch.stack([colbert_score_tensor(q, deck_embs[idx]) for idx in valid_idx]).float()
        order = torch.argsort(score_vec, descending=True).cpu().tolist()
        rank_map = {valid_idx[o]: r + 1 for r, o in enumerate(order)}
        gti_r = [rank_map[g] for g in gti_idx if g in rank_map]
        if gti_r:
            ranks.append(min(gti_r))

    if not ranks:
        return None, None
    return min(ranks), int(statistics.median(ranks))


def build_deck_for_row(
    gti_raw,
    q_embs: List[torch.Tensor],
    setting: Setting,
    doc_ids: List[str],
    pooled: np.ndarray,
    id_to_idx: Dict[str, int],
    index: faiss.Index,
    shard_store: ShardStore,
    shard_ord_arr: np.ndarray,
    local_idx_arr: np.ndarray,
    device: torch.device,
    rerank_batch_size: int,
) -> Tuple[List[str], List[float], List[str], List[int], List[str]]:
    if isinstance(gti_raw, np.ndarray):
        gti_raw = gti_raw.tolist()
    if not isinstance(gti_raw, list):
        gti_raw = [gti_raw]
    gti_docs = [str(x) for x in gti_raw]

    gti_idx = [id_to_idx[g] for g in gti_docs if g in id_to_idx]

    q_pool = torch.stack([q.float().mean(0) for q in q_embs], dim=0).mean(0).cpu().numpy()

    cand_ann: Dict[int, float] = {}

    for gidx in gti_idx:
        d, ind = search_index(index, pooled[gidx], setting.top_pool)
        for score, j in zip(d, ind):
            if 0 <= int(j) < len(doc_ids):
                jj = int(j)
                if jj in gti_idx:
                    continue
                prev = cand_ann.get(jj)
                if prev is None or float(score) > prev:
                    cand_ann[jj] = float(score)

    if not setting.hard_from_gtif:
        d, ind = search_index(index, q_pool, setting.top_pool)
        for score, j in zip(d, ind):
            if 0 <= int(j) < len(doc_ids):
                jj = int(j)
                if jj in gti_idx:
                    continue
                prev = cand_ann.get(jj)
                if prev is None or float(score) > prev:
                    cand_ann[jj] = float(score)

    if not cand_ann:
        cand_idx = random.sample(range(len(doc_ids)), k=min(setting.top_rerank, len(doc_ids)))
    else:
        cand_idx = [idx for idx, _ in sorted(cand_ann.items(), key=lambda x: x[1], reverse=True)[: setting.top_rerank]]

    reranked = rerank_candidates(
        q_embs=q_embs,
        cand_idx=cand_idx,
        shard_store=shard_store,
        shard_ord_arr=shard_ord_arr,
        local_idx_arr=local_idx_arr,
        device=device,
        limit=setting.deep_rerank_cap,
        batch_size=rerank_batch_size,
    )

    deck_idx: List[int] = []
    deck_scores: List[float] = []
    deck_types: List[str] = []

    for gi in gti_idx:
        if gi not in deck_idx:
            deck_idx.append(gi)
            deck_scores.append(float("inf"))
            deck_types.append("gti")

    for idx, sc in reranked:
        if len(deck_idx) >= 20:
            break
        if idx in deck_idx:
            continue
        deck_idx.append(idx)
        deck_scores.append(float(sc))
        deck_types.append("hard")
        if sum(1 for t in deck_types if t == "hard") >= setting.hard_k:
            break

    if setting.semi_k > 0 and len(deck_idx) < 20:
        tail = [idx for idx, _ in reranked[setting.hard_k : setting.hard_k + 200]]
        random.shuffle(tail)
        for idx in tail:
            if idx in deck_idx:
                continue
            deck_idx.append(idx)
            deck_scores.append(0.0)
            deck_types.append("semi")
            if sum(1 for t in deck_types if t == "semi") >= setting.semi_k or len(deck_idx) >= 20:
                break

    if len(deck_idx) < 20:
        for idx in range(len(doc_ids)):
            if idx in deck_idx:
                continue
            deck_idx.append(idx)
            deck_scores.append(0.0)
            deck_types.append("fallback")
            if len(deck_idx) >= 20:
                break

    deck_idx = deck_idx[:20]
    deck_docs = [doc_ids[i] for i in deck_idx]
    return deck_docs, deck_scores[:20], deck_types[:20], deck_idx, gti_docs


def evaluate_setting(
    gti_raw,
    q_embs: List[torch.Tensor],
    setting: Setting,
    doc_ids: List[str],
    pooled: np.ndarray,
    id_to_idx: Dict[str, int],
    index: faiss.Index,
    shard_store: ShardStore,
    shard_ord_arr: np.ndarray,
    local_idx_arr: np.ndarray,
    device: torch.device,
    rerank_batch_size: int,
) -> Tuple[List[str], List[float], List[str], List[int], List[str], int | None, int | None]:
    deck_docs, deck_scores, deck_types, deck_idx, gti_docs = build_deck_for_row(
        gti_raw=gti_raw,
        q_embs=q_embs,
        setting=setting,
        doc_ids=doc_ids,
        pooled=pooled,
        id_to_idx=id_to_idx,
        index=index,
        shard_store=shard_store,
        shard_ord_arr=shard_ord_arr,
        local_idx_arr=local_idx_arr,
        device=device,
        rerank_batch_size=rerank_batch_size,
    )
    gti_idx_set = {id_to_idx[g] for g in gti_docs if g in id_to_idx}
    best_rank, median_rank = compute_best_median_rank(
        q_embs=q_embs,
        deck_idx=deck_idx,
        gti_idx=gti_idx_set,
        shard_store=shard_store,
        shard_ord_arr=shard_ord_arr,
        local_idx_arr=local_idx_arr,
        device=device,
    )
    return deck_docs, deck_scores, deck_types, deck_idx, gti_docs, best_rank, median_rank


def process_dataset(args, dataset: str) -> None:
    qa_file = "opendocvqa_train.parquet" if dataset == "OpenDocVQA" else "vdr_ibm_train.parquet"
    qa_path = Path(args.qa_root) / dataset / qa_file
    qa = pd.read_parquet(qa_path, columns=["id", "query", "gti"])
    if args.max_rows > 0:
        qa = qa.head(args.max_rows)

    out_root = Path(args.out_root)
    decks_dir = out_root / "decks" / dataset
    logs_dir = out_root / "logs" / dataset
    stats_dir = out_root / "stats" / dataset
    decks_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[start] dataset={dataset} rank={args.rank}/{args.world_size} qa_total={len(qa)} resume={args.resume}",
        flush=True,
    )
    print(
        f"[config] dataset={dataset} s2(top_pool={SETTING2.top_pool},ann={SETTING2.top_rerank},deep={SETTING2.deep_rerank_cap}) "
        f"s1(top_pool={SETTING1.top_pool},ann={SETTING1.top_rerank},deep={SETTING1.deep_rerank_cap})",
        flush=True,
    )

    emb_root = Path(args.emb_root)
    print(f"[phase] dataset={dataset} cache-check start", flush=True)
    ensure_dataset_cache(emb_root, dataset, log_shard_every=args.log_shard_every, force=args.rebuild_cache)

    doc_ids, pooled, shard_ord_arr, local_idx_arr, shard_files, id_to_idx = load_dataset_cache(emb_root, dataset)
    print(f"[cache] dataset={dataset} loaded docs={len(doc_ids)} pooled_shape={pooled.shape}", flush=True)

    cdir = cache_paths(emb_root, dataset)["dir"]
    idx_path = index_cache_path(cdir, args)
    if args.prepare_cache_only:
        ensure_faiss_index_cache(pooled, idx_path, args)
        print(f"[done] dataset={dataset} prepare-cache-only", flush=True)
        return

    if not idx_path.exists():
        if args.rank == 0:
            print(f"[index] missing, rank0 building dataset={dataset}", flush=True)
            ensure_faiss_index_cache(pooled, idx_path, args)
        else:
            print(f"[index] waiting for rank0 index build dataset={dataset}", flush=True)
            ok = wait_for_file(idx_path, timeout_sec=args.index_wait_timeout)
            if not ok:
                raise TimeoutError(f"Timed out waiting for index file: {idx_path}")

    print(f"[phase] dataset={dataset} loading index from {idx_path}", flush=True)
    index_cpu = faiss.read_index(str(idx_path))
    if args.index == "ivf":
        index_cpu.nprobe = args.nprobe
    elif args.index == "hnsw":
        index_cpu.hnsw.efSearch = args.hnsw_efs

    gpu_id = 0
    if args.device.startswith("cuda") and ":" in args.device:
        gpu_id = int(args.device.split(":", 1)[1])

    faiss_gpu_ok = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
    use_faiss_gpu = args.faiss_gpu and args.device.startswith("cuda") and args.index == "ivf" and faiss_gpu_ok
    if use_faiss_gpu:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, index_cpu)
        print(f"[ready] dataset={dataset} index={args.index} loaded_on_gpu={gpu_id}", flush=True)
    else:
        index = index_cpu
        if args.faiss_gpu and args.index == "ivf" and not faiss_gpu_ok:
            print(
                "[ready] dataset={} index={} loaded_on_cpu (faiss-gpu bindings not available in this env)".format(
                    dataset, args.index
                ),
                flush=True,
            )
        else:
            print(f"[ready] dataset={dataset} index={args.index} loaded_on_cpu", flush=True)

    rew_path = Path(args.rewrite_root) / f"{dataset}_20260125_135340.jsonl"
    rewrites = load_query_rewrites(rew_path)
    print(f"[load] rewrites dataset={dataset} count={len(rewrites)}", flush=True)

    shard_store = ShardStore(shard_files=shard_files, max_cache_shards=args.max_cache_shards)

    device = torch.device(args.device)
    model = ColQwen2ForRetrieval.from_pretrained(
        "vidore/colqwen2-v1.0-hf", torch_dtype=torch.bfloat16, device_map=device
    )
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-hf")
    print(f"[ready] dataset={dataset} model loaded device={args.device}", flush=True)

    log_s2 = logs_dir / f"deck_build_setting2_rank{args.rank}.jsonl"
    log_s1 = logs_dir / f"deck_build_setting1_rank{args.rank}.jsonl"
    log_dec = logs_dir / f"decisions_rank{args.rank}.jsonl"
    log_fail = logs_dir / f"failures_rank{args.rank}.jsonl"

    stat_rows = []
    shard_total = 0
    shard_done = 0
    shard_skip_resume = 0
    for row in qa.itertuples(index=False):
        if shard_ok(str(row.id), args.rank, args.world_size):
            shard_total += 1

    # Line-buffered log streams so external monitoring (wc/tail) reflects progress promptly.
    with (
        log_s2.open("a", encoding="utf-8", buffering=1) as fh_s2,
        log_s1.open("a", encoding="utf-8", buffering=1) as fh_s1,
        log_dec.open("a", encoding="utf-8", buffering=1) as fh_dec,
        log_fail.open("a", encoding="utf-8", buffering=1) as fh_fail,
    ):
        t0 = time.time()
        for row in qa.itertuples(index=False):
            qid = str(row.id)
            if not shard_ok(qid, args.rank, args.world_size):
                continue

            deck_path = decks_dir / f"{qid}.pt"
            if args.resume and deck_path.exists():
                shard_skip_resume += 1
                continue

            rew = rewrites.get(qid, [])[:7]
            t_row = time.time()
            q_embs = query_embeddings(model, processor, str(row.query), rew, device)
            t_q = time.time()

            deck_docs, deck_scores, deck_types, _deck_idx, gti_docs, best_rank, median_rank = evaluate_setting(
                gti_raw=row.gti,
                q_embs=q_embs,
                setting=SETTING2,
                doc_ids=doc_ids,
                pooled=pooled,
                id_to_idx=id_to_idx,
                index=index,
                shard_store=shard_store,
                shard_ord_arr=shard_ord_arr,
                local_idx_arr=local_idx_arr,
                device=device,
                rerank_batch_size=args.rerank_batch_size,
            )
            t_s2 = time.time()
            fh_s2.write(json.dumps({"qa_id": qid, "best_rank": best_rank, "median_rank": median_rank}) + "\n")

            decision = "setting2_keep"
            if not pass_threshold(best_rank, median_rank, args):
                deck_docs, deck_scores, deck_types, _deck_idx, gti_docs, best_rank, median_rank = evaluate_setting(
                    gti_raw=row.gti,
                    q_embs=q_embs,
                    setting=SETTING1,
                    doc_ids=doc_ids,
                    pooled=pooled,
                    id_to_idx=id_to_idx,
                    index=index,
                    shard_store=shard_store,
                    shard_ord_arr=shard_ord_arr,
                    local_idx_arr=local_idx_arr,
                    device=device,
                    rerank_batch_size=args.rerank_batch_size,
                )
                fh_s1.write(json.dumps({"qa_id": qid, "best_rank": best_rank, "median_rank": median_rank}) + "\n")
                t_s1 = time.time()

                if not pass_threshold(best_rank, median_rank, args):
                    decision = "removed"
                    if deck_path.exists():
                        deck_path.unlink()
                    fh_fail.write(
                        json.dumps(
                            {
                                "qa_id": qid,
                                "best_rank": best_rank,
                                "median_rank": median_rank,
                                "reason": "failed_after_setting1",
                            }
                        )
                        + "\n"
                    )
                else:
                    decision = "setting1_keep"
            else:
                t_s1 = t_s2

            fh_dec.write(json.dumps({"qa_id": qid, "decision": decision}) + "\n")

            if decision != "removed":
                torch.save(
                    {
                        "qa_id": qid,
                        "doc_ids": deck_docs,
                        "scores": deck_scores,
                        "rank_type": deck_types,
                        "gti": gti_docs,
                        "best_rank": best_rank,
                        "median_rank": median_rank,
                    },
                    deck_path,
                )

            stat_rows.append({"qa_id": qid, "decision": decision, "best_rank": best_rank, "median_rank": median_rank})
            shard_done += 1
            if shard_done == 1:
                print(
                    f"[first] dataset={dataset} rank={args.rank} qid={qid} decision={decision} "
                    f"best={best_rank} median={median_rank} "
                    f"t_query={t_q-t_row:.2f}s t_s2={t_s2-t_q:.2f}s t_s1={t_s1-t_s2:.2f}s t_row={t_s1-t_row:.2f}s",
                    flush=True,
                )

            if shard_done % args.log_every == 0:
                elapsed = time.time() - t0
                rate = shard_done / max(elapsed, 1e-6)
                print(
                    f"[progress] dataset={dataset} rank={args.rank} done={shard_done}/{shard_total} "
                    f"skip_resume={shard_skip_resume} rate={rate:.3f} qa/s",
                    flush=True,
                )

    if stat_rows:
        pd.DataFrame(stat_rows).to_parquet(stats_dir / f"rank_stats_rank{args.rank}.parquet", index=False)

    print(
        f"[done] dataset={dataset} rank={args.rank} processed={shard_done}/{shard_total} skip_resume={shard_skip_resume}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="OpenDocVQA,VDR_ibm")
    ap.add_argument("--qa-root", default="data/VDR_processed_filtered_2/qa_gti_filtered_2plus")
    ap.add_argument("--emb-root", default="data/VDR_processed_filtered_2/embeddings/colqwen2")
    ap.add_argument("--rewrite-root", default="data/VDR_processed_filtered_2/filters/query_rewrites")
    ap.add_argument("--out-root", default="data/VDR_processed_filtered_2/Deck_processing/exp_setting2_mix_full")

    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)

    ap.add_argument("--prepare-cache-only", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--index-wait-timeout", type=int, default=3600)
    ap.add_argument("--max-cache-shards", type=int, default=8)

    ap.add_argument("--index", choices=["ivf", "hnsw"], default="ivf")
    ap.add_argument("--faiss-gpu", action="store_true")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=256)
    ap.add_argument("--ivf-train-size", type=int, default=50000)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-efc", type=int, default=200)
    ap.add_argument("--hnsw-efs", type=int, default=200)

    ap.add_argument("--best-rank-threshold", type=int, default=5)
    ap.add_argument("--median-rank-threshold", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--log-shard-every", type=int, default=16)
    ap.add_argument("--rerank-batch-size", type=int, default=16)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        if ds not in ("OpenDocVQA", "VDR_ibm"):
            raise SystemExit(f"Unsupported dataset: {ds}")
        process_dataset(args, ds)


if __name__ == "__main__":
    main()
