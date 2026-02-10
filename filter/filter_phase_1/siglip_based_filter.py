#!/usr/bin/env python3
"""SigLIP2 HF-native image embedding extraction (simple, GPU-sharded).

Flow:
1) Load model/processor (transformers-native)
2) Batch image embeddings
3) Save pooled embeddings for reuse
"""
from __future__ import annotations

import argparse
import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

try:
    from transformers import Siglip2Model, Siglip2Processor
except Exception:  # pragma: no cover - optional import
    Siglip2Model = None
    Siglip2Processor = None


def load_index(root: Path, dataset: str) -> list[tuple[str, Path]]:
    import pandas as pd

    idx_path = root / f"corpus_index/{dataset}.parquet"
    df = pd.read_parquet(idx_path, columns=["doc_id", "path"])
    items = []
    for doc_id, rel in zip(df["doc_id"].astype(str), df["path"].astype(str)):
        img_path = root / rel
        if img_path.exists():
            items.append((doc_id, img_path))
    return items


def run_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(args.model)
    use_siglip2 = cfg.model_type == "siglip2"
    if use_siglip2 and Siglip2Model is not None and Siglip2Processor is not None:
        model = Siglip2Model.from_pretrained(
            args.model,
            dtype=args.dtype,
            device_map=None,
            attn_implementation=args.attn_impl,
        ).eval()
        processor = Siglip2Processor.from_pretrained(args.model, use_fast=args.use_fast)
    else:
        model = AutoModel.from_pretrained(
            args.model,
            dtype=args.dtype,
            device_map=None,
            attn_implementation=args.attn_impl,
        ).eval()
        processor = AutoProcessor.from_pretrained(args.model, use_fast=args.use_fast)
    model = model.to(device, dtype=args.dtype)
    print(
        f"[gpu{rank}] model={args.model} model_type={cfg.model_type} "
        f"use_fast={args.use_fast} device={device}",
        flush=True,
    )

    root = Path(args.root)
    out_dir = root / args.out_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_index(root, args.dataset)
    # shard by rank
    items = items[rank::world_size]
    total = len(items)

    doc_ids: list[str] = []
    pooled_list: list[torch.Tensor] = []

    batch: list[Image.Image] = []
    batch_ids: list[str] = []
    start = time.time()
    batches = 0
    images_done = 0

    def flush() -> None:
        nonlocal batches, images_done
        if not batch:
            return
        if os.environ.get("PDB_FLUSH") == "1":
            import pdb
            pdb.set_trace()
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            if hasattr(model, "get_image_features"):
                pooled = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                if getattr(outputs, "pooler_output", None) is not None:
                    pooled = outputs.pooler_output
                elif getattr(outputs, "image_embeds", None) is not None:
                    pooled = outputs.image_embeds
                elif getattr(outputs, "last_hidden_state", None) is not None:
                    pooled = outputs.last_hidden_state.mean(dim=1)
                else:
                    raise RuntimeError("Unsupported model output for image embeddings.")
        pooled = F.normalize(pooled, p=2, dim=1).to("cpu")
        pooled_list.append(pooled)
        doc_ids.extend(batch_ids)
        batch.clear()
        batch_ids.clear()
        batches += 1
        images_done += len(pooled)
        if batches % args.log_every == 0:
            elapsed = time.time() - start
            rate = images_done / elapsed if elapsed > 0 else 0.0
            pct = (images_done / total * 100.0) if total > 0 else 0.0
            print(
                f"[gpu{rank}] batches={batches} images~={images_done}/{total} "
                f"({pct:.1f}%) rate={rate:.1f} img/s elapsed={elapsed:.1f}s",
                flush=True,
            )

    cv2 = None
    if args.decoder == "opencv":
        try:
            import cv2 as _cv2  # type: ignore
            cv2 = _cv2
        except Exception as exc:
            raise RuntimeError(f"OpenCV requested but not available: {exc}") from exc

    def decode_image(img_path: Path) -> Image.Image:
        if args.decoder == "opencv":
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imread failed")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        img = Image.open(img_path).convert("RGB")
        return img

    def load_one(pair: tuple[str, Path]) -> tuple[str, Image.Image] | None:
        doc_id, img_path = pair
        try:
            img = decode_image(img_path)
            return doc_id, img
        except Exception as exc:
            if args.log_decode_errors and rank == 0 and load_one.failures < args.log_decode_errors:
                print(f"[warn] decode failed: {img_path} ({exc})", flush=True)
            load_one.failures += 1
            return None
    load_one.failures = 0

    def iter_prefetch(items: list[tuple[str, Path]]):
        q: queue.Queue = queue.Queue(maxsize=max(1, args.prefetch))

        def producer():
            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                for result in ex.map(load_one, items, chunksize=args.num_workers):
                    q.put(result)
            q.put(None)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    for result in iter_prefetch(items):
        if result is None:
            continue
        doc_id, img = result
        batch.append(img)
        batch_ids.append(doc_id)
        if len(batch) >= args.batch_size:
            flush()

    flush()

    # Save pooled embeddings and doc_ids for reuse
    pooled_path = out_dir / f"pooled_rank{rank:02d}.pt"
    ids_path = out_dir / f"doc_ids_rank{rank:02d}.txt"
    torch.save(pooled_list, pooled_path)
    ids_path.write_text("\n".join(doc_ids))
    print(f"[gpu{rank}] saved pooled embeddings: {pooled_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root")
    parser.add_argument("--dataset", choices=["OpenDocVQA", "SlideVQA", "VDR_ibm", "all"], required=True)
    parser.add_argument("--model", default="google/siglip2-giant-opt-patch16-384")
    parser.add_argument("--out-dir", default="filters/siglip_embeddings")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--use-fast", action="store_true", default=True)
    parser.add_argument("--no-fast", dest="use_fast", action="store_false")
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--decoder", default="opencv", choices=["opencv", "pil"])
    parser.add_argument("--prefetch", type=int, default=512)
    parser.add_argument("--log-decode-errors", type=int, default=5)
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    args.dtype = dtype_map[args.dtype]

    datasets = ["OpenDocVQA", "SlideVQA", "VDR_ibm"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        args.dataset = dataset
        if args.gpus > 1:
            mp.spawn(run_worker, args=(args.gpus, args), nprocs=args.gpus, join=True)
        else:
            run_worker(0, 1, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
