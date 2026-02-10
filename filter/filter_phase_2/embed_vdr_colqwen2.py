#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
from typing import Iterable, List, Tuple

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed VDR corpus images with ColQwen2 (deck or shard mode)."
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Path to corpus_index parquet (e.g., OpenDocVQA.parquet, SlideVQA.parquet, VDR_ibm.parquet).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name for output folder (e.g., OpenDocVQA, SlideVQA, VDR_ibm).",
    )
    parser.add_argument(
        "--base-dir",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2",
        help="Base dir to resolve relative image paths in the index.",
    )
    parser.add_argument(
        "--output-dir",
        default="/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data/VDR_processed_filtered_2/embeddings/colqwen2",
        help="Base output directory.",
    )
    parser.add_argument("--model", default="vidore/colqwen2-v1.0-hf", help="HF model id.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for image encoding.")
    parser.add_argument(
        "--mode",
        choices=["auto", "deck", "shard"],
        default="auto",
        help="Embedding mode: auto=use deck_name if present else shard.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Number of images per shard file (shard mode).",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Optional comma-separated splits to include if 'split' column exists.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Limit number of units (decks or shards) for testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip decks/shards that already have embedding files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes (for multi-GPU parallelism).",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids to use (e.g., 0,1,2,3). Defaults to 0..num_workers-1.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (default: cuda:0 if available else cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--use-fast-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast image processor (default: True).",
    )
    parser.add_argument(
        "--save-ragged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save embeddings as a list of tensors (ragged lengths, ColQwen-style).",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_images(paths: Iterable[str]) -> List[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def embed_images(
    model,
    processor,
    paths: List[str],
    batch_size: int,
    save_ragged: bool,
):
    import torch

    items: List[torch.Tensor] = []
    for i in range(0, len(paths), batch_size):
        images = load_images(paths[i : i + batch_size])
        inputs = processor(images=images).to(model.device)
        with torch.no_grad():
            embeddings = model(**inputs).embeddings
        embeddings = embeddings.cpu()
        items.extend(list(embeddings.unbind(0)))

    if save_ragged:
        return items

    max_len = max(t.shape[0] for t in items) if items else 0
    if max_len == 0:
        return torch.empty((0, 0, 0))

    dim = items[0].shape[1]
    out = items[0].new_zeros((len(items), max_len, dim))
    for i, t in enumerate(items):
        out[i, : t.shape[0]] = t
    return out


def _parse_gpu_ids(args: argparse.Namespace) -> List[int]:
    if args.gpu_ids:
        return [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if args.num_workers > 1:
        return list(range(args.num_workers))
    return [0]


def _load_df(args: argparse.Namespace):
    pf = pq.ParquetFile(args.index)
    names = set(pf.schema.names)
    cols = ["doc_id", "path"]
    if "split" in names:
        cols.append("split")
    if "deck_name" in names:
        cols.append("deck_name")
    if "page_index" in names:
        cols.append("page_index")

    table = pq.read_table(args.index, columns=cols)
    df = table.to_pandas()
    if args.splits and "split" in df.columns:
        split_set = {s.strip() for s in args.splits.split(",") if s.strip()}
        df = df[df["split"].isin(split_set)]
    df["abs_path"] = df["path"].apply(lambda p: os.path.join(args.base_dir, p))
    return df


def _detect_mode(df, args: argparse.Namespace) -> str:
    if args.mode != "auto":
        return args.mode
    if "deck_name" in df.columns and df["deck_name"].notna().any():
        return "deck"
    return "shard"


def _worker(rank: int, gpu_id: int, args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor
    from transformers.utils.import_utils import is_flash_attn_2_available

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    df = _load_df(args)
    mode = _detect_mode(df, args)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    model = ColQwen2ForRetrieval.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
    ).to(device)
    processor = ColQwen2Processor.from_pretrained(
        args.model, use_fast=args.use_fast_processor
    )

    dataset_out_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    if mode == "deck":
        groups = df.groupby("deck_name", sort=False)
        units = list(groups)
        if args.max_units is not None:
            units = units[: args.max_units]
        units = [u for i, u in enumerate(units) if i % args.num_workers == rank]

        for deck_name, deck_df in tqdm(units, desc=f"Decks (rank {rank})"):
            split = deck_df["split"].iloc[0] if "split" in deck_df.columns else "all"
            out_dir = os.path.join(dataset_out_dir, split)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{deck_name}.pt")
            if args.resume and os.path.exists(out_path):
                continue

            if "page_index" in deck_df.columns:
                deck_df = deck_df.sort_values("page_index")

            doc_ids = deck_df["doc_id"].tolist()
            paths = deck_df["abs_path"].tolist()
            page_index = deck_df["page_index"].tolist() if "page_index" in deck_df.columns else None

            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} images for deck {deck_name}")

            embeddings = embed_images(model, processor, paths, args.batch_size, args.save_ragged)
            if args.save_ragged:
                emb_dtype = str(embeddings[0].dtype) if embeddings else "unknown"
            else:
                emb_dtype = str(embeddings.dtype)

            torch.save(
                {
                    "dataset": args.dataset_name,
                    "deck_name": deck_name,
                    "split": split,
                    "doc_ids": doc_ids,
                    "page_index": page_index,
                    "embeddings": embeddings,
                    "embeddings_format": "list" if args.save_ragged else "tensor",
                    "model": args.model,
                    "dtype": emb_dtype,
                },
                out_path,
            )
        return

    items = list(zip(df["doc_id"].tolist(), df["abs_path"].tolist()))
    if "split" in df.columns:
        splits = df["split"].tolist()
    else:
        splits = ["all"] * len(items)

    shard_indices = list(range(0, len(items), args.shard_size))
    if args.max_units is not None:
        shard_indices = shard_indices[: args.max_units]
    shard_indices = [s for i, s in enumerate(shard_indices) if i % args.num_workers == rank]

    for shard_idx in tqdm(shard_indices, desc=f"Shards (rank {rank})"):
        shard_items = items[shard_idx : shard_idx + args.shard_size]
        shard_doc_ids = [x[0] for x in shard_items]
        shard_paths = [x[1] for x in shard_items]
        shard_splits = splits[shard_idx : shard_idx + args.shard_size]

        out_path = os.path.join(dataset_out_dir, f"shard_{shard_idx:07d}.pt")
        if args.resume and os.path.exists(out_path):
            continue

        missing = [p for p in shard_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images in shard {shard_idx}")

        embeddings = embed_images(model, processor, shard_paths, args.batch_size, args.save_ragged)
        if args.save_ragged:
            emb_dtype = str(embeddings[0].dtype) if embeddings else "unknown"
        else:
            emb_dtype = str(embeddings.dtype)

        torch.save(
            {
                "dataset": args.dataset_name,
                "shard_start": shard_idx,
                "doc_ids": shard_doc_ids,
                "paths": shard_paths,
                "splits": shard_splits,
                "embeddings": embeddings,
                "embeddings_format": "list" if args.save_ragged else "tensor",
                "model": args.model,
                "dtype": emb_dtype,
            },
            out_path,
        )


def main() -> None:
    args = parse_args()
    if args.num_workers <= 1:
        gpu_id = _parse_gpu_ids(args)[0]
        _worker(0, gpu_id, args)
        return

    gpu_ids = _parse_gpu_ids(args)
    if len(gpu_ids) < args.num_workers:
        raise ValueError("gpu-ids count must be >= num-workers")

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(args.num_workers):
        p = ctx.Process(target=_worker, args=(rank, gpu_ids[rank], args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"Worker exited with code {p.exitcode}")


if __name__ == "__main__":
    main()
