#!/usr/bin/env python3
"""Quick DeepSeek-OCR sanity test on a few example images."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModel, AutoTokenizer


def pick_default_images() -> list[Path]:
    base = Path("/opt/dlami/nvme/isdslab/HyunBin/DDAI_Revised/verl/data")
    candidates = [
        base / "VDR_processed_filtered_1/filters/docvqa_lowres_512/corpus/OpenDocVQA/images/docvqa/yjxf0227_6.png",
        base / "VDR_processed_filtered_1/filters/docvqa_lowres_512/corpus/OpenDocVQA/images/docvqa/qyll0226_1.png",
        base / "VDR_processed_filtered_1/corpus/OpenDocVQA/images/docvqa/pmgl0226_1.png",
    ]
    return [p for p in candidates if p.exists()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown. ")
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", action="store_true", default=True)
    parser.add_argument("--no-crop", dest="crop_mode", action="store_false")
    parser.add_argument("--save-results", action="store_true", default=True)
    parser.add_argument("--output-dir", default="/tmp/deepseek_ocr_out")
    parser.add_argument("--images", nargs="*", default=[])
    args = parser.parse_args()

    images = [Path(p) for p in args.images] if args.images else pick_default_images()
    if not images:
        print("No images found. Provide --images paths.")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    for img in images:
        print("\n===", img)
        res = model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=str(img),
            output_path=str(out_dir),
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=args.crop_mode,
            save_results=args.save_results,
            test_compress=False,
        )
        # res may be dict or str depending on custom code
        print(res if isinstance(res, str) else str(res))

    print(f"\nSaved outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
