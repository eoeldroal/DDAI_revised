#!/usr/bin/env python3
"""Test DeepSeek-OCR via vLLM OpenAI-compatible API."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys
import time


def file_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--prompt", default="Free OCR.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--images", nargs="+", required=True)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        print("openai package missing. Install with: pip install openai", file=sys.stderr)
        raise SystemExit(1) from exc

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600)

    for img_path in args.images:
        path = Path(img_path)
        if not path.exists():
            print(f"Missing image: {path}", file=sys.stderr)
            continue
        data_url = file_to_data_url(path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]

        start = time.time()
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=0.0,
            extra_body={
                "skip_special_tokens": False,
                "vllm_xargs": {
                    "ngram_size": 30,
                    "window_size": 90,
                    "whitelist_token_ids": [128821, 128822],
                },
            },
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content
        print("\n===", path)
        print(f"Elapsed: {elapsed:.2f}s")
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
