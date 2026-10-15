"""Sampling worker (one GPU, one shard of a prompt file).

Input rows ``{"id", "prompt"}``; output rows add ``"responses"`` and ``"num_tokens"`` (n each).
"""

from __future__ import annotations

import argparse

from mutual_taught.inference.vllm_utils import generate
from mutual_taught.utils.io import read_jsonl, setup_logging, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    setup_logging()

    rows = read_jsonl(args.input)[args.shard :: args.num_shards]
    samples = generate(
        args.model,
        [r["prompt"] for r in rows],
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    for row, outs in zip(rows, samples, strict=True):
        row["responses"] = [text for text, _ in outs]
        row["num_tokens"] = [n for _, n in outs]
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
