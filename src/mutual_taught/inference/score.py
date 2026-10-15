"""Reward-scoring worker (one GPU, one shard). Adds ``"scores"`` aligned with ``"responses"``."""

from __future__ import annotations

import argparse
from typing import Any

from mutual_taught.modeling import conversation_ids, ensure_pad_token, right_pad
from mutual_taught.utils.io import read_jsonl, setup_logging, write_jsonl


def score_sequences(
    model: Any, pad_token_id: int, sequences: list[list[int]], max_tokens_per_batch: int
) -> list[float]:
    """Reward of every token-id sequence, batched by length; returned in input order."""
    import torch

    device = next(model.parameters()).device
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    scores = [0.0] * len(sequences)
    batch: list[int] = []

    @torch.no_grad()
    def flush() -> None:
        input_ids, attention_mask = right_pad([sequences[i] for i in batch], pad_token_id)
        logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        for i, value in zip(batch, logits[:, 0].float().tolist(), strict=True):
            scores[i] = value
        batch.clear()

    for i in order:
        # ascending lengths: the incoming sequence is the widest of the batch
        if batch and len(sequences[i]) * (len(batch) + 1) > max_tokens_per_batch:
            flush()
        batch.append(i)
    if batch:
        flush()
    return scores


def score_rows(rows: list[dict], model: Any, tokenizer: Any, max_length: int, max_tokens_per_batch: int) -> list[dict]:
    """Add ``"scores"`` (one per response) to every row; over-long sequences keep their end."""
    sequences, owners = [], []
    for idx, row in enumerate(rows):
        for response in row["responses"]:
            sequences.append(conversation_ids(tokenizer, row["prompt"], response)[-max_length:])
            owners.append(idx)
    scores = score_sequences(model, tokenizer.pad_token_id, sequences, max_tokens_per_batch)
    for row in rows:
        row["scores"] = []
    for idx, value in zip(owners, scores, strict=True):
        rows[idx]["scores"].append(value)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--max_tokens_per_batch", type=int, default=32768)
    p.add_argument("--attn_implementation", default="sdpa")
    args = p.parse_args()
    setup_logging()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = read_jsonl(args.input)[args.shard :: args.num_shards]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.bfloat16, attn_implementation=args.attn_implementation
    )
    ensure_pad_token(tokenizer, model)
    model.cuda().eval()
    write_jsonl(args.output, score_rows(rows, model, tokenizer, args.max_length, args.max_tokens_per_batch))


if __name__ == "__main__":
    main()
