"""Generate a policy's answers in the input formats of AlpacaEval 2 and Arena-Hard v0.1.

    mutual-taught answers --benchmark alpaca_eval --model PATH --output outputs.json
    mutual-taught answers --benchmark arena_hard --model PATH --question-file question.jsonl --output answers.jsonl

AlpacaEval prompts come from ``tatsu-lab/alpaca_eval`` (alpaca_eval.json). For Arena-Hard pass
``data/arena-hard-v0.1/question.jsonl`` from https://github.com/lmarena-ai/arena-hard-auto.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from mutual_taught.utils.io import read_json, read_jsonl, write_json, write_jsonl

BENCHMARKS = ("alpaca_eval", "arena_hard")


def generate_answers(
    benchmark: str,
    model: str,
    output: str | Path,
    model_id: str | None = None,
    question_file: str | Path | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 4096,
    tensor_parallel_size: int = 1,
    seed: int = 42,
) -> None:
    from mutual_taught.inference.vllm_utils import generate

    model_id = model_id or Path(model).name
    if benchmark == "alpaca_eval":
        from huggingface_hub import hf_hub_download

        questions = read_json(hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset"))
        prompts = [q["instruction"] for q in questions]
    elif benchmark == "arena_hard":
        if question_file is None:
            raise ValueError("--question-file is required for arena_hard")
        questions = read_jsonl(question_file)
        prompts = [q["turns"][0]["content"] for q in questions]
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}, expected one of {BENCHMARKS}")

    outputs = generate(
        model,
        prompts,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
    )
    if benchmark == "alpaca_eval":
        write_json(
            output,
            [
                {"instruction": q["instruction"], "output": out[0][0], "generator": model_id, "dataset": q["dataset"]}
                for q, out in zip(questions, outputs, strict=True)
            ],
        )
    else:
        write_jsonl(
            output,
            (
                {
                    "question_id": q["question_id"],
                    "answer_id": uuid.uuid4().hex,
                    "model_id": model_id,
                    "choices": [{"index": 0, "turns": [{"content": out[0][0], "token_len": out[0][1]}]}],
                    "tstamp": time.time(),
                }
                for q, out in zip(questions, outputs, strict=True)
            ),
        )
