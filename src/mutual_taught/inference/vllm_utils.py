"""vLLM helpers shared by the sampling worker and benchmark answer generation."""

from __future__ import annotations

from typing import Any

from mutual_taught.modeling import prompt_ids


def stop_token_ids(model_path: str, tokenizer: Any) -> list[int]:
    """EOS ids from the tokenizer and generation config, plus common end-of-turn tokens."""
    from transformers import GenerationConfig

    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    try:
        eos = GenerationConfig.from_pretrained(model_path).eos_token_id
    except OSError:
        eos = None
    if eos is not None:
        ids.update(eos if isinstance(eos, list) else [eos])
    for token in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(ids)


def generate(
    model: str,
    prompts: list[str],
    *,
    n: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 4096,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    seed: int = 42,
) -> list[list[tuple[str, int]]]:
    """Sample ``n`` responses per prompt; returns ``(text, num_generated_tokens)`` per sample."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        seed=seed,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        # no per-request seed, so the n samples differ; LLM(seed=...) keeps the run reproducible
        stop_token_ids=stop_token_ids(model, tokenizer),
    )
    inputs = [{"prompt_token_ids": prompt_ids(tokenizer, p)} for p in prompts]
    outputs = llm.generate(inputs, params)
    return [[(c.text.strip(), len(c.token_ids)) for c in out.outputs] for out in outputs]
