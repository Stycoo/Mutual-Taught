"""Chat formatting and reward-model helpers shared by sampling, scoring and training."""

from __future__ import annotations

from typing import Any

PAD_TOKEN = "<|mt_pad|>"


def prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    """Token ids of a single-turn user prompt followed by the assistant header.

    Tokenizing through the chat template (instead of passing text to vLLM) avoids a second BOS.
    """
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)


def conversation_ids(tokenizer: Any, prompt: str, response: str) -> list[int]:
    """Token ids of ``(prompt, response)`` as the reward model sees it."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    )


def ensure_pad_token(tokenizer: Any, model: Any) -> None:
    """Make padding unambiguous for a sequence-classification reward model.

    HF classification heads read the score at the last non-pad token. If the pad id also occurs
    inside chat-formatted text (e.g. pad == ``<|eot_id|>``), the score would be read at the wrong
    position, so a dedicated pad token is added (and the embeddings resized when needed).
    """
    probe = set(conversation_ids(tokenizer, "probe", "probe"))
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id in probe:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        # compare with config.vocab_size, not the weight shape: under ZeRO-3 weights are partitioned
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id


def right_pad(sequences: list[list[int]], pad_token_id: int):
    """Stack token-id lists into right-padded ``input_ids`` and ``attention_mask`` tensors."""
    import torch

    width = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask
