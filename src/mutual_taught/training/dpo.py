"""E-step policy update (Eq. 4): DPO starting from pi_{t-1}, with pi_{t-1} as the reference model.

Extra args (see ``training.common``): beta, loss_type, max_prompt_length, save_steps.
Data rows: ``{"prompt", "chosen", "rejected"}`` as plain strings.
A weights-only checkpoint is saved every ``save_steps``; each one is a candidate for model selection.
"""

from __future__ import annotations

from typing import Any

from mutual_taught.training.common import load_args, model_kwargs, training_kwargs
from mutual_taught.utils.io import read_jsonl, setup_logging


def dpo_config(a: dict[str, Any]):
    from trl import DPOConfig

    return DPOConfig(
        **training_kwargs(a),
        beta=a["beta"],
        loss_type=a.get("loss_type", "sigmoid"),
        max_length=a["max_length"],
        max_prompt_length=a["max_prompt_length"],
        # keep the prompt and cut over-long responses at the end (TRL's default "keep_end" would
        # drop the start of the sequence, i.e. the prompt)
        truncation_mode="keep_start",
        save_strategy="steps",
        save_steps=a["save_steps"],
        save_only_model=True,
    )


def main(argv: list[str] | None = None) -> None:
    a = load_args(argv)
    setup_logging()

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    # Build the config before the models so that ZeRO-3 partitions them at load time.
    config = dpo_config(a)
    tokenizer = AutoTokenizer.from_pretrained(a["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(a["model"], **model_kwargs(a))
    # loaded explicitly: TRL cannot copy a ZeRO-3 model into a reference model
    ref_model = AutoModelForCausalLM.from_pretrained(a["model"], **model_kwargs(a))

    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "chosen": [{"role": "assistant", "content": r["chosen"]}],
                "rejected": [{"role": "assistant", "content": r["rejected"]}],
            }
            for r in read_jsonl(a["data"])
        ]
    )
    trainer = DPOTrainer(
        model=model, ref_model=ref_model, args=config, train_dataset=dataset, processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model(a["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(a["output_dir"])


if __name__ == "__main__":
    main()
