"""M-step reward-model update (Eq. 5): maximize the Bradley-Terry log-likelihood of preference pairs.

The objective, -log sigmoid(r(chosen) - r(rejected)), matches RLHFlow/RLHF-Reward-Modeling, the
framework used in the paper. Data rows: ``{"prompt", "chosen", "rejected"}`` as plain strings.
Pairs longer than ``max_length`` tokens are dropped, not truncated.
"""

from __future__ import annotations

import logging
from typing import Any

from mutual_taught.modeling import conversation_ids, ensure_pad_token, right_pad
from mutual_taught.training.common import load_args, model_kwargs, training_kwargs
from mutual_taught.utils.io import read_jsonl, setup_logging

logger = logging.getLogger(__name__)


class PairCollator:
    """Interleave pairs: rows 0, 2, 4, ... are chosen, rows 1, 3, 5, ... rejected."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        sequences = [s for f in features for s in (f["chosen_ids"], f["rejected_ids"])]
        input_ids, attention_mask = right_pad(sequences, self.pad_token_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def bradley_terry_loss(rewards):
    """Mean negative log-likelihood for interleaved (chosen, rejected) rewards."""
    import torch.nn.functional as F

    chosen, rejected = rewards[0::2].float(), rewards[1::2].float()
    return -F.logsigmoid(chosen - rejected).mean(), chosen, rejected


def main(argv: list[str] | None = None) -> None:
    a = load_args(argv)
    setup_logging()

    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    class BradleyTerryTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits[:, 0]
            loss, chosen, rejected = bradley_terry_loss(rewards)
            if return_outputs:
                return loss, {"rewards_chosen": chosen, "rewards_rejected": rejected}
            return loss

    args = TrainingArguments(**training_kwargs(a), remove_unused_columns=False, save_strategy="no")
    tokenizer = AutoTokenizer.from_pretrained(a["model"])
    model = AutoModelForSequenceClassification.from_pretrained(a["model"], num_labels=1, **model_kwargs(a))
    ensure_pad_token(tokenizer, model)
    model.config.use_cache = False

    pairs = [
        {
            "chosen_ids": conversation_ids(tokenizer, r["prompt"], r["chosen"]),
            "rejected_ids": conversation_ids(tokenizer, r["prompt"], r["rejected"]),
        }
        for r in read_jsonl(a["data"])
    ]
    kept = [p for p in pairs if max(len(p["chosen_ids"]), len(p["rejected_ids"])) <= a["max_length"]]
    logger.info(
        "reward-model pairs: %d (%d dropped as longer than %d)", len(kept), len(pairs) - len(kept), a["max_length"]
    )

    trainer = BradleyTerryTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(kept),
        data_collator=PairCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    # the loss is a per-micro-batch mean, so let Trainer divide it by the accumulation steps
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    trainer.save_model(a["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(a["output_dir"])


if __name__ == "__main__":
    main()
