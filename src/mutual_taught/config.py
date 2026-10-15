"""Experiment configuration.

Defaults are the paper's settings (Section 5.1 and Appendix B of arXiv:2506.06292v2). A YAML file
only needs the keys it changes; unknown keys are rejected so typos fail loudly. Fields typed
``PerIteration`` accept a scalar or a list with one value per iteration.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mutual_taught.algorithms.filtering import FILTER_MODES

PerIteration = Any  # a scalar, or a list with one value per iteration

# fields that may change when a run is resumed: they affect speed or memory, not results
RESUMABLE_CHANGES = frozenset(
    {
        "gpus",
        "deepspeed_config",
        "sampling.gpu_memory_utilization",
        "scoring.max_tokens_per_batch",
        "dpo.per_device_batch_size",
        "reward_training.per_device_batch_size",
    }
)


def per_iteration_field(default: Any) -> Any:
    return field(default=default, metadata={"per_iteration": True})


def per_iteration(value: PerIteration, t: int) -> Any:
    """Value of a ``PerIteration`` field for iteration ``t`` (1-based)."""
    if isinstance(value, (list, tuple)):
        if t > len(value):
            raise ValueError(f"per-iteration value {value} has no entry for iteration {t}")
        return value[t - 1]
    return value


@dataclass
class ModelsConfig:
    policy: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # pi_0
    reward: str = "sfairXC/FsfairX-LLaMA3-RM-v0.1"  # r_0
    attn_implementation: str = "sdpa"  # or "flash_attention_2"


@dataclass
class DataConfig:
    dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    # D_MS, the fixed model-selection set: 2,000 UltraFeedback prompts (Appendix B)
    ms_split: str = "test_prefs"
    ms_size: int = 2000
    prompt_field: str = "prompt"
    # partitions in source order: D_1 (policy), D_R (reward model), D_2 (policy) (Section 5.1)
    partition_order: list[str] = field(default_factory=lambda: ["pm_1", "rm", "pm_2"])
    partition_size: int | None = 20000  # None: split all prompts equally
    split_mode: str = "sequential"  # sequential | random
    # longer chat-formatted prompts are dropped; must not exceed dpo.max_prompt_length
    max_prompt_tokens: int = 1800


@dataclass
class SamplingConfig:
    num_samples: int = 5  # M
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 4096
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9


@dataclass
class EStepConfig:
    length_control: PerIteration = per_iteration_field(True)  # Appendix B


@dataclass
class DPOConfig:
    beta: PerIteration = per_iteration_field(0.01)
    loss_type: str = "sigmoid"  # TRL DPO loss; "ipo" for the IPO baseline
    learning_rate: PerIteration = per_iteration_field(7.0e-7)
    global_batch_size: int = 128
    per_device_batch_size: int = 2
    max_length: int = 2048
    max_prompt_length: int = 1800
    num_epochs: int = 1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    save_steps: int = 50  # every checkpoint is a model-selection candidate
    gradient_checkpointing: bool = True


@dataclass
class ModelSelectionConfig:
    enabled: bool = True
    tau: float = 0.6  # halt when the best checkpoint's win rate is below tau (Appendix H)


@dataclass
class MStepConfig:
    enabled: bool = True  # False: the RM is never updated (iterative-DPO baseline)
    filter: str = "lqf"  # lqf | hqs | dst | none (Appendix F)
    rm_data: str = "mixed"  # mixed | pc | st (Section 5.3)
    # self-training pairs: E-step pairs of the first iteration, or of the current one
    self_training_source: str = "first"  # first | current
    rm_init: str = "base"  # base: every RM update starts from r_0 (Appendix B) | previous


@dataclass
class RewardTrainingConfig:
    learning_rate: float = 2.0e-6
    global_batch_size: int = 512
    per_device_batch_size: int = 1
    max_length: int = 2048
    num_epochs: int = 1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = True


@dataclass
class ScoringConfig:
    max_length: int = 8192
    max_tokens_per_batch: int = 32768


@dataclass
class ExperimentConfig:
    run_dir: str = "outputs/mutual-taught"
    seed: int = 42
    gpus: list[str] | None = None  # None: all GPUs in CUDA_VISIBLE_DEVICES / nvidia-smi
    iterations: int = 2  # T
    deepspeed_config: str | None = None  # None: the bundled ZeRO-3 config
    models: ModelsConfig = field(default_factory=ModelsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    estep: EStepConfig = field(default_factory=EStepConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    mstep: MStepConfig = field(default_factory=MStepConfig)
    reward_training: RewardTrainingConfig = field(default_factory=RewardTrainingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    def validate(self) -> None:
        checks = {
            "mstep.filter": (self.mstep.filter, FILTER_MODES),
            "mstep.rm_data": (self.mstep.rm_data, ("mixed", "pc", "st")),
            "mstep.self_training_source": (self.mstep.self_training_source, ("first", "current")),
            "mstep.rm_init": (self.mstep.rm_init, ("base", "previous")),
            "data.split_mode": (self.data.split_mode, ("sequential", "random")),
        }
        for key, (value, allowed) in checks.items():
            if value not in allowed:
                raise ValueError(f"{key}={value!r}, expected one of {allowed}")
        parts = self.data.partition_order
        missing = [f"pm_{t}" for t in range(1, self.iterations + 1) if f"pm_{t}" not in parts]
        if self.mstep.enabled and "rm" not in parts:
            missing.append("rm")
        if missing or "ms" in parts:
            raise ValueError(f"data.partition_order={parts}: missing {missing} ('ms' is reserved for D_MS)")
        if self.data.max_prompt_tokens > self.dpo.max_prompt_length:
            raise ValueError("data.max_prompt_tokens must be <= dpo.max_prompt_length, or DPO would truncate prompts")
        if self.dpo.max_prompt_length >= self.dpo.max_length:
            raise ValueError("dpo.max_prompt_length must be < dpo.max_length")
        for key, value in _per_iteration_values(self):
            if isinstance(value, list) and len(value) < self.iterations:
                raise ValueError(f"{key}={value} has fewer than iterations={self.iterations} entries")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _per_iteration_values(cfg: ExperimentConfig):
    for section in dataclasses.fields(cfg):
        sub = getattr(cfg, section.name)
        if dataclasses.is_dataclass(sub):
            for f in dataclasses.fields(sub):
                if f.metadata.get("per_iteration"):
                    yield f"{section.name}.{f.name}", getattr(sub, f.name)


def _coerce(value: Any, default: Any, key: str, per_iteration: bool) -> Any:
    """Check ``value`` against the type of the field's default.

    YAML reads ``1e-6`` (no decimal point) as a string, so numeric strings are converted.
    """
    if per_iteration and isinstance(value, list):
        if not value:
            raise ValueError(f"{key}: empty per-iteration list")
        return [_coerce(v, default, key, False) for v in value]
    if value is None or default is None or isinstance(default, list):
        return value
    kind = type(default)
    if kind is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{key}: expected true/false, got {value!r}")
        return value
    if kind in (int, float):
        if isinstance(value, bool):
            raise TypeError(f"{key}: expected a number, got {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise TypeError(f"{key}: expected a number, got {value!r}") from None
        if kind is int:
            if not number.is_integer():
                raise TypeError(f"{key}: expected an integer, got {value!r}")
            return int(number)
        return number
    if kind is str and not isinstance(value, str):
        raise TypeError(f"{key}: expected a string, got {value!r}")
    return value


def _update(obj: Any, values: dict, prefix: str = "") -> Any:
    fields = {f.name: f for f in dataclasses.fields(obj)}
    defaults = type(obj)()
    for key, value in values.items():
        if key not in fields:
            raise KeyError(f"unknown config key {prefix}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a mapping")
            _update(current, value, f"{prefix}{key}.")
        else:
            per_it = fields[key].metadata.get("per_iteration", False)
            setattr(obj, key, _coerce(value, getattr(defaults, key), f"{prefix}{key}", per_it))
    return obj


def changed_keys(old: dict, new: dict, prefix: str = "") -> set[str]:
    """Dotted keys whose values differ between two ``ExperimentConfig.to_dict()`` results."""
    keys = set()
    for key in old.keys() | new.keys():
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            keys |= changed_keys(a, b, f"{prefix}{key}.")
        elif a != b:
            keys.add(f"{prefix}{key}")
    return keys


def _override(cfg: ExperimentConfig, assignment: str) -> None:
    """Apply ``section.key=value``; the value is parsed as YAML (``0.1``, ``[true, false]``, ``null``)."""
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise ValueError(f"override must look like section.key=value, got {assignment!r}")
    *parents, leaf = key.split(".")
    nested: dict = {leaf: yaml.safe_load(raw)}
    for parent in reversed(parents):
        nested = {parent: nested}
    _update(cfg, nested)


def load_config(paths: str | Path | list | None = None, overrides: list[str] = ()) -> ExperimentConfig:
    """Paper defaults, updated by each YAML file in order, then by ``key=value`` overrides."""
    cfg = ExperimentConfig()
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for path in paths or []:
        with open(path) as f:
            _update(cfg, yaml.safe_load(f) or {})
    for assignment in overrides:
        _override(cfg, assignment)
    cfg.validate()
    return cfg
