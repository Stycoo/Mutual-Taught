# Mutual-Taught: Co-adapting Policy and Reward Models

[![arXiv](https://img.shields.io/badge/arXiv-2506.06292-b31b1b.svg)](https://arxiv.org/abs/2506.06292)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Code for **Mutual-Taught for Co-adapting Policy and Reward Models**.

## The problem

During preference optimization, the policy drifts away from the data the reward model was trained
on, and the reward model's feedback degrades.

## The method

Mutual-Taught improves the policy and the reward model together, without new human labels, in an
EM-style loop:

- **E-step:** the policy is trained with DPO on pairs ranked by the current reward model.
- **M-step:** the reward model is trained on pseudo-preference pairs, in which the policy's
  response *after* the E-step is preferred over its response *before* it.

Two mechanisms keep the loop stable:

- **Model selection:** checkpoint selection by win rate, with early stopping at τ.
- **Data filtering:** variance-aware filtering of noisy pseudo-pairs.

With Llama-3-8B-Instruct and FsfairX-LLaMA3-RM-v0.1 on UltraFeedback, the paper reports:

- the policy reaches 54.1 LC win rate on AlpacaEval 2;
- the reward model reaches 87.0 on RewardBench, on par with GPT-4o-2024-08-06.

```
for t = 1..T
  E-step     sample M=5 responses from π_{t-1} on D_t → rank with r_{t-1} → DPO → checkpoints {π_t^k}
  selection  π_t = argmax_k  win rate of π_t^k over π_{t-1} on D_MS under r_{t-1};  stop if < τ
  M-step     on D_R: pairs (y_t ≻ y_{t-1}), drop Δr ≤ −std  →  Bradley-Terry training of r_t (from r_0)
```

## Installation

```bash
git clone https://github.com/Stycoo/Mutual-Taught.git && cd Mutual-Taught
conda create -n mutual-taught python=3.12 -y && conda activate mutual-taught
pip install -e ".[dev]"
pip install flash-attn==2.8.3 --no-build-isolation   # optional, or a prebuilt wheel for torch 2.8
huggingface-cli login                                # Llama-3 is a gated model
```

Requirements:

- Python ≥ 3.10 and CUDA GPUs.
- Pinned stack: vLLM 0.11.0, torch 2.8.0, transformers 4.57.1, TRL 0.24.0, DeepSpeed 0.17.6.
- An 8B run needs GPUs with enough memory for ZeRO-3 full fine-tuning: 2× H200 or 8× A100 80GB.

## Quick start

A run with small models (Qwen2.5-0.5B-Instruct policy, Skywork-Reward-V2-Qwen3-0.6B reward model)
goes through every stage of the loop:

```bash
mutual-taught run --config configs/smoke.yaml
```

The main experiment of the paper:

```bash
mutual-taught run --config configs/mutual_taught/llama3_8b_fsfairx.yaml
```

- **GPUs.** The pipeline uses every GPU in `CUDA_VISIBLE_DEVICES`. Sampling and scoring run one
  worker per GPU; training runs DeepSpeed ZeRO-3 across all GPUs.
- **Resuming.** Re-running the same command resumes an interrupted run: finished stages are skipped,
  and sampled responses and reward scores are cached. Settings that only affect speed or memory,
  such as GPUs or per-device batch sizes, may change between resumes.
- **Offline compute nodes.** `mutual-taught download --config <yaml>` fetches the models and dataset ahead of time.
- **Overrides.** Several YAML files are applied in order (`--config base.yaml ablation.yaml`), and any
  value can be changed from the command line:

```bash
mutual-taught run --config configs/mutual_taught/llama3_8b_fsfairx.yaml \
  --set run_dir=outputs/ablation_dst mstep.filter=dst
```

Other experiments of the paper:

- **Baselines:** `configs/baselines/` (iterative DPO, offline DPO).
- **Reward-model generalization (Table 3):** `configs/generalization/mistral_7b.yaml`.
- **Benchmarks:** `mutual-taught answers` writes answers in the AlpacaEval 2 and Arena-Hard formats,
  which are then judged with the official tools.

## Configuration

Defaults reproduce the paper and are documented in
[`src/mutual_taught/config.py`](src/mutual_taught/config.py). The most important settings:

| Key | Default | Meaning |
|---|---|---|
| `iterations` | 2 | number of E/M iterations T |
| `data.partition_order` / `partition_size` | `[pm_1, rm, pm_2]` / 20000 | UltraFeedback partitions D_1, D_R, D_2 |
| `sampling.num_samples`, `temperature`, `top_p` | 5, 0.8, 0.95 | E-step sampling |
| `estep.length_control` | true | chosen = shortest above-average response |
| `dpo.beta`, `dpo.learning_rate`, `dpo.global_batch_size` | 0.01, 7e-7, 128 | E-step DPO |
| `model_selection.enabled`, `model_selection.tau` | true, 0.6 | checkpoint selection and early stop |
| `mstep.filter` | `lqf` | pseudo-pair filter: `lqf`, `hqs`, `dst`, `none` |
| `mstep.rm_data` | `mixed` | RM data: `mixed` (self-training + policy comparison), `pc`, `st` |
| `mstep.rm_init` | `base` | each M-step starts from r_0 |
| `reward_training.learning_rate`, `global_batch_size` | 2e-6, 512 | M-step Bradley-Terry training |

Settings marked as per-iteration accept one value per iteration, e.g. `estep.length_control=[true,false]`.

## Outputs

```
<run_dir>/
├── config.json                 resolved config; resuming with different results-relevant settings is refused
├── prompts/                    D_1, D_R, D_2 (pm_*.jsonl, rm.jsonl) and D_MS (ms.jsonl)
├── iter{t}/
│   ├── estep_pairs.jsonl       E-step DPO pairs
│   ├── policy/checkpoint-*/    E-step checkpoints (model-selection candidates)
│   ├── selection.json          win rate of every checkpoint, selected π_t, halt flag
│   ├── rm_pairs.jsonl          M-step data (source: pc | st)
│   ├── mstep_stats.json        ε_t and filtering statistics
│   └── reward/                 r_t
├── summary.json                final policy and reward model
├── cache/                      sampled responses and reward scores
└── logs/                       pipeline log and one log per worker / training run
```

## Repository structure

```
src/mutual_taught/
├── algorithms/      pure logic: partition, preference (length control), selection, filtering
├── inference/       vLLM sampling and reward-scoring workers
├── training/        DPO (E-step) and Bradley-Terry reward-model (M-step) trainers
├── evaluation/      AlpacaEval 2 / Arena-Hard answer generation
├── pipeline.py      the Mutual-Taught loop (Algorithm 1)
├── config.py        typed configuration with the paper defaults
├── data.py          UltraFeedback prompt partitions
└── cli.py           `mutual-taught` command
configs/             experiment configs (main, baselines, generalization, smoke test)
cluster/pbs/         PBS job scripts (environment setup, run/resume)
```

## Development

```bash
ruff check src && ruff format --check src
pre-commit install
```

## Status

This is a refactored implementation of the method; it is not the code that produced the paper's
numbers, and it has not yet been validated end to end on GPUs. Please open an issue if your numbers
differ from the paper.

## Citation

```bibtex
@article{shi2025mutualtaught,
  title   = {Mutual-Taught for Co-adapting Policy and Reward Models},
  author  = {Shi, Tianyuan and Huang, Canbin and Wan, Fanqi and Zhong, Longguang and Yang, Ziyi and
             Shen, Weizhou and Quan, Xiaojun and Yan, Ming},
  journal = {arXiv preprint arXiv:2506.06292},
  year    = {2025}
}
```

## Acknowledgements

The original experiments used:

- [Alignment Handbook](https://github.com/huggingface/alignment-handbook);
- [RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling);
- the sampling and length-control setup of [SimPO](https://github.com/princeton-nlp/SimPO).

This implementation is built on [TRL](https://github.com/huggingface/trl),
[vLLM](https://github.com/vllm-project/vllm) and [DeepSpeed](https://github.com/microsoft/DeepSpeed).

## License

[Apache 2.0](LICENSE)
