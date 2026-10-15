"""The Mutual-Taught loop (Algorithm 1 of arXiv:2506.06292v2).

For t = 1..T:

1. **E-step** (Eq. 4): sample M responses from pi_{t-1} on D_t, rank them with r_{t-1} into
   preference pairs, and DPO-train pi_{t-1} (also the reference model), saving checkpoints.
2. **Model selection** (Eq. 6-7): pi_t is the checkpoint with the highest win rate over pi_{t-1}
   on D_MS under r_{t-1}. If that win rate is below tau the loop halts and keeps (pi_{t-1}, r_{t-1}).
3. **Pseudo-pairs**: for x in D_R, y_t ~ pi_t is preferred over y_{t-1} ~ pi_{t-1}.
4. **Filtering** (Eq. 8): drop pairs with r_{t-1}(y_t) - r_{t-1}(y_{t-1}) <= -eps_t, where
   eps_t = std of r_{t-1}(y_{t-1}) over D_R.
5. **M-step** (Eq. 5): Bradley-Terry training of the reward model, starting from r_0, on the
   filtered pseudo-pairs mixed with self-training pairs (Section 5.3).

The orchestrator uses no GPU itself. Sampling and scoring run one worker per GPU, training
runs torchrun + DeepSpeed ZeRO-3 over all GPUs. Every stage writes its outputs atomically and
is skipped when they exist, so re-running an interrupted run resumes it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

from mutual_taught.algorithms import filter_pseudo_pairs, select_checkpoint, select_preference_pair, win_rate
from mutual_taught.config import RESUMABLE_CHANGES, ExperimentConfig, changed_keys, per_iteration
from mutual_taught.utils.io import is_done, mark_done, read_json, read_jsonl, write_json, write_jsonl
from mutual_taught.utils.launch import launch, module_cmd, resolve_gpus, torchrun_cmd, wait_all

logger = logging.getLogger(__name__)


def _key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:16]


def _model_id(model: str | Path) -> str:
    """Cache identity of a model: its path, plus the save time for local directories.

    A directory that is deleted and retrained (e.g. an iteration redone by hand) gets a new
    identity, so its old generations and scores are not reused.
    """
    config = Path(model) / "config.json"
    return f"{model}@{config.stat().st_mtime_ns}" if config.is_file() else str(model)


class MutualTaught:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir).resolve()
        self.prompts = self.run_dir / "prompts"
        self.logs = self.run_dir / "logs"
        self.gpus = resolve_gpus(cfg.gpus)
        if cfg.deepspeed_config:
            self.ds_config = str(Path(cfg.deepspeed_config).resolve())
        else:
            self.ds_config = str(resources.files("mutual_taught") / "resources" / "deepspeed_zero3.json")
        # fail before any GPU work rather than at the first training stage
        for name, section in (("dpo", cfg.dpo), ("reward_training", cfg.reward_training)):
            per_step = section.per_device_batch_size * len(self.gpus)
            if section.global_batch_size % per_step:
                raise ValueError(
                    f"{name}.global_batch_size={section.global_batch_size} is not divisible by "
                    f"per_device_batch_size x GPUs = {section.per_device_batch_size} x {len(self.gpus)}"
                )

    # ------------------------------------------------------------------ GPU workers (cached)

    def _sharded(self, module: str, out_path: Path, args: list[str], tag: str) -> list[dict]:
        """Run ``module`` once per GPU on interleaved shards and merge the shards in input order."""
        if out_path.exists():
            return read_jsonl(out_path)
        shard_dir = out_path.with_name(out_path.stem + ".shards")
        if shard_dir.exists():  # left over from an interrupted attempt
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True)
        n = len(self.gpus)
        procs = [
            launch(
                module_cmd(
                    module, *args, "--shard", str(i), "--num_shards", str(n), "--output", str(shard_dir / f"{i}.jsonl")
                ),
                [gpu],
                self.logs / f"{tag}.shard{i}.log",
            )
            for i, gpu in enumerate(self.gpus)
        ]
        wait_all(procs, tag)
        shards = [read_jsonl(shard_dir / f"{i}.jsonl") for i in range(n)]
        total = sum(len(s) for s in shards)
        rows = [shards[j % n][j // n] for j in range(total)]  # shard i holds rows i, i+n, ...
        write_jsonl(out_path, rows)
        shutil.rmtree(shard_dir)
        return rows

    def generate(self, model: str, prompt_set: str, n: int, seed: int) -> Path:
        """Responses of ``model`` on ``prompts/<prompt_set>.jsonl``; cached by model path and settings."""
        s = self.cfg.sampling
        key = _key(_model_id(model), prompt_set, n, seed, s.temperature, s.top_p, s.max_new_tokens)
        out = self.run_dir / "cache" / "generations" / f"{prompt_set}-n{n}-{key}.jsonl"
        if not out.exists():
            write_json(
                out.with_suffix(".meta.json"), {"model": str(model), "prompt_set": prompt_set, "n": n, "seed": seed}
            )
        args = [
            "--model",
            str(model),
            "--input",
            str(self.prompts / f"{prompt_set}.jsonl"),
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--temperature",
            str(s.temperature),
            "--top_p",
            str(s.top_p),
            "--max_new_tokens",
            str(s.max_new_tokens),
            "--max_model_len",
            str(s.max_model_len),
            "--gpu_memory_utilization",
            str(s.gpu_memory_utilization),
        ]
        self._sharded("mutual_taught.inference.generate", out, args, f"generate-{out.stem}")
        return out

    def sample_and_score(self, model: str, reward_model: str, prompt_set: str, n: int, seed: int) -> list[dict]:
        """Responses of ``model`` scored by ``reward_model`` (both steps cached)."""
        generations = self.generate(model, prompt_set, n, seed)
        out = self.run_dir / "cache" / "scores" / f"{generations.stem}-rm{_key(_model_id(reward_model))}.jsonl"
        if not out.exists():
            write_json(
                out.with_suffix(".meta.json"), {"generations": str(generations), "reward_model": str(reward_model)}
            )
        sc = self.cfg.scoring
        args = [
            "--model",
            str(reward_model),
            "--input",
            str(generations),
            "--max_length",
            str(sc.max_length),
            "--max_tokens_per_batch",
            str(sc.max_tokens_per_batch),
            "--attn_implementation",
            self.cfg.models.attn_implementation,
        ]
        return self._sharded("mutual_taught.inference.score", out, args, f"score-{out.stem}")

    def _train(self, module: str, out_dir: Path, args: dict, global_batch: int, per_device: int, tag: str) -> None:
        if is_done(out_dir):
            return
        if out_dir.exists():
            logger.warning("removing incomplete training output %s", out_dir)
            shutil.rmtree(out_dir)
        n = len(self.gpus)
        args = dict(
            args,
            output_dir=str(out_dir),
            per_device_batch_size=per_device,
            gradient_accumulation_steps=global_batch // (per_device * n),
            deepspeed=self.ds_config,
            seed=self.cfg.seed,
            attn_implementation=self.cfg.models.attn_implementation,
        )
        args_file = self.logs / f"{tag}.args.json"
        write_json(args_file, args)
        wait_all([launch(torchrun_cmd(module, n, str(args_file)), self.gpus, self.logs / f"{tag}.log")], tag)
        mark_done(out_dir)

    # ------------------------------------------------------------------ one iteration

    def e_step(self, t: int, policy: str, reward_model: str, it_dir: Path) -> Path:
        cfg = self.cfg
        pairs_path = it_dir / "estep_pairs.jsonl"
        if not pairs_path.exists():
            rows = self.sample_and_score(policy, reward_model, f"pm_{t}", cfg.sampling.num_samples, cfg.seed + t)
            length_control = per_iteration(cfg.estep.length_control, t)
            pairs = []
            for r in rows:
                picked = select_preference_pair(r["scores"], r["num_tokens"], length_control)
                if picked is None:
                    continue
                c, j = picked
                pairs.append(
                    {
                        "id": r["id"],
                        "prompt": r["prompt"],
                        "chosen": r["responses"][c],
                        "rejected": r["responses"][j],
                        "chosen_score": r["scores"][c],
                        "rejected_score": r["scores"][j],
                    }
                )
            logger.info(
                "iter %d E-step: %d pairs from %d prompts (length_control=%s)", t, len(pairs), len(rows), length_control
            )
            write_jsonl(pairs_path, pairs)

        d = cfg.dpo
        policy_dir = it_dir / "policy"
        self._train(
            "mutual_taught.training.dpo",
            policy_dir,
            {
                "model": str(policy),
                "data": str(pairs_path),
                "beta": per_iteration(d.beta, t),
                "loss_type": d.loss_type,
                "learning_rate": per_iteration(d.learning_rate, t),
                "num_epochs": d.num_epochs,
                "lr_scheduler_type": d.lr_scheduler_type,
                "warmup_ratio": d.warmup_ratio,
                "max_length": d.max_length,
                "max_prompt_length": d.max_prompt_length,
                "save_steps": d.save_steps,
                "gradient_checkpointing": d.gradient_checkpointing,
            },
            d.global_batch_size,
            d.per_device_batch_size,
            f"iter{t}-dpo",
        )
        return policy_dir

    @staticmethod
    def candidates(policy_dir: Path) -> dict[str, Path]:
        """Saved checkpoints in step order, plus the final model when it is not the last checkpoint."""
        ckpts = sorted(policy_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        found = {p.name: p for p in ckpts}
        if not ckpts or int(ckpts[-1].name.split("-")[1]) != read_json(ckpts[-1] / "trainer_state.json")["max_steps"]:
            found["final"] = policy_dir
        return found

    def model_selection(
        self, t: int, policy: str, reward_model: str, policy_dir: Path, it_dir: Path
    ) -> tuple[Path, bool]:
        """Eq. 6-7. Returns the selected policy and whether the loop halts."""
        cfg = self.cfg
        if not cfg.model_selection.enabled:
            return policy_dir, False
        previous = {r["id"]: r["scores"][0] for r in self.sample_and_score(policy, reward_model, "ms", 1, cfg.seed)}
        cands = self.candidates(policy_dir)
        rates = {}
        for name, path in cands.items():
            rows = self.sample_and_score(str(path), reward_model, "ms", 1, cfg.seed)
            rates[name] = win_rate([r["scores"][0] for r in rows], [previous[r["id"]] for r in rows])
            logger.info("iter %d model selection: %s win rate %.3f", t, name, rates[name])
        tau = cfg.model_selection.tau
        best, best_rate, halt = select_checkpoint(rates, list(cands), tau)
        write_json(
            it_dir / "selection.json",
            {
                "win_rates": rates,
                "selected": best,
                "selected_path": str(cands[best]),
                "win_rate": best_rate,
                "tau": tau,
                "halt": halt,
            },
        )
        return cands[best], halt

    def m_step(self, t: int, policy: str, new_policy: Path, reward_model: str, it_dir: Path) -> Path:
        cfg, m = self.cfg, self.cfg.mstep
        data_path = it_dir / "rm_pairs.jsonl"
        if not data_path.exists():
            new = self.sample_and_score(str(new_policy), reward_model, "rm", 1, cfg.seed)  # y_t
            old_by_id = {r["id"]: r for r in self.sample_and_score(policy, reward_model, "rm", 1, cfg.seed)}
            old = [old_by_id[r["id"]] for r in new]  # y_{t-1}
            decisions, stats = filter_pseudo_pairs(
                [r["scores"][0] for r in new], [r["scores"][0] for r in old], m.filter
            )
            pc = []
            for n_row, o_row, keep in zip(new, old, decisions, strict=True):
                if keep is None:
                    continue
                win, lose = (n_row, o_row) if keep else (o_row, n_row)
                pc.append(
                    {
                        "id": n_row["id"],
                        "prompt": n_row["prompt"],
                        "source": "pc",
                        "chosen": win["responses"][0],
                        "rejected": lose["responses"][0],
                    }
                )
            st_iter = 1 if m.self_training_source == "first" else t
            st = [dict(r, source="st") for r in read_jsonl(self.run_dir / f"iter{st_iter}" / "estep_pairs.jsonl")]
            data = {"mixed": pc + st, "pc": pc, "st": st}[m.rm_data]
            random.Random(cfg.seed + t).shuffle(data)
            summary = {
                **stats.to_dict(),
                "rm_data": m.rm_data,
                "num_pc": sum(r["source"] == "pc" for r in data),
                "num_st": sum(r["source"] == "st" for r in data),
                "num_total": len(data),
            }
            write_json(it_dir / "mstep_stats.json", summary)
            logger.info("iter %d M-step data: %s", t, summary)
            write_jsonl(data_path, data)

        r = cfg.reward_training
        rm_dir = it_dir / "reward"
        self._train(
            "mutual_taught.training.reward",
            rm_dir,
            {
                "model": cfg.models.reward if m.rm_init == "base" else str(reward_model),
                "data": str(data_path),
                "learning_rate": r.learning_rate,
                "num_epochs": r.num_epochs,
                "lr_scheduler_type": r.lr_scheduler_type,
                "warmup_ratio": r.warmup_ratio,
                "max_length": r.max_length,
                "gradient_checkpointing": r.gradient_checkpointing,
            },
            r.global_batch_size,
            r.per_device_batch_size,
            f"iter{t}-rm",
        )
        return rm_dir

    # ------------------------------------------------------------------ main loop

    def _check_run_dir(self) -> None:
        saved = self.run_dir / "config.json"
        current = self.cfg.to_dict()
        if saved.exists():
            changed = changed_keys(read_json(saved), current) - RESUMABLE_CHANGES
            if changed:
                raise SystemExit(
                    f"cannot resume {self.run_dir}: {sorted(changed)} differ from {saved}. Use a new run_dir "
                    f"(cached stages are keyed by model and prompt set, not by hyperparameters). "
                    f"Only {sorted(RESUMABLE_CHANGES)} may change between resumes."
                )
        self.cfg.save(saved)

    def run(self) -> dict:
        cfg = self.cfg
        self._check_run_dir()
        if not is_done(self.prompts):
            from mutual_taught.data import prepare_prompts

            prepare_prompts(cfg, self.prompts)

        policy, reward_model = cfg.models.policy, cfg.models.reward
        history = []
        for t in range(1, cfg.iterations + 1):
            it_dir = self.run_dir / f"iter{t}"
            logger.info("===== iteration %d: policy=%s reward_model=%s", t, policy, reward_model)
            policy_dir = self.e_step(t, policy, reward_model, it_dir)
            new_policy, halt = self.model_selection(t, policy, reward_model, policy_dir, it_dir)
            if halt:
                logger.info("iter %d: best win rate below tau; halting with the previous models", t)
                history.append({"iteration": t, "halted": True})
                break
            new_rm = self.m_step(t, policy, new_policy, reward_model, it_dir) if cfg.mstep.enabled else reward_model
            history.append({"iteration": t, "halted": False, "policy": str(new_policy), "reward_model": str(new_rm)})
            policy, reward_model = str(new_policy), str(new_rm)

        summary = {"iterations": history, "final_policy": policy, "final_reward_model": reward_model}
        write_json(self.run_dir / "summary.json", summary)
        logger.info("finished: %s", summary)
        return summary
