"""Command-line interface: ``mutual-taught {run,prepare-data,answers}``."""

from __future__ import annotations

import argparse
from pathlib import Path

from mutual_taught import __version__
from mutual_taught.config import load_config
from mutual_taught.utils.io import setup_logging


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        required=True,
        nargs="+",
        help="experiment YAML (see configs/); several files are applied in order",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="override config values, e.g. dpo.beta=0.1 mstep.filter=dst",
    )


def download(cfg) -> None:
    """Fetch the policy, reward model and dataset of ``cfg`` (skips local paths)."""
    import logging

    from huggingface_hub import snapshot_download

    logger = logging.getLogger(__name__)
    for repo, repo_type in ((cfg.models.policy, "model"), (cfg.models.reward, "model"), (cfg.data.dataset, "dataset")):
        if Path(repo).exists():
            logger.info("skip %s (local path)", repo)
            continue
        path = snapshot_download(repo, repo_type=repo_type)
        logger.info("%s %s -> %s", repo_type, repo, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mutual-taught", description="Mutual-Taught (arXiv:2506.06292)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run (or resume) the full Mutual-Taught loop")
    _add_config_args(p_run)

    p_data = sub.add_parser("prepare-data", help="only build the prompt partitions")
    _add_config_args(p_data)

    p_dl = sub.add_parser("download", help="download a config's models and dataset into the HF cache")
    _add_config_args(p_dl)

    p_ans = sub.add_parser("answers", help="generate benchmark answers (AlpacaEval 2 / Arena-Hard)")
    p_ans.add_argument("--benchmark", required=True, choices=["alpaca_eval", "arena_hard"])
    p_ans.add_argument("--model", required=True)
    p_ans.add_argument("--output", required=True)
    p_ans.add_argument("--model-id", default=None)
    p_ans.add_argument("--question-file", default=None, help="Arena-Hard question.jsonl")
    p_ans.add_argument("--temperature", type=float, default=0.0)
    p_ans.add_argument("--top-p", type=float, default=1.0)
    p_ans.add_argument("--max-new-tokens", type=int, default=4096)
    p_ans.add_argument("--tensor-parallel-size", type=int, default=1)

    args = parser.parse_args(argv)
    if args.command == "answers":
        from mutual_taught.evaluation.answers import generate_answers

        setup_logging()
        generate_answers(
            args.benchmark,
            args.model,
            args.output,
            model_id=args.model_id,
            question_file=args.question_file,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        return

    cfg = load_config(args.config, args.overrides)
    if args.command == "download":
        setup_logging()
        download(cfg)
    elif args.command == "prepare-data":
        from mutual_taught.data import prepare_prompts

        setup_logging()
        prepare_prompts(cfg, Path(cfg.run_dir) / "prompts")
    else:
        from mutual_taught.pipeline import MutualTaught

        setup_logging(Path(cfg.run_dir) / "logs" / "pipeline.log")
        MutualTaught(cfg).run()


if __name__ == "__main__":
    main()
