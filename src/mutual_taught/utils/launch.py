"""Launching GPU workers as subprocesses (one per GPU, or torchrun over all GPUs)."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_PARENT = Path(__file__).resolve().parents[2]  # the directory containing mutual_taught/


def resolve_gpus(gpus: Sequence[str | int] | None) -> list[str]:
    """GPU ids from the config, else ``CUDA_VISIBLE_DEVICES``, else every GPU nvidia-smi lists."""
    if gpus:
        return [str(g) for g in gpus]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [g.strip() for g in visible.split(",") if g.strip()]
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU")]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def module_cmd(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def torchrun_cmd(module: str, num_procs: int, *args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_procs}",
        f"--master_port={free_port()}",
        "-m",
        module,
        *args,
    ]


def launch(cmd: list[str], gpus: Sequence[str], log_path: Path) -> subprocess.Popen:
    """Start ``cmd`` on ``gpus`` with stdout/stderr appended to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # make the package importable in the child even when it is not pip-installed
    pythonpath = os.pathsep.join(filter(None, [str(PACKAGE_PARENT), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), PYTHONPATH=pythonpath)
    logger.info("launch [GPU %s] %s (log: %s)", ",".join(gpus), " ".join(cmd), log_path)
    with open(log_path, "a") as log:
        return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wait_all(procs: Sequence[subprocess.Popen], what: str, poll_seconds: float = 5.0) -> None:
    """Wait for all processes; as soon as one fails, terminate the others and raise."""
    while True:
        codes = [p.poll() for p in procs]
        failed = [c for c in codes if c not in (None, 0)]
        if failed or all(c == 0 for c in codes):
            break
        time.sleep(poll_seconds)
    if failed:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        raise RuntimeError(f"{what} failed with exit codes {[p.returncode for p in procs]}; see its logs")
