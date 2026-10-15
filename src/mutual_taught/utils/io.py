"""File helpers. Writes are atomic (temp file + rename) so an interrupted run never leaves half files."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DONE_MARKER = "_DONE"


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def is_done(directory: str | Path) -> bool:
    return (Path(directory) / DONE_MARKER).exists()


def mark_done(directory: str | Path) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / DONE_MARKER).touch()


def setup_logging(log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
