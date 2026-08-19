"""JSONL run logging. No W&B, no cloud trackers.

One directory per run holding the config, a JSONL step log and any eval
reports, so a run is a thing on disk that can be diffed and replotted rather
than a page on someone else's website.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")


def create_run_dir(tag: str, root: Path = RUNS_DIR) -> Path:
    """`runs/<timestamp>_<tag>/`."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{stamp}_{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class JsonlLogger:
    """Append-only step log, flushed every line.

    Flushing every write costs nothing at one line per optimizer step and
    means a run killed at step 400 still has 400 steps of history — which is
    the case that matters, since the runs worth reading are often the ones
    that died.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, **fields: Any) -> dict:
        record = {"time": datetime.now().isoformat(timespec="seconds"), **fields}
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_config(run_dir: Path, config: dict) -> Path:
    path = Path(run_dir) / "config.json"
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    return path


def read_log(path: Path) -> list[dict]:
    """Every step record from a run's JSONL log."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
