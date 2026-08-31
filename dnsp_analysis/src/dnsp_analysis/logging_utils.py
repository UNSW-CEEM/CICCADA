"""Logging and JSON manifest utilities."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


LOGGER_NAME = "dnsp_analysis"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _json_default(value: Any) -> Any:
    if isinstance(value, (Path, datetime, date)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a small JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)

