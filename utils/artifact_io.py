"""
utils/artifact_io.py
--------------------
Simple JSON artifact save/load helpers used by pipeline stages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def save_artifact(
    data: Any,
    path: str | Path,
    description: str = "",
    logger_instance: logging.Logger | None = None,
    default: Any = None,
) -> None:
    """Serialize *data* to JSON at *path*, creating parent dirs as needed."""
    log = logger_instance or logging.getLogger(__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)
    if description:
        log.debug(f"Artifact saved [{description}]: {path}")


def load_artifact(
    path: str | Path,
    description: str = "",
    logger_instance: logging.Logger | None = None,
) -> Any:
    """Load a JSON artifact from *path*. Returns None if file does not exist."""
    log = logger_instance or logging.getLogger(__name__)
    path = Path(path)
    if not path.exists():
        log.warning(f"Artifact not found [{description}]: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if description:
        log.debug(f"Artifact loaded [{description}]: {path}")
    return data