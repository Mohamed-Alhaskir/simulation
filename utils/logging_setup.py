"""Logging configuration for the pipeline."""

import logging
import sys


def setup_logging(level: str = "INFO"):
    """Configure structured logging to stdout."""
    log_format = (
        "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Suppress noisy third-party loggers
    for name in ["httpx", "httpcore", "urllib3", "faster_whisper"]:
        logging.getLogger(name).setLevel(logging.WARNING)
