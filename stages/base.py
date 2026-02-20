"""Base class for all pipeline stages."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import json
class BaseStage(ABC):
    """Abstract base for a pipeline stage."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _resolve_artifact(artifact):
        """Re-load a slimmed artifact from disk if it has been replaced by its path string."""
        if isinstance(artifact, str) and Path(artifact).exists():
            with open(artifact, encoding="utf-8") as f:
                return json.load(f)
        return artifact
        ...

    def _get_stage_config(self, key: str) -> dict:
        """Retrieve this stage's section from the global config."""
        return self.config.get(key, {})
