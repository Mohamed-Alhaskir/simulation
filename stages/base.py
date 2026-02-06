"""Base class for all pipeline stages."""

import logging
from abc import ABC, abstractmethod


class BaseStage(ABC):
    """Abstract base for a pipeline stage."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, ctx: dict) -> dict:
        """
        Execute the stage.

        Parameters
        ----------
        ctx : dict
            Pipeline context containing session info, config, and accumulated
            artifacts from previous stages.

        Returns
        -------
        dict
            Updated context with this stage's artifacts added.
        """
        ...

    def _get_stage_config(self, key: str) -> dict:
        """Retrieve this stage's section from the global config."""
        return self.config.get(key, {})
