"""
Freeze Manifest
================
Generates and verifies the cryptographic freeze manifest that locks
the entire pipeline configuration before confirmatory data analysis.

The manifest includes:
- Pipeline version
- Git commit hash
- Model identifiers and versions
- Prompt template hashes
- Configuration hash
- Random seeds
- Timestamp of freeze
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


class FreezeManifest:
    """Immutable record of the pipeline state at freeze time."""

    def __init__(self, pipeline_version: str, config: dict):
        self.pipeline_version = pipeline_version
        self.config = config
        self._manifest = self._build()

    def _build(self) -> dict:
        return {
            "pipeline_version": self.pipeline_version,
            "git_commit": self._get_git_commit(),
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "seeds": {
                "global": self.config.get("pipeline", {}).get("seed", None),
                "llm": self.config.get("llm", {}).get("seed", None),
            },
            "models": {
                "asr": {
                    "engine": self.config.get("asr", {}).get("model_type"),
                    "model_name": self.config.get("asr", {}).get("model_name"),
                    "compute_type": self.config.get("asr", {}).get("compute_type"),
                    "beam_size": self.config.get("asr", {}).get("beam_size"),
                },
                "diarization": {
                    "model": self.config.get("asr", {})
                    .get("diarization", {})
                    .get("model"),
                },
                "llm": {
                    "backend": self.config.get("llm", {}).get("backend"),
                    "model_path": self.config.get("llm", {}).get("model_path"),
                    "temperature": self.config.get("llm", {}).get("temperature"),
                    "context_length": self.config.get("llm", {}).get("context_length"),
                },
            },
            "prompt_template_hash": self._hash_prompt_template(),
            "config_hash": self._hash_config(),
        }

    @staticmethod
    def _get_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None

    def _hash_prompt_template(self) -> str | None:
        template_path = self.config.get("llm", {}).get("prompt_template")
        if template_path and Path(template_path).exists():
            content = Path(template_path).read_bytes()
            return hashlib.sha256(content).hexdigest()
        return None

    def _hash_config(self) -> str:
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def digest(self) -> str:
        """Return a single SHA-256 digest of the entire manifest."""
        manifest_str = json.dumps(self._manifest, sort_keys=True)
        return hashlib.sha256(manifest_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {**self._manifest, "manifest_digest": self.digest()}

    def save(self, path: str):
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_and_verify(cls, path: str, current_config: dict, pipeline_version: str) -> bool:
        """
        Load a saved manifest and verify it matches the current pipeline state.
        Returns True if the pipeline has NOT changed since freeze.
        """
        with open(path, "r") as f:
            saved = json.load(f)

        current = cls(pipeline_version, current_config)
        current_dict = current.to_dict()

        # Compare critical fields
        mismatches = []
        for key in ["pipeline_version", "config_hash", "prompt_template_hash"]:
            if saved.get(key) != current_dict.get(key):
                mismatches.append(key)

        if saved.get("models") != current_dict.get("models"):
            mismatches.append("models")

        if saved.get("seeds") != current_dict.get("seeds"):
            mismatches.append("seeds")

        if mismatches:
            print(f"⚠ FREEZE VIOLATION — changed fields: {mismatches}")
            return False

        print("✓ Freeze manifest verified — pipeline unchanged.")
        return True
