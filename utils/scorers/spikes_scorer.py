"""
SPIKES Structural Annotation Scorer
====================================
Evaluates bad-news delivery using the SPIKES protocol (Baile et al., 2000).
Detects presence/sequence of six steps and cites evidence from transcript.

SPIKES steps:
  S1  Setting up
  P   Patient's perception
  I   Invitation
  K   Knowledge delivery
  E   Empathic response
  S2  Strategy and summary
"""

import json
import logging
from pathlib import Path
from typing import Any

from utils.llm_backends import LLMBackend
from utils.artifact_io import save_artifact


# SPIKES step definitions (Baile et al., 2000)
SPIKES_STEPS: list[dict[str, str]] = [
    {
        "id": "S1",
        "name": "Setting up",
        "description": (
            "Clinician arranges privacy, invites significant others if appropriate, "
            "sits down, establishes eye contact, manages interruptions and time."
        ),
    },
    {
        "id": "P",
        "name": "Patient's perception",
        "description": (
            "Before telling, clinician uses open-ended questions to establish "
            "what the patient already knows (e.g. 'What have you been told so far?')."
        ),
    },
    {
        "id": "I",
        "name": "Invitation",
        "description": (
            "Clinician checks how much detail the patient wants "
            "(e.g. 'Would you like me to explain the results in detail?')."
        ),
    },
    {
        "id": "K",
        "name": "Knowledge - delivering information",
        "description": (
            "Clinician gives a warning phrase before bad news, delivers in plain "
            "language, in small chunks, checks understanding periodically, "
            "avoids false reassurance."
        ),
    },
    {
        "id": "E",
        "name": "Empathic response",
        "description": (
            "Clinician observes patient emotion, names it, identifies the reason, "
            "makes an empathic connecting statement. Uses validating and exploratory "
            "responses. Allows silence. Does not rush past emotion."
        ),
    },
    {
        "id": "S2",
        "name": "Strategy and summary",
        "description": (
            "Clinician checks patient is ready to discuss next steps, presents "
            "treatment options or plan, checks for misunderstanding, invites "
            "patient questions, summarises the discussion."
        ),
    },
]

_SPIKES_NULL_STUB: dict = {
    "skipped": True,
    "reason": "SPIKES not applicable for this scenario type",
    "steps": [],
    "sequence_correct": None,
    "sequence_note": "SPIKES pass was not run for this scenario.",
    "overall_spikes_note": "SPIKES pass was not run for this scenario.",
}


class SpikesScorer:
    """Scorer for SPIKES protocol compliance."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def score(
        self,
        context: dict,
        backend: LLMBackend,
        cfg: dict,
        strictness_preamble: str = "",
        output_dir: Path = None,
    ) -> dict:
        """
        Score SPIKES protocol compliance.

        Args:
            context: Assembled context dict with diarized_transcript, verbal_features, etc.
            backend: LLM backend instance
            cfg: LLM config dict
            strictness_preamble: Strictness calibration text to inject into prompt
            output_dir: Directory to save prompt and raw output (optional)

        Returns:
            SPIKES annotation dict with steps, sequence validation, and evidence
        """
        self.logger.info("SPIKES structural annotation")

        # Build prompt
        spikes_prompt = self._build_spikes_prompt(context, strictness_preamble)

        if output_dir:
            (Path(output_dir) / "spikes_prompt.txt").write_text(
                spikes_prompt, encoding="utf-8"
            )

        # Generate
        spikes_raw = backend.generate(spikes_prompt, cfg)

        if output_dir:
            (Path(output_dir) / "spikes_raw_output.txt").write_text(
                spikes_raw, encoding="utf-8"
            )

        # Parse
        spikes_annotation = self._parse_spikes_output(spikes_raw)

        # Validate
        self._validate_spikes(spikes_annotation)

        n_present = sum(
            1 for s in spikes_annotation.get("steps", []) if s.get("present")
        )
        self.logger.info(
            f"SPIKES annotation complete: "
            f"{n_present}/{len(SPIKES_STEPS)} steps identified"
        )

        return spikes_annotation

    def _build_spikes_prompt(self, context: dict, strictness_preamble: str) -> str:
        """Build SPIKES evaluation prompt from context."""
        transcript_text = self._format_transcript(context["diarized_transcript"])
        interaction_text = json.dumps(
            context["verbal_features"], indent=2, ensure_ascii=False
        )
        phases_text = json.dumps(
            context["conversation_phases"], indent=2, ensure_ascii=False
        )

        if context.get("video_nvb"):
            video_nvb_section = (
                "Nutze diese Metriken als ergänzende Belege für S1 (Augenkontakt, "
                "Sitzen auf Augenhöhe) und E (nonverbale Empathiereaktionen).\n\n"
                + json.dumps(context["video_nvb"], indent=2, ensure_ascii=False)
                + "\n"
            )
        else:
            video_nvb_section = (
                "_Videoanalyse nicht verfügbar. "
                "S1 und E ausschließlich auf Basis des Transkripts bewerten._\n"
            )

        # Build items_block from SPIKES_STEPS
        items_block = self._build_items_block()

        return self._render_template(
            "spikes_prompt.j2",
            transcript=transcript_text,
            interaction=interaction_text,
            conversation_phases=phases_text,
            video_nvb_section=video_nvb_section,
            strictness_preamble=strictness_preamble,
            items_block=items_block,
        )

    def _build_items_block(self) -> str:
        """Format SPIKES steps as a text block for the prompt."""
        lines = []
        for i, step in enumerate(SPIKES_STEPS, 1):
            lines.append(f"**{i}. {step['id']} — {step['name']}**")
            lines.append(f"{step['description']}\n")
        return "\n".join(lines)

    def _parse_spikes_output(self, raw: str) -> dict:
        """Parse SPIKES LLM output using generic JSON parser."""
        import re

        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Try to find and parse JSON
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract first JSON object
            match = re.search(r"\{[^{}]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        # Failed to parse
        self.logger.error("Failed to parse SPIKES LLM output as JSON")
        return {"parse_error": True, "raw_output": raw}

    def _validate_spikes(self, annotation: dict) -> None:
        """Validate SPIKES annotation structure."""
        if annotation.get("parse_error"):
            self.logger.error(
                "SPIKES annotation has parse errors — LUCAS Pass 2 will "
                "proceed without SPIKES context."
            )
            return

        steps = annotation.get("steps", [])
        if len(steps) != len(SPIKES_STEPS):
            self.logger.warning(
                f"SPIKES: expected {len(SPIKES_STEPS)} steps, got {len(steps)}"
            )

        expected_ids = {s["id"] for s in SPIKES_STEPS}
        returned_ids = {s.get("id") for s in steps}
        missing = expected_ids - returned_ids
        if missing:
            self.logger.warning(f"SPIKES: missing step ids: {missing}")

        absent = [s["id"] for s in steps if not s.get("present")]
        if absent:
            self.logger.info(f"SPIKES: steps not identified: {absent}")

        if not annotation.get("sequence_correct"):
            self.logger.warning(
                f"SPIKES sequencing issue: {annotation.get('sequence_note', '')}"
            )

    # ────────────────────────────────────────────────────────────────
    # Shared helper methods (duplicated from s5_analysis to avoid import)
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_transcript(segments: list[dict]) -> str:
        """Format transcript segments for prompt inclusion."""
        return "\n".join(
            f"[{s['speaker']}] ({s['start']:.1f}-{s['end']:.1f}s): {s['text']}"
            for s in segments
        )

    @staticmethod
    def _render_template(template_name: str, **kwargs) -> str:
        """Render Jinja2 template."""
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        templates_dir = Path(__file__).parent.parent.parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )
        return env.get_template(template_name).render(**kwargs)
