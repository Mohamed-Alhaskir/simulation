"""
Stage 5: LLM-Based Analysis & Rating
======================================
Orchestrates three assessment frameworks:
  - SPIKES (bad-news delivery, conditional)
  - LUCAS (communication skills, all scenarios)
  - Clinical Content (medical knowledge, scenario-specific)

Each framework is evaluated by its own scorer class:
  - SpikesScorer: SPIKES protocol compliance
  - LucasMultipassScorer: LUCAS items A-J (7-pass multipass)
  - ClinicalContentScorer: Scenario-specific clinical modules

This file is a thin orchestrator that coordinates the three scorers.
"""

import json
from pathlib import Path

from stages.base import BaseStage
from utils.scorers.spikes_scorer import SpikesScorer, _SPIKES_NULL_STUB
from utils.scorers.clinical_content_scorer import ClinicalContentScorer
from utils.llm_backends import get_llm_backend, LLMBackend
from utils.artifact_io import save_artifact
from utils.assessment_schemas import LUCAS_ITEMS, LUCAS_MAX_SCORE


# ──────────────────────────────────────────────────────────────────────
# Scenario registry — loaded from templates/scenario_catalog.json
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_SCENARIO_CONFIG: dict = {
    "uses_spikes": False,
    "module_ids": [],
    "display_name": "Unbekanntes Szenario",
}
# LUCAS_ITEMS and LUCAS_MAX_SCORE are imported from assessment_schemas (see imports above)


def _load_scenario_catalog() -> dict[str, dict]:
    """Load scenario metadata from templates/scenario_catalog.json."""
    catalog_path = Path(__file__).parent.parent / "templates" / "scenario_catalog.json"
    if not catalog_path.exists():
        return {}
    try:
        with open(catalog_path) as f:
            data = json.load(f)
        return data.get("scenarios", {})
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠ Error loading scenario catalog: {e}")
        return {}


class LLMAnalysisStage(BaseStage):
    """Stage 5: Orchestrate SPIKES, LUCAS, and Clinical Content scoring."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._llm_backend: LLMBackend | None = None

    def _initialize_backend(self, cfg: dict) -> LLMBackend:
        """Initialize the LLM backend (shared across all scorers)."""
        if self._llm_backend is None:
            backend_name = cfg.get("backend", "llama_cpp")
            self._llm_backend = get_llm_backend(
                backend_name, cfg, logger_instance=self.logger
            )
        return self._llm_backend

    def run(self, ctx: dict) -> dict:
        """
        Main orchestrator: Run SPIKES → LUCAS → Clinical Content.

        Args:
            ctx: Pipeline context with artifacts from previous stages

        Returns:
            Updated context with analysis results
        """
        cfg = self._get_stage_config("llm")
        strictness = int(cfg.get("strictness", 2))
        strictness_preamble = self._strictness_preamble(strictness)
        output_dir = Path(ctx["output_base"]) / "05_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ────────────────────────────────────────────────────────────
        # 1. Resolve scenario config
        # ────────────────────────────────────────────────────────────

        _metadata = self._resolve_artifact(
            ctx.get("artifacts", {}).get("metadata")
        ) or {}
        scenario_id = _metadata.get("scenario", {}).get("id", "").strip()

        if not scenario_id:
            from utils.scenario_map import resolve_scenario_id

            session_id = ctx.get("session_id", "")
            scenario_id = resolve_scenario_id(session_id)
            if scenario_id:
                self.logger.info(
                    f"scenario_id resolved from mapping file: "
                    f"'{session_id}' → '{scenario_id}'"
                )
            else:
                self.logger.warning(
                    f"Could not resolve scenario_id for session '{session_id}'. "
                    "Running with _DEFAULT_SCENARIO_CONFIG (SPIKES off, no modules)."
                )
        else:
            from utils.scenario_map import _canonicalise

            canonical = _canonicalise(scenario_id)
            if canonical != scenario_id:
                self.logger.info(
                    f"scenario_id normalised: '{scenario_id}' → '{canonical}'"
                )
                scenario_id = canonical

        scenario_catalog = _load_scenario_catalog()
        scenario_cfg = scenario_catalog.get(scenario_id, _DEFAULT_SCENARIO_CONFIG)
        uses_spikes = scenario_cfg["uses_spikes"]

        self.logger.info(
            f"Scenario: '{scenario_id}' "
            f"({scenario_cfg.get('display_name', 'unknown')}) | "
            f"SPIKES={'enabled' if uses_spikes else 'skipped'} | "
            f"modules={scenario_cfg['module_ids']}"
        )

        # ────────────────────────────────────────────────────────────
        # 2. Assemble and save context (audit record)
        # ────────────────────────────────────────────────────────────

        context = self._build_context(ctx)
        self.logger.info(
            f"Context assembled: "
            f"{len(context['diarized_transcript'])} transcript segments, "
            f"{context['verbal_features']['summary']['total_turns']} turns, "
            f"video_nvb={'present' if context.get('video_nvb') else 'absent'}"
        )
        context_path = output_dir / "assembled_context.json"
        save_artifact(
            context, context_path,
            description="assembled_context", logger_instance=self.logger
        )

        backend = self._initialize_backend(cfg)

        # ────────────────────────────────────────────────────────────
        # 3. Pass 1 — SPIKES (conditional)
        # ────────────────────────────────────────────────────────────

        if uses_spikes:
            spikes_scorer = SpikesScorer(logger=self.logger)
            spikes_annotation = spikes_scorer.score(
                context=context,
                backend=backend,
                cfg=cfg,
                strictness_preamble=strictness_preamble,
                output_dir=output_dir,
            )
        else:
            self.logger.info(
                f"Pass 1: SPIKES skipped "
                f"(scenario '{scenario_id}' is not a bad-news delivery)"
            )
            spikes_annotation = dict(_SPIKES_NULL_STUB)
            spikes_annotation["scenario_id"] = scenario_id

        spikes_path = output_dir / "spikes_annotation.json"
        save_artifact(
            spikes_annotation, spikes_path,
            description="spikes_annotation", logger_instance=self.logger
        )

        # ────────────────────────────────────────────────────────────
        # 4. Pass 2 — LUCAS (multi-pass scorer)
        # ────────────────────────────────────────────────────────────

        self.logger.info("Pass 2: LUCAS scoring (multipass)")
        from utils.scorers.lucas_multipass import LucasMultipassScorer

        lucas_scorer = LucasMultipassScorer(backend=backend, cfg=cfg, strictness=strictness)
        lucas_context = dict(context)
        lucas_context["spikes_annotation"] = spikes_annotation
        lucas_analysis = lucas_scorer.score(lucas_context)
        self._validate_lucas(lucas_analysis)

        lucas_path = output_dir / "lucas_analysis.json"
        save_artifact(
            lucas_analysis, lucas_path,
            description="lucas_analysis", logger_instance=self.logger
        )
        total = lucas_analysis.get("total_score", 0)
        self.logger.info(f"LUCAS scoring complete: {total}/{LUCAS_MAX_SCORE}")

        # ────────────────────────────────────────────────────────────
        # 5. Pass 3 — Clinical Content (one LLM call per module)
        # ────────────────────────────────────────────────────────────

        cc_scorer = ClinicalContentScorer(logger=self.logger)
        module_ids = scenario_cfg.get("module_ids", [])
        cc_scored = cc_scorer.score(
            context=context,
            backend=backend,
            cfg=cfg,
            module_ids=module_ids,
            scenario_id=scenario_id,
            strictness_preamble=strictness_preamble,
            output_dir=output_dir,
        )

        cc_path = output_dir / "clinical_content.json"
        save_artifact(
            cc_scored, cc_path,
            description="clinical_content", logger_instance=self.logger
        )

        # ────────────────────────────────────────────────────────────
        # 6. Combine into final analysis artifact
        # ────────────────────────────────────────────────────────────

        analysis = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_cfg.get("display_name", ""),
            "passes_run": {
                "spikes": uses_spikes,
                "lucas": True,
                "clinical_content": not cc_scored.get("skipped", False),
            },
            "spikes_annotation": spikes_annotation,
            "lucas_analysis": lucas_analysis,
            "lucas_total_score": total,
            "lucas_max_score": LUCAS_MAX_SCORE,
            "clinical_content": cc_scored,
        }

        analysis_path = output_dir / "analysis.json"
        save_artifact(
            analysis, analysis_path,
            description="analysis", logger_instance=self.logger
        )

        ctx["artifacts"].update({
            "assembled_context": context,
            "assembled_context_path": str(context_path),
            "spikes_annotation": spikes_annotation,
            "spikes_annotation_path": str(spikes_path),
            "spikes_was_run": uses_spikes,
            "lucas_analysis": lucas_analysis,
            "lucas_analysis_path": str(lucas_path),
            "clinical_content": cc_scored,
            "clinical_content_path": str(cc_path),
            "analysis": analysis,
            "analysis_path": str(analysis_path),
        })

        return ctx

    def _build_context(self, ctx: dict) -> dict:
        """Assemble context dict from pipeline artifacts."""
        transcript = self._resolve_artifact(ctx["artifacts"]["transcript"])
        features = self._resolve_artifact(ctx["artifacts"]["features"])
        verbal = features["verbal"]

        context: dict = {
            "diarized_transcript": [
                {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in transcript
            ],
            "verbal_features": {
                "summary": verbal["summary"],
                "pause_details": verbal["pauses"][:10],
                "interruption_details": verbal["interruptions"][:10],
            },
            "conversation_phases": features.get("phases", []),
            "patient_vitals": features.get("vitals"),
        }

        video_features = self._resolve_artifact(
            ctx["artifacts"].get("video_features")
        )
        if video_features:
            context["video_nvb"] = video_features
        else:
            self.logger.info(
                "No video NVB features in ctx — video analysis stage "
                "was skipped or disabled."
            )
            context["video_nvb"] = None

        return context

    def _validate_lucas(self, analysis: dict) -> None:
        """Validate LUCAS analysis output structure and scoring."""
        if analysis.get("parse_error"):
            self.logger.error("LUCAS analysis has parse errors.")
            return

        raw_items = analysis.get("lucas_items") or analysis.get("items", [])
        if not raw_items:
            self.logger.warning("LUCAS: no items found in output")
            return

        items_out: dict[str, dict] = {}
        for out in raw_items:
            item_id = out.get("item") or out.get("id")
            if item_id:
                items_out[item_id] = out

        total_computed = 0
        valid = True

        for rubric_item in LUCAS_ITEMS:
            item_id = rubric_item["id"]
            if item_id not in items_out:
                self.logger.warning(f"LUCAS: item '{item_id}' missing from output")
                valid = False
                continue

            out = items_out[item_id]
            score_raw = (
                out.get("rating") if out.get("rating") is not None
                else out.get("score")
            )
            allowed = [int(k) for k in rubric_item["scale"]["labels"].keys()]

            try:
                score = int(score_raw)
            except (TypeError, ValueError):
                self.logger.warning(
                    f"LUCAS item '{item_id}': score '{score_raw}' is not an integer"
                )
                valid = False
                continue

            if score not in allowed:
                self.logger.warning(
                    f"LUCAS item '{item_id}': score {score} not in {allowed}"
                )
                valid = False
            else:
                total_computed += score

            if not out.get("evidence"):
                self.logger.warning(f"LUCAS item '{item_id}': no evidence provided")

        claimed_raw = analysis.get("total_score")
        try:
            claimed = int(claimed_raw)
        except (TypeError, ValueError):
            claimed = None

        if claimed is not None and claimed != total_computed:
            self.logger.warning(
                f"LUCAS: claimed total {claimed} != computed {total_computed}. "
                "Overwriting."
            )
        analysis["total_score"] = total_computed

        if valid:
            self.logger.info(f"LUCAS validation passed: {total_computed}/{LUCAS_MAX_SCORE}")

    @staticmethod
    def _strictness_preamble(level: int) -> str:
        """Return a calibration block injected at the top of every prompt."""
        if level == 1:
            return (
                "SCORING CALIBRATION — LENIENT (Level 1/3)\n"
                "Applies to RATING DECISIONS ONLY. Follow all pass-specific search instructions unchanged.\n"
                "When deciding a rating: apply benefit of the doubt.\n"
                "- Partial or implicit evidence is sufficient for a higher rating.\n"
                "- Borderline cases: score UP.\n"
                "- A criterion counts as met if the evidence reasonably supports it, even if not fully explicit.\n\n"
            )
        if level == 3:
            return (
                "SCORING CALIBRATION — STRICT (Level 3/3)\n"
                "Applies to RATING DECISIONS ONLY. Follow all pass-specific search instructions unchanged.\n"
                "When deciding a rating: apply a high evidentiary standard.\n"
                "- Only explicit, unambiguous, clearly observable behaviour receives credit.\n"
                "- No inference; ambiguous cases score DOWN.\n"
                "- This level reflects a high-stakes examination standard.\n\n"
            )
        # level == 2 (standard) — no preamble needed
        return ""

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._llm_backend is not None:
            self._llm_backend.cleanup()
            self._llm_backend = None
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
