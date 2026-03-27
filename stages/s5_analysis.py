"""
Stage 5: LLM-Based Analysis & Rating
======================================
Generic instrument loop driven by scenario_catalog.json.

Every scenario defines a list of instruments:
  "Diabetes":      ["SPIKES", "LUCAS", "Diabetes_CC"]
  "LP_Aufklaerung":["LUCAS", "GSLP", "LP_Aufklaerung"]
  "Bauchschmerzen":["LUCAS"]

For each instrument:
  scorer = InstrumentScorer(f"instruments/{inst_id}.json", backend, cfg)
  results[inst_id] = scorer.score(context)

SPIKES runs first (when present) so its output is available as spikes_annotation
for the LUCAS pass that scores empathy/responsiveness items F and G.
"""

import json
from pathlib import Path

from stages.base import BaseStage
from utils.scorers.instrument_scorer import InstrumentScorer
from utils.llm_backends import get_llm_backend, LLMBackend
from utils.artifact_io import save_artifact


_DEFAULT_SCENARIO_CONFIG: dict = {
    "instruments": [],
    "display_name": "Unbekanntes Szenario",
}


def _load_scenario_catalog() -> dict[str, dict]:
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
    """Stage 5: Generic instrument loop."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._llm_backend: LLMBackend | None = None

    def _initialize_backend(self, cfg: dict) -> LLMBackend:
        if self._llm_backend is None:
            backend_name = cfg.get("backend", "llama_cpp")
            self._llm_backend = get_llm_backend(
                backend_name, cfg, logger_instance=self.logger
            )
        return self._llm_backend

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("llm")
        output_dir = Path(ctx["output_base"]) / "05_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Resolve scenario ────────────────────────────────────────────

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
                    "Running with empty instrument list."
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
        instruments = scenario_cfg.get("instruments", [])

        self.logger.info(
            f"Scenario: '{scenario_id}' "
            f"({scenario_cfg.get('display_name', 'unknown')}) | "
            f"instruments={instruments}"
        )

        # ── 2. Assemble context ────────────────────────────────────────────

        context = self._build_context(ctx)
        context["scenario_context"] = scenario_cfg.get("scenario_context", {})
        context["scenario_display_name"] = scenario_cfg.get("display_name", "")
        self.logger.info(
            f"Context assembled: "
            f"{len(context['diarized_transcript'])} transcript segments, "
            f"{context['verbal_features']['summary'].get('total_turns', '?')} turns, "
            f"video_nvb={'present' if context.get('video_nvb') else 'absent'}"
        )
        context_path = output_dir / "assembled_context.json"
        save_artifact(
            context, context_path,
            description="assembled_context", logger_instance=self.logger
        )

        backend = self._initialize_backend(cfg)

        # ── 3. Run instruments in order ────────────────────────────────────

        results: dict[str, dict] = {}
        instruments_dir = Path(__file__).parent.parent / "instruments"

        for inst_id in instruments:
            instrument_path = instruments_dir / f"{inst_id}.json"
            if not instrument_path.exists():
                self.logger.error(
                    f"Instrument file not found: {instrument_path} — skipping"
                )
                continue

            self.logger.info(f"Scoring instrument: {inst_id}")
            try:
                scorer = InstrumentScorer(
                    instrument_path=instrument_path,
                    backend=backend,
                    cfg=cfg,
                    logger=self.logger,
                )
                result = scorer.score(context)
                results[inst_id] = result

                # Save per-instrument artifact
                inst_path = output_dir / f"{inst_id.lower()}_result.json"
                save_artifact(
                    result, inst_path,
                    description=f"{inst_id}_result", logger_instance=self.logger
                )

                raw = result.get("summary", {}).get("raw_score", "?")
                max_s = result.get("summary", {}).get("max_possible_score", "?")
                self.logger.info(
                    f"{inst_id} complete: {raw}/{max_s}"
                )

            except Exception as e:
                self.logger.error(f"Instrument {inst_id} failed: {e}", exc_info=True)
                results[inst_id] = {"error": str(e), "instrument_id": inst_id}

        # ── 4. Combine into final analysis artifact ────────────────────────

        analysis = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_cfg.get("display_name", ""),
            "instruments_run": list(results.keys()),
            "results": results,
        }

        # Backward-compatible flattened fields for downstream scripts
        if "LUCAS" in results:
            lucas = results["LUCAS"]
            lucas_items = lucas.get("items", [])
            analysis["lucas_analysis"] = {
                "lucas_items": [
                    {**item, "item": item.get("id")}
                    for item in lucas_items
                ],
                "total_score": lucas.get("summary", {}).get("raw_score", 0),
                "overall_summary": lucas.get("overall_note", ""),
            }
            analysis["lucas_total_score"] = lucas.get("summary", {}).get("raw_score", 0)
            analysis["lucas_max_score"] = lucas.get("summary", {}).get("max_possible_score", 18)

        if "SPIKES" in results:
            spikes = results["SPIKES"]
            analysis["spikes_annotation"] = spikes
            analysis["spikes_was_run"] = True
        else:
            analysis["spikes_was_run"] = False

        # Clinical content (first non-LUCAS, non-SPIKES instrument)
        cc_instruments = [
            k for k in results
            if k not in ("LUCAS", "SPIKES")
        ]
        if cc_instruments:
            # Merge all clinical instruments into a single clinical_content entry
            cc_items = []
            cc_raw = 0
            cc_max = 0
            for cid in cc_instruments:
                cr = results[cid]
                cc_items.extend(cr.get("items", []))
                cc_raw += cr.get("summary", {}).get("raw_score", 0)
                cc_max += cr.get("summary", {}).get("max_possible_score", 0)
            analysis["clinical_content"] = {
                "items": cc_items,
                "raw_score": cc_raw,
                "max_possible_score": cc_max,
                "normalised_score_pct": (
                    round(cc_raw / cc_max * 100, 1) if cc_max > 0 else 0.0
                ),
            }

        analysis_path = output_dir / "analysis.json"
        save_artifact(
            analysis, analysis_path,
            description="analysis", logger_instance=self.logger
        )

        ctx["artifacts"].update({
            "assembled_context": context,
            "assembled_context_path": str(context_path),
            "analysis": analysis,
            "analysis_path": str(analysis_path),
            # Backward-compatible keys
            "lucas_analysis": analysis.get("lucas_analysis", {}),
            "spikes_annotation": analysis.get("spikes_annotation", {}),
            "spikes_was_run": analysis.get("spikes_was_run", False),
            "clinical_content": analysis.get("clinical_content", {}),
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

    def cleanup(self) -> None:
        if self._llm_backend is not None:
            self._llm_backend.cleanup()
            self._llm_backend = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
