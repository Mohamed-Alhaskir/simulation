"""
Stage 5: LLM-Based Analysis & Rating
======================================
Three-pass LLM inference against LUCAS, SPIKES, and Clinical Content frameworks.

Pass 1 — SPIKES structural annotation  [scenario-conditional]
    Only runs for bad_news_delivery scenarios (e.g. Diabetes).
    Reads the transcript and identifies where each of the six SPIKES steps
    occurred, flags absent or mis-sequenced steps, and cites specific turns
    as evidence. Output: spikes_annotation.json.

    Skipped for consent_conversation and history_consultation scenarios.
    When skipped, an empty stub is written so downstream passes remain stable.

Pass 2 — LUCAS scoring
    Scores all ten LUCAS items (A-J) using the transcript, verbal features,
    video NVB features, and (when available) the SPIKES annotation from Pass 1.
    Output: lucas_analysis.json.

    LUCAS applies to ALL scenario types.

Pass 3 — Clinical Content scoring  [scenario-conditional]
    Scores clinical content exclusively against the scenario-specific module(s)
    declared in _SCENARIO_CONFIG and loaded from
    templates/clinical_modules/<scenario_id>.json.

    Scenarios may declare multiple module files (e.g. LP has a §630e structural
    module AND a clinical content module). All module items are merged into a
    single prompt. Output: clinical_content.json.

    If no module is declared or found for the scenario, Pass 3 is skipped and
    a null stub is written to clinical_content.json.

LUCAS scoring rubric (University of Liverpool):
  A  Greeting and introduction     0 / 1
  B  Identity check                0 / 1
  C  Audibility and clarity        0 / 1 / 2
  D  Non-verbal behaviour          0 / 1 / 2
  E  Questions, prompts, expl.     0 / 1 / 2
  F  Empathy and responsiveness    0 / 1 / 2
  G  Clarification & summarising   0 / 1 / 2
  H  Consulting style & org.       0 / 1 / 2
  I  Professional behaviour        0 / 2 (no borderline)
  J  Professional spoken conduct   0 / 2 (no borderline)
  Maximum total: 18

SPIKES steps (Baile et al., 2000):
  S1  Setting up
  P   Patient's perception
  I   Invitation
  K   Knowledge delivery
  E   Empathic response
  S2  Strategy and summary

Scenario registry:
  Each scenario_id maps to a config dict with:
    uses_spikes (bool)      — whether Pass 1 runs
    module_ids  (list[str]) — ordered list of clinical module filenames to load
                              (without .json extension); merged in order
"""

import json
import re
from pathlib import Path
from typing import Any

from stages.base import BaseStage
from utils.llm_backends import get_llm_backend, LLMBackend
from utils.artifact_io import save_artifact, load_artifact


# ------------------------------------------------------------------
# Scenario registry
# ------------------------------------------------------------------
# Declares which passes apply to each scenario_id and which clinical
# module files to load (merged in listed order).
#
# uses_spikes:
#   True  → bad_news_delivery: SPIKES annotation runs as Pass 1 and its
#            output feeds into LUCAS (items F, G, H) and the analysis.
#   False → consent_conversation / history_consultation: SPIKES is skipped;
#            a null stub is written to spikes_annotation.json so downstream
#            code and report generation never need to branch on file existence.
#
# module_ids:
#   List of filenames (without .json) under templates/clinical_modules/.
#   Multiple entries are merged into a single item list for Pass 3.
#   Empty list → Pass 3 is skipped entirely (no module, nothing to score).
# ------------------------------------------------------------------

_SCENARIO_CONFIG: dict[str, dict] = {
    "Diabetes": {
        "uses_spikes": True,
        "module_ids": ["Diabetes"],
        "display_name": "Diagnoseübermittlung Diabetes Mellitus Typ 1",
    },
    "LP_Aufklaerung": {
        "uses_spikes": False,
        "module_ids": ["LP_Aufklaerung"],
        "display_name": "LP-Aufklärung bei V.a. Meningitis",
    },
    "Bauchschmerzen": {
        "uses_spikes": False,
        "module_ids": ["Bauchschmerzen"],
        "display_name": "Bauchschmerzen – Anamnese bei akutem abdominalem Schmerz",
    },
}

# Fallback config for unknown / unregistered scenario_ids.
# No SPIKES, no module → Pass 3 will be skipped with a warning.
_DEFAULT_SCENARIO_CONFIG: dict = {
    "uses_spikes": False,
    "module_ids": [],
    "display_name": "Unbekanntes Szenario",
}

# Null SPIKES stub — written when Pass 1 is skipped so downstream code
# can always load spikes_annotation.json without branching.
_SPIKES_NULL_STUB: dict = {
    "skipped": True,
    "reason": "SPIKES not applicable for this scenario type",
    "steps": [],
    "sequence_correct": None,
    "sequence_note": "SPIKES pass was not run for this scenario.",
    "overall_spikes_note": "SPIKES pass was not run for this scenario.",
}

# Null Clinical Content stub — written when Pass 3 is skipped (no module
# declared for the scenario) so downstream report generation never needs
# to branch on file existence.
_CC_NULL_STUB: dict = {
    "skipped": True,
    "reason": "No clinical content module declared for this scenario",
    "items": [],
    "raw_score": 0,
    "max_applicable_score": 0,
    "normalised_score_pct": None,
    "category_scores_pct": {},
    "critical_misses": [],
    "critical_false_positives": [],
    "has_critical_miss": False,
    "overall_clinical_note": "",
    "_source_modules": [],
}


# ------------------------------------------------------------------
# LUCAS item definitions
# ------------------------------------------------------------------

LUCAS_ITEMS: list[dict[str, Any]] = [
    {
        "id": "A",
        "name": "Greeting and introduction",
        "section": "introductions",
        "description": (
            "Competent: greets patient, states full name, states job title, "
            "provides brief explanation of why approaching the patient. "
            "Unacceptable: omission of any of these four elements."
        ),
        "scale": {"min": 0, "max": 1, "labels": {"0": "unacceptable", "1": "competent"}},
        "evidence_sources": ["transcript"],
    },
    {
        "id": "B",
        "name": "Identity check",
        "section": "introductions",
        "description": (
            "Competent: checks patient full name AND one other identifier "
            "(e.g. DOB, address). Unacceptable: omission of either element."
        ),
        "scale": {"min": 0, "max": 1, "labels": {"0": "unacceptable", "1": "competent"}},
        "evidence_sources": ["transcript"],
    },
    {
        "id": "C",
        "name": "Audibility and clarity of speech",
        "section": "general",
        "description": (
            "Competent: speech consistently clear and audible. "
            "Borderline: occasional clarity issues. "
            "Unacceptable: consistently unclear or inaudible."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "1": "borderline", "2": "competent"}},
        "evidence_sources": ["transcript"],
        "note": "Primary evidence is transcript readability and coherence.",
    },
    {
        "id": "D",
        "name": "Non-verbal behaviour",
        "section": "general",
        "description": (
            "Includes eye-contact, positioning, posture, facial expressions, "
            "gestures and mannerisms. Competent: appropriate and sustained NVB. "
            "Borderline: inconsistent NVB. "
            "Unacceptable: NVB that undermines verbal communication."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "1": "borderline", "2": "competent"}},
        "evidence_sources": ["video_nvb"],
        "note": (
            "Use gaze_on_target rate, arm_openness_distribution, "
            "smile scores, and head movement from the video NVB section."
        ),
    },
    {
        "id": "E",
        "name": "Questions, prompts and/or explanations",
        "section": "general",
        "description": (
            "Includes (i) exploration of patient needs, feelings and concerns; "
            "(ii) comprehensibility of questions and explanations. "
            "NOTE: does NOT assess medical content of history-taking. "
            "Competent: effective open and closed questioning; clear explanations. "
            "Borderline: some exploration but inconsistent. "
            "Unacceptable: poor or absent exploration."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "1": "borderline", "2": "competent"}},
        "evidence_sources": ["transcript", "verbal_features"],
    },
    {
        "id": "F",
        "name": "Empathy and responsiveness",
        "section": "general",
        "description": (
            "Includes adaptation and sensitivity to patient needs. "
            "Competent: consistently empathic, adapts to patient cues. "
            "Borderline: some empathic responses but inconsistent. "
            "Unacceptable: little or no empathy; fails to respond to distress."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "1": "borderline", "2": "competent"}},
        "evidence_sources": ["transcript", "video_nvb", "spikes_annotation"],
        "note": "SPIKES step E (empathic response) is direct evidence for this item.",
    },
    {
        "id": "G",
        "name": "Clarification and summarising",
        "section": "general",
        "description": (
            "Includes elicitation of patient queries. "
            "Competent: regularly checks understanding, summarises, invites questions. "
            "Borderline: some checking but inconsistent. "
            "Unacceptable: no checking, no summarising, queries not elicited."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "1": "borderline", "2": "competent"}},
        "evidence_sources": ["transcript", "spikes_annotation"],
        "note": "SPIKES steps S2 (strategy/summary) and K (chunk and check) are relevant.",
    },
    {
        "id": "H",
        "name": "Consulting style and organisation",
        "section": "general",
        "description": (
            "Includes orderliness, balance of open and closed questions, "
            "and time management. "
            "Competent: well-structured, appropriate balance, good pacing. "
            "Borderline: some structure but disorganised at times. "
            "Unacceptable: chaotic, poor balance, rushed or poorly timed."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "1": "borderline", "2": "competent"}},
        "evidence_sources": ["transcript", "verbal_features", "conversation_phases", "spikes_annotation"],
        "note": (
            "SPIKES annotation provides sequencing evidence. "
            "Verbal features provide speaking ratio, turn balance, and pause data."
        ),
    },
    {
        "id": "I",
        "name": "Professional behaviour",
        "section": "professional_conduct",
        "description": (
            "Competent: courteous, kind, thoughtful behaviour throughout. "
            "Unacceptable: overly casual, disinterested, discourteous, or thoughtless."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "2": "competent"}},
        "evidence_sources": ["transcript", "video_nvb"],
        "note": "No borderline for this item. Score is 0 or 2 only.",
    },
    {
        "id": "J",
        "name": "Professional spoken/verbal conduct",
        "section": "professional_conduct",
        "description": (
            "Competent: remarks are (i) respectful AND (ii) avoid major inaccuracy "
            "AND (iii) within own competence AND (iv) reassurance is appropriate. "
            "Unacceptable: remarks are (i) disrespectful OR (ii) major inaccuracy OR "
            "(iii) outside own competence OR (iv) reassurance is inappropriate."
        ),
        "scale": {"min": 0, "max": 2, "labels": {"0": "unacceptable", "2": "competent"}},
        "evidence_sources": ["transcript"],
        "note": "No borderline for this item. Score is 0 or 2 only.",
    },
]

LUCAS_MAX_SCORE = 18


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


class LLMAnalysisStage(BaseStage):
    """
    Three-pass LLM assessment: SPIKES (conditional) → LUCAS → Clinical Content.

    Pass routing is controlled by the scenario registry (_SCENARIO_CONFIG).
    SPIKES only runs for bad_news_delivery scenarios. Clinical content loads
    one or more scenario-specific modules and merges them before scoring.
    """

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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("llm")
        output_dir = Path(ctx["output_base"]) / "05_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Resolve scenario config ─────────────────────────────────
        # Resolution order:
        #   a) metadata["scenario"]["id"]  — set by Stage 1 at ingest
        #   b) session_scenario_map.json   — fallback if Stage 1 didn't inject it
        #      (also handles the case where Stage 1 ran before the map existed)
        # In both cases the raw value is normalised to canonical casing so that
        # "diabetes" and "Diabetes" both resolve to the same registry entry.
        _metadata = self._resolve_artifact(
            ctx.get("artifacts", {}).get("metadata")
        ) or {}
        scenario_id = _metadata.get("scenario", {}).get("id", "").strip()

        if not scenario_id:
            # Stage 1 didn't inject a scenario_id — fall back to the mapping file.
            from utils.scenario_map import resolve_scenario_id
            session_id  = ctx.get("session_id", "")
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
            # Normalise whatever Stage 1 injected (guards against casing drift)
            from utils.scenario_map import _canonicalise
            canonical = _canonicalise(scenario_id)
            if canonical != scenario_id:
                self.logger.info(
                    f"scenario_id normalised: '{scenario_id}' → '{canonical}'"
                )
                scenario_id = canonical

        scenario_cfg = _SCENARIO_CONFIG.get(scenario_id, _DEFAULT_SCENARIO_CONFIG)
        uses_spikes  = scenario_cfg["uses_spikes"]

        self.logger.info(
            f"Scenario: '{scenario_id}' "
            f"({scenario_cfg.get('display_name', 'unknown')}) | "
            f"SPIKES={'enabled' if uses_spikes else 'skipped'} | "
            f"modules={scenario_cfg['module_ids']}"
        )

        # ── 2. Assemble and save context (audit record) ────────────────
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

        # ── 3. Pass 1 — SPIKES (conditional) ──────────────────────────
        if uses_spikes:
            self.logger.info("Pass 1: SPIKES structural annotation")
            spikes_prompt = self._build_spikes_prompt(context)
            (output_dir / "spikes_prompt.txt").write_text(
                spikes_prompt, encoding="utf-8"
            )
            spikes_raw = backend.generate(spikes_prompt, cfg)
            (output_dir / "spikes_raw_output.txt").write_text(
                spikes_raw, encoding="utf-8"
            )
            spikes_annotation = self._parse_output(spikes_raw, "spikes")
            self._validate_spikes(spikes_annotation)

            n_present = sum(
                1 for s in spikes_annotation.get("steps", []) if s.get("present")
            )
            self.logger.info(
                f"SPIKES annotation complete: "
                f"{n_present}/{len(SPIKES_STEPS)} steps identified"
            )
        else:
            # Write null stub so downstream code / report stage never needs
            # to branch on whether the file exists.
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

        # ── 4. Pass 2 — LUCAS (always runs) ───────────────────────────
        self.logger.info("Pass 2: LUCAS scoring")
        lucas_prompt = self._build_lucas_prompt(context, spikes_annotation)
        (output_dir / "lucas_prompt.txt").write_text(
            lucas_prompt, encoding="utf-8"
        )
        lucas_raw = backend.generate(lucas_prompt, cfg)
        (output_dir / "lucas_raw_output.txt").write_text(
            lucas_raw, encoding="utf-8"
        )
        lucas_analysis = self._parse_output(lucas_raw, "lucas")
        self._validate_lucas(lucas_analysis)

        lucas_path = output_dir / "lucas_analysis.json"
        save_artifact(
            lucas_analysis, lucas_path,
            description="lucas_analysis", logger_instance=self.logger
        )
        total = lucas_analysis.get("total_score", 0)
        self.logger.info(f"LUCAS scoring complete: {total}/{LUCAS_MAX_SCORE}")

        # ── 5. Pass 3 — Clinical Content ───────────────────────────────
        self.logger.info("Pass 3: Clinical Content scoring")

        # Load and merge all modules declared for this scenario.
        # Returns None if no modules are declared or none could be loaded.
        merged_module = self._load_and_merge_modules(
            scenario_cfg["module_ids"], scenario_id
        )

        if merged_module is None:
            # No module available — skip Pass 3 and write null stub.
            self.logger.warning(
                f"Pass 3: skipped — no clinical content module found "
                f"for scenario '{scenario_id}'"
            )
            cc_scored = dict(_CC_NULL_STUB)
            cc_scored["scenario_id"] = scenario_id
        else:
            cc_prompt = self._build_clinical_content_prompt(context, merged_module)
            (output_dir / "clinical_content_prompt.txt").write_text(
                cc_prompt, encoding="utf-8"
            )
            cc_raw = backend.generate(cc_prompt, cfg)
            (output_dir / "clinical_content_raw_output.txt").write_text(
                cc_raw, encoding="utf-8"
            )
            cc_parsed = self._parse_output(cc_raw, "clinical_content")
            cc_scored = self._score_clinical_content(cc_parsed, merged_module)
            self._validate_clinical_content(cc_scored)

        cc_path = output_dir / "clinical_content.json"
        save_artifact(
            cc_scored, cc_path,
            description="clinical_content", logger_instance=self.logger
        )

        if not cc_scored.get("skipped"):
            n_critical_miss = len(cc_scored.get("critical_misses", []))
            self.logger.info(
                f"Clinical content complete: "
                f"{cc_scored.get('normalised_score_pct', '?')}% "
                f"({cc_scored.get('raw_score', '?')}/"
                f"{cc_scored.get('max_applicable_score', '?')}), "
                f"{n_critical_miss} critical miss(es)"
            )

        # ── 6. Combine into final analysis artifact ────────────────────
        analysis = {
            "scenario_id":        scenario_id,
            "scenario_name":      scenario_cfg.get("display_name", ""),
            "passes_run": {
                "spikes":           uses_spikes,
                "lucas":            True,
                "clinical_content": not cc_scored.get("skipped", False),
            },
            "spikes_annotation":  spikes_annotation,
            "lucas_analysis":     lucas_analysis,
            "lucas_total_score":  total,
            "lucas_max_score":    LUCAS_MAX_SCORE,
            "clinical_content":   cc_scored,
        }

        analysis_path = output_dir / "analysis.json"
        save_artifact(
            analysis, analysis_path,
            description="analysis", logger_instance=self.logger
        )

        # Propagate all artifacts into ctx
        ctx["artifacts"].update({
            "assembled_context":          context,
            "assembled_context_path":     str(context_path),
            "spikes_annotation":          spikes_annotation,
            "spikes_annotation_path":     str(spikes_path),
            "spikes_was_run":             uses_spikes,
            "lucas_analysis":             lucas_analysis,
            "lucas_analysis_path":        str(lucas_path),
            "clinical_content":           cc_scored,
            "clinical_content_path":      str(cc_path),
            "analysis":                   analysis,
            "analysis_path":              str(analysis_path),
        })

        return ctx

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_context(self, ctx: dict) -> dict:
        transcript = self._resolve_artifact(ctx["artifacts"]["transcript"])
        features   = self._resolve_artifact(ctx["artifacts"]["features"])
        verbal     = features["verbal"]

        context: dict = {
            "diarized_transcript": [
                {
                    "speaker": seg["speaker"],
                    "start":   seg["start"],
                    "end":     seg["end"],
                    "text":    seg["text"],
                }
                for seg in transcript
            ],
            "verbal_features": {
                "summary":                verbal["summary"],
                "pause_details":          verbal["pauses"][:10],
                "interruption_details":   verbal["interruptions"][:10],
            },
            "conversation_phases": features.get("phases", []),
            "patient_vitals":      features.get("vitals"),
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

    # ------------------------------------------------------------------
    # Clinical module loader — multi-module with merge
    # ------------------------------------------------------------------

    def _load_and_merge_modules(
        self,
        module_ids: list[str],
        scenario_id: str,
    ) -> dict | None:
        """
        Load one or more clinical module JSON files and merge them into a
        single module dict for use in Pass 3.

        Merge rules:
        - `items` arrays are concatenated in declaration order.
        - `name`, `description`, and `scenario_type` are taken from the first module.
        - `_source_modules` records which files were loaded (for audit).
        - Each item is tagged with `_source_module` for traceability.

        Returns None if no module files could be loaded.
        """
        if not module_ids:
            self.logger.info(
                f"No module_ids declared for scenario '{scenario_id}'. "
                "Running generic core checklist only."
            )
            return None

        templates_dir = Path("templates/clinical_modules")
        loaded: list[dict] = []

        for mid in module_ids:
            candidates = [
                templates_dir / f"{mid}.json",
                templates_dir / f"{mid.lower()}.json",
            ]
            found = False
            for p in candidates:
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        module = json.load(f)
                    self.logger.info(f"Clinical module loaded: {p}")
                    loaded.append(module)
                    found = True
                    break
            if not found:
                self.logger.warning(
                    f"Clinical module '{mid}' not found in {templates_dir}. "
                    "Skipping this module."
                )

        if not loaded:
            self.logger.warning(
                f"No clinical modules could be loaded for scenario "
                f"'{scenario_id}'. Running generic core checklist only."
            )
            return None

        if len(loaded) == 1:
            return loaded[0]

        # ── Merge multiple modules ─────────────────────────────────────
        merged: dict = {
            "id":              scenario_id,
            "name":            loaded[0].get("name", scenario_id),
            "description":     loaded[0].get("description", ""),
            "scenario_type":   loaded[0].get("scenario_type", "unknown"),
            "_source_modules": [m.get("id", "?") for m in loaded],
            "items":           [],
        }

        # Concatenate items — deduplicate by id (first occurrence wins)
        seen_ids: set[str] = set()
        for m in loaded:
            for item in m.get("items", []):
                iid = item.get("id")
                if iid and iid not in seen_ids:
                    # Tag each item with its source module for traceability
                    tagged = dict(item)
                    tagged["_source_module"] = m.get("id", "?")
                    merged["items"].append(tagged)
                    seen_ids.add(iid)

        self.logger.info(
            f"Merged {len(loaded)} modules for scenario '{scenario_id}': "
            f"{len(merged['items'])} total items"
        )
        return merged

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_spikes_prompt(self, context: dict) -> str:
        transcript_text  = self._format_transcript(context["diarized_transcript"])
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

        return self._render_template(
            "spikes_prompt.j2",
            transcript=transcript_text,
            interaction=interaction_text,
            conversation_phases=phases_text,
            video_nvb_section=video_nvb_section,
        )

    def _build_lucas_prompt(
        self, context: dict, spikes_annotation: dict
    ) -> str:
        transcript_text = self._format_transcript(context["diarized_transcript"])
        verbal_summary  = json.dumps(
            context["verbal_features"], indent=2, ensure_ascii=False
        )
        phases_summary  = json.dumps(
            context["conversation_phases"], indent=2, ensure_ascii=False
        )
        spikes_summary  = json.dumps(spikes_annotation, indent=2, ensure_ascii=False)

        if context.get("video_nvb"):
            video_nvb_section = self._summarise_video_for_llm(context["video_nvb"])
        else:
            video_nvb_section = (
                "## Nonverbale Verhaltensmetriken\n\n"
                "_Videoanalyse nicht verfügbar. "
                "Item D ausschließlich auf Basis des Transkripts bewerten. "
                "Keine Aussagen über nonverbales Verhalten machen, die nicht "
                "textuell belegbar sind._\n"
            )

        return self._render_template(
            "lucas_prompt.j2",
            transcript=transcript_text,
            interaction=verbal_summary,
            conversation_phases=phases_summary,
            spikes_annotation=spikes_summary,
            video_nvb_section=video_nvb_section,
        )

    def _build_clinical_content_prompt(
        self, context: dict, merged_module: dict
    ) -> str:
        """
        Build Pass 3 prompt for clinical content scoring.

        Scores only the items declared in the scenario module(s). There is
        no generic core checklist — all items come from the PDF-derived
        module files for this scenario. Only called when merged_module is
        not None (caller guarantees this).
        """
        module_items   = merged_module.get("items", [])
        source_modules = merged_module.get("_source_modules", [])

        # ── Items block ────────────────────────────────────────────────
        if source_modules:
            items_block = (
                f"## Bewertungsitems: {merged_module.get('name', '')}\n"
                f"_(Zusammengeführt aus: {', '.join(source_modules)})_\n\n"
            )
        else:
            items_block = (
                f"## Bewertungsitems: {merged_module.get('name', '')}\n\n"
            )

        for item in module_items:
            crit_tag   = " [CRITICAL]" if item.get("critical", False) else ""
            source_tag = (
                f" _(Modul: {item['_source_module']})_"
                if item.get("_source_module") else ""
            )
            items_block += (
                f"**{item['id']} — {item['name']}**{crit_tag}{source_tag}\n"
                f"Beschreibung: {item['description']}\n"
            )
            sg = item.get("scoring_guidance", {})
            if sg:
                items_block += (
                    f"  - 2: {sg.get('2', '')}\n"
                    f"  - 1: {sg.get('1', '')}\n"
                    f"  - 0: {sg.get('0', '')}\n"
                )
            items_block += "\n"

        # ── Transcript (capped at 300 segments to save tokens) ─────────
        transcript_segs = context["diarized_transcript"][:300]
        transcript_text = self._format_transcript(transcript_segs)

        # ── JSON output schema — one entry per module item ─────────────
        schema_items = ""
        for item in module_items:
            schema_items += (
                f"    {{{{\n"
                f"      \"id\": \"{item['id']}\",\n"
                f"      \"name\": \"{item['name']}\",\n"
                f"      \"category\": \"{item.get('_source_module', 'Szenario-Modul')}\",\n"
                f"      \"critical\": {str(item.get('critical', False)).lower()},\n"
                f"      \"rating\": <0|1|2|\"NA\">,\n"
                f"      \"justification\": \"<ein Satz>\",\n"
                f"      \"evidence\": [\"<[MM:SS] Zitat>\"]\n"
                f"    }}}},\n"
            )

        return self._render_template(
            "clinical_content_prompt.j2",
            items_block=items_block,
            transcript_text=transcript_text,
            schema_items=schema_items,
            scoring_preamble=merged_module.get("scoring_preamble", ""),
        )

    # ------------------------------------------------------------------
    # Video NVB summariser
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_video_for_llm(video_features: dict) -> str:
        """
        Pre-interpret raw video metric dicts into concise German prose.
        Prevents the LLM from hallucinating nonverbal observations.
        """
        lines = ["## Nonverbale Verhaltensmetriken (vorinterpretierte Videoanalyse)\n"]
        lines.append(
            "Die folgenden Metriken sind bereits interpretiert. "
            "Sie sind die PRIMÄRE und einzige valide Evidenzquelle für Item D. "
            "Am Ende dieses Abschnitts stehen vorformatierte evidence-Strings, "
            "die direkt ins evidence-Feld von Item D kopiert werden sollen. "
            "Für Item F (Empathie) und I (Professionelles Verhalten) dienen "
            "sie als ergänzende Belege neben Transkriptzitaten.\n"
        )

        d1       = video_features.get("D1_eye_contact", {})
        gaze     = d1.get("gaze_on_target", {})
        gaze_rate = gaze.get("rate")
        gaze_rel = d1.get("reliability", "unbekannt")
        pct = level = None
        if gaze_rate is not None:
            pct = round(gaze_rate * 100)
            level = "gut" if gaze_rate >= 0.75 else ("moderat" if gaze_rate >= 0.50 else "niedrig")
            lines.append(
                f"**Augenkontakt (D1):** {pct}% der detektierten Frames auf "
                f"Gesprächspartner gerichtet → {level} (Zuverlässigkeit: {gaze_rel})"
            )

        d2      = video_features.get("D2_positioning", {})
        h2p     = d2.get("Height_to_patient", {})
        h2p_mean = h2p.get("mean")
        d2_rel  = d2.get("reliability", "unbekannt")
        if h2p_mean is not None:
            pos_interp = (
                "ungefähr auf Augenhöhe der Bezugsperson (günstige Positionierung)"
                if h2p_mean <= 0.35
                else "tendenziell höher als die Bezugsperson positioniert"
            )
            lines.append(
                f"**Positionierung (D2):** Augenhöhe normalisiert = {round(h2p_mean, 3)} "
                f"→ {pos_interp} (Zuverlässigkeit: {d2_rel})"
            )

        d3      = video_features.get("D3_posture", {})
        arm_dev = d3.get("baseline_arm_deviation", {}).get("mean")
        d3_rel  = d3.get("reliability", "unbekannt")
        posture = None
        if arm_dev is not None:
            if abs(arm_dev) < 0.3:
                posture = "offen/entspannt (nahe am individuellen Ruhewert)"
            elif arm_dev < -0.3:
                posture = "leicht geschlossen/angespannt (unter individuellem Ruhewert)"
            else:
                posture = "weit offen (über individuellem Ruhewert)"
            lines.append(
                f"**Körperhaltung / Armoffenheit (D3):** {posture} "
                f"(mittlere Abweichung vom Ruhewert: {round(arm_dev, 2)}, "
                f"Zuverlässigkeit: {d3_rel})"
            )

        d4         = video_features.get("D4_facial_expressions", {})
        pos_expr   = d4.get("positive_expression_rate", {})
        expr_rate  = pos_expr.get("rate")
        d4_rel     = d4.get("reliability", "unbekannt")
        pct_e = expr_level = None
        if expr_rate is not None:
            pct_e = round(expr_rate * 100)
            expr_level = (
                "erkennbar positiv/freundlich" if expr_rate >= 0.15
                else ("überwiegend neutral" if expr_rate >= 0.05
                      else "kaum positive Mimik")
            )
            lines.append(
                f"**Mimik (D4):** positive Gesichtsausdruck-Rate = {pct_e}% "
                f"→ {expr_level} (Zuverlässigkeit: {d4_rel})"
            )

        d5            = video_features.get("D5_gestures_and_mannerisms", {})
        fidget        = d5.get("hand_movement_periodicity", {})
        is_repetitive = fidget.get("is_repetitive", False)
        fidget_strength = fidget.get("periodicity_strength")
        d5_rel        = d5.get("reliability", "unbekannt")
        fidget_str    = (
            f"ja (Stärke: {round(fidget_strength, 2)})" if is_repetitive else "nein"
        )
        lines.append(
            f"**Wiederholende Handbewegungen / Fidgeting (D5):** {fidget_str} "
            f"(Zuverlässigkeit: {d5_rel})"
        )

        lines.append("")
        if (gaze_rate is not None and gaze_rate >= 0.75
                and arm_dev is not None and abs(arm_dev) < 0.3
                and not is_repetitive):
            lines.append(
                "**Gesamteinschätzung NVB:** Alle Hauptindikatoren im positiven Bereich "
                "→ nonverbales Verhalten ist förderlich für das Engagement. "
                "Entspricht LUCAS D:2 (Competent), sofern keine konkreten "
                "Transkripthinweise auf ablenkende Signale vorliegen."
            )
        elif (gaze_rate is not None and gaze_rate < 0.50) or is_repetitive:
            lines.append(
                "**Gesamteinschätzung NVB:** Mindestens ein Indikator deutlich auffällig "
                "→ nonverbales Verhalten möglicherweise ablenkend. "
                "Prüfe Transkript auf konkrete Hinweise vor Bewertung."
            )
        else:
            lines.append(
                "**Gesamteinschätzung NVB:** Gemischtes Bild — überwiegend positiv "
                "mit einzelnen auffälligen Werten. "
                "Transkriptkontext für D-Bewertung heranziehen."
            )

        # Phase-level flags
        phase_summaries = video_features.get("phase_summaries", [])
        phase_flags = []
        for ph in phase_summaries:
            ph_name    = ph.get("phase", "?")
            ph_d5      = ph.get("D5_gestures_and_mannerisms", {})
            ph_hmp     = ph_d5.get("hand_movement_periodicity", {})
            ph_pitch   = ph_d5.get("head_movement", {}).get("pitch_periodicity", {})
            ph_rel     = ph_d5.get("reliability", "unbekannt")
            ph_d3      = ph.get("D3_posture", {})
            ph_arm_dev = ph_d3.get("baseline_arm_deviation", {}).get("mean")
            issues = []
            if ph_hmp.get("is_repetitive"):
                issues.append(
                    f"Wiederholende Handbewegungen "
                    f"(Stärke: {round(ph_hmp.get('periodicity_strength', 0), 2)})"
                )
            if ph_pitch.get("is_repetitive"):
                issues.append(
                    f"Wiederholende Kopfbewegungen/Pitch "
                    f"(Stärke: {round(ph_pitch.get('periodicity_strength', 0), 2)})"
                )
            if ph_arm_dev is not None and ph_arm_dev < -0.5:
                issues.append(
                    f"Geschlossene Körperhaltung "
                    f"(Armabweichung: {round(ph_arm_dev, 2)} SD)"
                )
            if issues:
                phase_flags.append(
                    f"  ⚠ Phase '{ph_name}': "
                    + ", ".join(issues)
                    + f" (Zuverlässigkeit: {ph_rel})"
                )
        if phase_flags:
            lines.append("")
            lines.append(
                "**Phasenspezifische NVB-Auffälligkeiten** "
                "(auch wenn Globalwert unauffällig ist):"
            )
            lines.extend(phase_flags)
            lines.append(
                "  → Phasenspezifische Signale in D-Bewertung berücksichtigen; "
                "bei Dauer < 60s oder Stärke < 0.4 als Grenzfall behandeln."
            )

        # Pre-formatted evidence strings
        lines.append("")
        lines.append("## Vorformatierte evidence-Strings für Item D")
        lines.append(
            "Kopiere die folgenden Strings DIREKT in das evidence-Feld von Item D. "
            "Ersetze sie NICHT durch Transkriptzitate. "
            "Transkriptzitate sind für Item D UNGÜLTIG."
        )
        ev_lines = []
        if pct is not None:
            ev_lines.append(
                f'- "Augenkontakt (D1): {pct}% der Frames auf Gesprächspartner '
                f'gerichtet → {level} (Zuverlässigkeit: {gaze_rel})"'
            )
        if posture is not None:
            ev_lines.append(
                f'- "Körperhaltung (D3): {posture} '
                f'(Abweichung vom Ruhewert: {round(arm_dev, 2)} SD, '
                f'Zuverlässigkeit: {d3_rel})"'
            )
        ev_lines.append(
            f'- "Wiederholende Handbewegungen / Fidgeting (D5): {fidget_str} '
            f'(Zuverlässigkeit: {d5_rel})"'
        )
        if pct_e is not None:
            ev_lines.append(
                f'- "Mimik (D4): positive Ausdrucksrate {pct_e}% '
                f'→ {expr_level} (Zuverlässigkeit: {d4_rel})"'
            )
        lines.extend(ev_lines)

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_transcript(segments: list[dict]) -> str:
        return "\n".join(
            f"[{s['speaker']}] ({s['start']:.1f}-{s['end']:.1f}s): {s['text']}"
            for s in segments
        )

    @staticmethod
    def _render_template(template_name: str, **kwargs) -> str:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
        templates_dir = Path(__file__).parent.parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )
        return env.get_template(template_name).render(**kwargs)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_evidence(result: dict) -> dict:
        for key in ("lucas_items", "items"):
            for item in result.get(key, []):
                ev = item.get("evidence")
                if isinstance(ev, list) and len(ev) > 3:
                    item["evidence"] = ev[:3]
                    item.setdefault("_evidence_truncated", True)
        return result

    @staticmethod
    def _normalise_timestamps(result: dict) -> dict:
        def _fix(text: str) -> str:
            def _repl(m):
                mm = int(m.group(1))
                ss = float(m.group(2))
                if ss >= 60:
                    extra = int(ss) // 60
                    ss   -= extra * 60
                    mm   += extra
                return f"[{mm:02d}:{int(ss):02d}]"
            return re.sub(r"\[(\d+):(\d+(?:\.\d+)?)\]", _repl, text)

        for key in ("lucas_items", "items"):
            for item in result.get(key, []):
                ev = item.get("evidence")
                if isinstance(ev, list):
                    item["evidence"] = [
                        _fix(e) if isinstance(e, str) else e for e in ev
                    ]
        return result

    def _parse_output(self, raw: str, pass_name: str) -> dict:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = self._take_first_json_object(text)

        try:
            return self._normalise_timestamps(self._cap_evidence(json.loads(text)))
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return self._normalise_timestamps(
                    self._cap_evidence(json.loads(match.group()))
                )
            except json.JSONDecodeError:
                pass

        try:
            salvaged = self._salvage_corrupt_json(text, pass_name)
            if salvaged:
                return self._normalise_timestamps(self._cap_evidence(salvaged))
        except Exception:
            pass

        self.logger.error(f"Failed to parse {pass_name} LLM output as JSON")
        return {"parse_error": True, "pass": pass_name, "raw_output": raw}

    @staticmethod
    def _take_first_json_object(text: str) -> str:
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[: i + 1]
        return text

    def _salvage_corrupt_json(self, text: str, pass_name: str) -> dict | None:
        import re as _re

        def _extract_array_content(txt: str, key: str) -> str | None:
            start = txt.find(f'"{key}"')
            if start == -1:
                return None
            bracket_start = txt.find("[", start)
            if bracket_start == -1:
                return None
            depth = 0
            in_str = esc = False
            for i in range(bracket_start, len(txt)):
                ch = txt[i]
                if esc:
                    esc = False; continue
                if ch == "\\" and in_str:
                    esc = True; continue
                if ch == '"' and not esc:
                    in_str = not in_str; continue
                if in_str:
                    continue
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return txt[bracket_start + 1: i]
            return None

        def _extract_good_items(arr_text: str) -> list:
            good = []
            parts = _re.split(r"(?<=\}),\s*(?=\{)", arr_text.strip())
            for part in parts:
                part = part.strip().rstrip(",")
                try:
                    obj = json.loads(part)
                    if isinstance(obj, dict) and ("item" in obj or "id" in obj):
                        good.append(obj)
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"{pass_name}: Dropped corrupt item: {part[:80]!r}"
                    )
            return good

        items_key = "items" if pass_name == "clinical_content" else "lucas_items"
        note_key  = "overall_clinical_note" if pass_name == "clinical_content" else "overall_summary"

        # Strategy 1 — corrupt item removal
        arr_content = (
            _extract_array_content(text, "lucas_items")
            or _extract_array_content(text, "items")
        )
        if arr_content:
            good_items = _extract_good_items(arr_content)
            if good_items:
                self.logger.warning(
                    f"{pass_name}: Salvaged {len(good_items)} items (Strategy 1)"
                )
                return {
                    items_key: good_items,
                    "total_score": sum(
                        i.get("rating", i.get("score", 0))
                        for i in good_items
                        if str(i.get("rating", "NA")).upper() != "NA"
                    ),
                    note_key: "[Salvaged — corrupt items removed]",
                    "_salvaged": True,
                }

        # Strategy 2 — truncation recovery
        for pattern, suffix in [
            (r"\}\s*,\s*\n", "\n  ]\n}"),
            (r"\}", "\n  ]\n}"),
        ]:
            closes = [m.end() for m in _re.finditer(pattern, text)]
            if closes:
                snippet = text[: closes[-1]].rstrip().rstrip(",")
                try:
                    result = json.loads(snippet + suffix)
                    items = result.get("lucas_items", result.get("items", []))
                    if items:
                        self.logger.warning(
                            f"{pass_name}: Salvaged {len(items)} items (Strategy 2)"
                        )
                        result.setdefault("_salvaged", True)
                        result.setdefault(note_key, "[Salvaged — truncated output]")
                        return result
                except json.JSONDecodeError:
                    continue

        # Strategy 3 — pattern extraction
        item_matches = _re.findall(
            r'\{\s*"item"\s*:\s*"[A-J]"[\s\S]{20,500}?\}', text
        )
        good_items = []
        for raw_item in item_matches:
            try:
                obj = json.loads(raw_item)
                if "item" in obj and "rating" in obj:
                    good_items.append(obj)
            except json.JSONDecodeError:
                pass
        if good_items:
            self.logger.warning(
                f"{pass_name}: Salvaged {len(good_items)} items (Strategy 3)"
            )
            return {
                items_key: good_items,
                "total_score": sum(
                    i.get("rating", 0)
                    for i in good_items
                    if str(i.get("rating", "NA")).upper() != "NA"
                ),
                note_key: "[Salvaged — pattern extraction]",
                "_salvaged": True,
            }

        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_spikes(self, annotation: dict) -> None:
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

    def _validate_lucas(self, analysis: dict) -> None:
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

    # ------------------------------------------------------------------
    # Clinical content scoring
    # ------------------------------------------------------------------

    def _score_clinical_content(
        self, parsed: dict, merged_module: dict
    ) -> dict:
        """
        Post-process LLM clinical content output.

        Scores only the items declared in the merged scenario module.
        There is no generic core — all expected items come from the module.
        """
        items = parsed.get("items") or parsed.get("lucas_items", [])
        scored_by_id = {i.get("id"): i for i in items if i.get("id")}

        # Expected items are exactly what the module declares
        expected = [
            {
                "id":       m["id"],
                "name":     m["name"],
                "category": m.get("_source_module", "Szenario-Modul"),
                "critical": m.get("critical", False),
            }
            for m in merged_module.get("items", [])
        ]

        raw_score      = 0
        max_applicable = 0
        critical_misses: list[dict] = []
        critical_fps:    list[dict] = []
        category_scores: dict[str, dict] = {}
        enriched:        list[dict] = []

        for exp in expected:
            iid         = exp["id"]
            name        = exp["name"]
            cat         = exp["category"]
            is_critical = exp["critical"]

            scored     = scored_by_id.get(iid, {})
            rating_raw = scored.get("rating", "NA")

            if str(rating_raw).upper() == "NA":
                rating = "NA"
            else:
                try:
                    rating = max(0, min(2, int(rating_raw)))
                except (TypeError, ValueError):
                    rating = "NA"

            just     = scored.get("justification", "")
            is_wrong = just.upper().startswith("FALSCH:")

            if rating != "NA":
                raw_score      += rating
                max_applicable += 2
                cat_entry = category_scores.setdefault(cat, {"raw": 0, "max": 0})
                cat_entry["raw"] += rating
                cat_entry["max"] += 2

            if is_critical and rating == 0:
                critical_misses.append(
                    {"id": iid, "name": name, "category": cat, "justification": just}
                )
            if is_wrong:
                critical_fps.append(
                    {"id": iid, "name": name, "category": cat, "justification": just}
                )

            enriched.append({
                "id":            iid,
                "name":          name,
                "category":      cat,
                "critical":      is_critical,
                "rating":        rating,
                "justification": just,
                "evidence":      scored.get("evidence", [])[:2],
                "wrong":         is_wrong,
            })

        normalised_pct = (
            round(raw_score / max_applicable * 100, 1)
            if max_applicable > 0 else None
        )
        category_pct = {
            cat: round(v["raw"] / v["max"] * 100, 1) if v["max"] > 0 else None
            for cat, v in category_scores.items()
        }
        source_modules = merged_module.get(
            "_source_modules", [merged_module.get("id", "?")]
        )

        return {
            "items":                    enriched,
            "raw_score":                raw_score,
            "max_applicable_score":     max_applicable,
            "normalised_score_pct":     normalised_pct,
            "category_scores_pct":      category_pct,
            "critical_misses":          critical_misses,
            "critical_false_positives": critical_fps,
            "has_critical_miss":        len(critical_misses) > 0,
            "overall_clinical_note":    parsed.get("overall_clinical_note", ""),
            "_source_modules":          source_modules,
            "_salvaged":                parsed.get("_salvaged", False),
        }

    def _validate_clinical_content(self, cc: dict) -> None:
        if cc.get("parse_error"):
            self.logger.error("Clinical content has parse errors.")
            return

        n = len(cc.get("items", []))
        if n == 0:
            self.logger.warning("Clinical content: no items scored.")

        n_cm = len(cc.get("critical_misses", []))
        if n_cm:
            self.logger.warning(
                f"Clinical content: {n_cm} CRITICAL MISS(ES): "
                + ", ".join(
                    f"{m['id']} ({m['name']})" for m in cc["critical_misses"]
                )
            )

        n_fp = len(cc.get("critical_false_positives", []))
        if n_fp:
            self.logger.warning(
                f"Clinical content: {n_fp} CRITICAL FALSE POSITIVE(S): "
                + ", ".join(
                    f"{m['id']} ({m['name']})"
                    for m in cc["critical_false_positives"]
                )
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._llm_backend is not None:
            self._llm_backend.cleanup()
            self._llm_backend = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()