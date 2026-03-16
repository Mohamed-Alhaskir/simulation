"""
Clinical Content Scorer
=======================
Evaluates medical knowledge against scenario-specific clinical modules.
Handles module loading, merging, scoring, and result combination.
"""

import json
import logging
from pathlib import Path
from typing import Any

from utils.llm_backends import LLMBackend
from utils.artifact_io import save_artifact


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


class ClinicalContentScorer:
    """Scorer for scenario-specific clinical content assessment."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def score(
        self,
        context: dict,
        backend: LLMBackend,
        cfg: dict,
        module_ids: list[str],
        scenario_id: str,
        strictness_preamble: str = "",
        output_dir: Path = None,
    ) -> dict:
        """
        Score clinical content against scenario-specific modules.

        Args:
            context: Assembled context dict with diarized_transcript, etc.
            backend: LLM backend instance
            cfg: LLM config dict
            module_ids: List of module IDs to load (e.g., ["Diabetes"])
            scenario_id: Scenario identifier for context
            strictness_preamble: Strictness calibration text to inject
            output_dir: Directory to save prompts and outputs (optional)

        Returns:
            Scored clinical content dict, or null stub if no modules found
        """
        self.logger.info("Pass 3: Clinical Content scoring")

        if not module_ids:
            self.logger.warning(
                f"Pass 3: skipped — no clinical modules found for '{scenario_id}'"
            )
            cc_scored = dict(_CC_NULL_STUB)
            cc_scored["scenario_id"] = scenario_id
            return cc_scored

        # Load and merge modules
        cc_module_results: list[dict] = []

        for mid in module_ids:
            single_module = self._load_and_merge_modules([mid], scenario_id)
            if single_module is None:
                self.logger.warning(
                    f"Pass 3: module '{mid}' not found, skipping"
                )
                continue

            mod_label = single_module.get("id", mid)
            self.logger.info(f"Pass 3: scoring module '{mod_label}'")

            # Build and run prompt
            cc_prompt = self._build_clinical_content_prompt(
                context, single_module, strictness_preamble
            )

            if output_dir:
                output_dir_path = Path(output_dir)
                (output_dir_path / f"clinical_content_{mod_label}_prompt.txt").write_text(
                    cc_prompt, encoding="utf-8"
                )

            cc_raw = backend.generate(cc_prompt, cfg)

            if output_dir:
                (output_dir_path / f"clinical_content_{mod_label}_raw_output.txt").write_text(
                    cc_raw, encoding="utf-8"
                )

            # Parse and score
            cc_parsed = self._parse_clinical_content_output(cc_raw)
            cc_scored_mod = self._score_clinical_content(cc_parsed, single_module)
            cc_module_results.append(cc_scored_mod)

        # Combine results if multiple modules
        if not cc_module_results:
            self.logger.warning(
                f"Pass 3: skipped — no clinical modules found for '{scenario_id}'"
            )
            cc_scored = dict(_CC_NULL_STUB)
            cc_scored["scenario_id"] = scenario_id
        elif len(cc_module_results) == 1:
            cc_scored = cc_module_results[0]
        else:
            cc_scored = self._combine_cc_results(cc_module_results)

        self._validate_clinical_content(cc_scored)

        if not cc_scored.get("skipped"):
            n_critical_miss = len(cc_scored.get("critical_misses", []))
            self.logger.info(
                f"Clinical content complete: "
                f"{cc_scored.get('normalised_score_pct', '?')}% "
                f"({cc_scored.get('raw_score', '?')}/"
                f"{cc_scored.get('max_applicable_score', '?')}), "
                f"{n_critical_miss} critical miss(es)"
            )

        return cc_scored

    def _load_and_merge_modules(
        self,
        module_ids: list[str],
        scenario_id: str,
    ) -> dict | None:
        """Load and merge clinical content modules from JSON files."""
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
            m = loaded[0]
            label = m.get("name") or m.get("id", "?")
            for item in m.get("items", []):
                item.setdefault("_source_module", label)
            return m

        # Merge multiple modules
        merged: dict = {
            "id": scenario_id,
            "name": loaded[0].get("name", scenario_id),
            "description": loaded[0].get("description", ""),
            "scenario_type": loaded[0].get("scenario_type", "unknown"),
            "_source_modules": [m.get("id", "?") for m in loaded],
            "items": [],
        }

        seen_ids: set[str] = set()
        for m in loaded:
            for item in m.get("items", []):
                iid = item.get("id")
                if iid and iid not in seen_ids:
                    tagged = dict(item)
                    tagged["_source_module"] = m.get("name") or m.get("id", "?")
                    merged["items"].append(tagged)
                    seen_ids.add(iid)

        self.logger.info(
            f"Merged {len(loaded)} modules for scenario '{scenario_id}': "
            f"{len(merged['items'])} total items"
        )
        return merged

    def _build_clinical_content_prompt(
        self,
        context: dict,
        merged_module: dict,
        strictness_preamble: str,
    ) -> str:
        """Build clinical content evaluation prompt."""
        module_items = merged_module.get("items", [])
        source_modules = merged_module.get("_source_modules", [])

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
            crit_tag = " [CRITICAL]" if item.get("critical", False) else ""
            source_tag = (
                f" _(Modul: {item['_source_module']})_"
                if item.get("_source_module")
                else ""
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

        transcript_segs = context["diarized_transcript"][:300]
        transcript_text = self._format_transcript(transcript_segs)

        schema_items = ""
        for item in module_items:
            schema_items += (
                f"    {{\n"
                f"      \"id\": \"{item['id']}\",\n"
                f"      \"name\": \"{item['name']}\",\n"
                f"      \"category\": \"{item.get('_source_module', 'Szenario-Modul')}\",\n"
                f"      \"critical\": {str(item.get('critical', False)).lower()},\n"
                f"      \"rating\": <0|1|2|\"NA\">,\n"
                f"      \"justification\": \"<ein Satz>\",\n"
                f"      \"evidence\": [\"<[MM:SS] Zitat>\"]\n"
                f"    }},\n"
            )

        return self._render_template(
            "clinical_content_prompt.j2",
            items_block=items_block,
            transcript_text=transcript_text,
            schema_items=schema_items,
            scoring_preamble=merged_module.get("scoring_preamble", ""),
            strictness_preamble=strictness_preamble,
        )

    def _parse_clinical_content_output(self, raw: str) -> dict:
        """Parse clinical content LLM output using generic JSON parser."""
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
        self.logger.error("Failed to parse clinical content LLM output as JSON")
        return {"parse_error": True, "raw_output": raw}

    def _score_clinical_content(
        self, parsed: dict, merged_module: dict
    ) -> dict:
        """Score parsed clinical content against module expectations."""
        items = parsed.get("items") or parsed.get("lucas_items", [])
        scored_by_id = {i.get("id"): i for i in items if i.get("id")}

        expected = [
            {
                "id": m["id"],
                "name": m["name"],
                "category": m.get("_source_module", "Szenario-Modul"),
                "critical": m.get("critical", False),
            }
            for m in merged_module.get("items", [])
        ]

        raw_score = 0
        max_applicable = 0
        critical_misses: list[dict] = []
        critical_fps: list[dict] = []
        category_scores: dict[str, dict] = {}
        enriched: list[dict] = []

        for exp in expected:
            iid = exp["id"]
            name = exp["name"]
            cat = exp["category"]
            is_critical = exp["critical"]

            scored = scored_by_id.get(iid, {})
            rating_raw = scored.get("rating", "NA")

            if str(rating_raw).upper() == "NA":
                rating = "NA"
            else:
                try:
                    rating = max(0, min(2, int(rating_raw)))
                except (TypeError, ValueError):
                    rating = "NA"

            just = scored.get("justification", "")
            is_wrong = just.upper().startswith("FALSCH:")

            if rating != "NA":
                raw_score += rating
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

            enriched.append(
                {
                    "id": iid,
                    "name": name,
                    "category": cat,
                    "critical": is_critical,
                    "rating": rating,
                    "justification": just,
                    "evidence": scored.get("evidence", [])[:2],
                    "wrong": is_wrong,
                }
            )

        normalised_pct = (
            round(raw_score / max_applicable * 100, 1)
            if max_applicable > 0
            else None
        )
        category_pct = {
            cat: round(v["raw"] / v["max"] * 100, 1) if v["max"] > 0 else None
            for cat, v in category_scores.items()
        }
        source_modules = merged_module.get(
            "_source_modules", [merged_module.get("id", "?")]
        )

        return {
            "items": enriched,
            "raw_score": raw_score,
            "max_applicable_score": max_applicable,
            "normalised_score_pct": normalised_pct,
            "category_scores_pct": category_pct,
            "critical_misses": critical_misses,
            "critical_false_positives": critical_fps,
            "has_critical_miss": len(critical_misses) > 0,
            "overall_clinical_note": parsed.get("overall_clinical_note", ""),
            "_source_modules": source_modules,
            "_salvaged": parsed.get("_salvaged", False),
        }

    @staticmethod
    def _combine_cc_results(results: list[dict]) -> dict:
        """Merge per-module clinical content scored dicts into one combined result."""
        all_items: list[dict] = []
        raw_score = max_applicable = 0
        category_scores: dict[str, dict] = {}
        critical_misses: list[dict] = []
        critical_fps: list[dict] = []
        notes: list[str] = []
        source_modules: list[str] = []

        for r in results:
            all_items.extend(r.get("items", []))
            raw_score += r.get("raw_score", 0)
            max_applicable += r.get("max_applicable_score", 0)
            critical_misses.extend(r.get("critical_misses", []))
            critical_fps.extend(r.get("critical_false_positives", []))
            if r.get("overall_clinical_note"):
                notes.append(r["overall_clinical_note"])
            source_modules.extend(r.get("_source_modules", []))
            # re-aggregate category scores from items (exact integers, no rounding loss)
            for item in r.get("items", []):
                if item.get("rating") not in (None, "NA"):
                    cat = item.get("category", "")
                    cs = category_scores.setdefault(cat, {"raw": 0, "max": 0})
                    cs["raw"] += int(item["rating"])
                    cs["max"] += 2

        norm_pct = (
            round(raw_score / max_applicable * 100, 1) if max_applicable else None
        )
        cat_pct = {
            k: round(v["raw"] / v["max"] * 100, 1) if v["max"] else None
            for k, v in category_scores.items()
        }

        return {
            "items": all_items,
            "raw_score": raw_score,
            "max_applicable_score": max_applicable,
            "normalised_score_pct": norm_pct,
            "category_scores_pct": cat_pct,
            "critical_misses": critical_misses,
            "critical_false_positives": critical_fps,
            "has_critical_miss": bool(critical_misses),
            "overall_clinical_note": " | ".join(notes),
            "_source_modules": source_modules,
            "_salvaged": any(r.get("_salvaged") for r in results),
        }

    def _validate_clinical_content(self, cc: dict) -> None:
        """Validate clinical content scoring result."""
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
