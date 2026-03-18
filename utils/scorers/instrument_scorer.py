"""
instrument_scorer.py
====================
Unified multi-pass scoring engine driven by instrument JSON files and Jinja2 templates.

Every instrument (LUCAS, SPIKES, Diabetes_CC, GSLP, LP_Aufklaerung, …) is defined by:
  - instruments/<ID>.json   — items, passes, scales, validation rules
  - templates/instruments/<pass>.j2 — one template per pass

Interface:
    scorer = InstrumentScorer("instruments/LUCAS.json", backend, cfg, logger=logger)
    result = scorer.score(context)   # → standardized ScoringResult dict

Pass types:
    standard        — normal scoring pass
    aggregation     — LUCAS pass 7: overall summary + evidence conflict check
    sequence_check  — SPIKES pass 7: validate S→P→I→K→E→S2 order
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from utils.json_utils import repair_unescaped_quotes


# ── helpers (module-level, shared) ────────────────────────────────────────────

_ROOT = Path(__file__).parent.parent.parent
_TEMPLATES_DIR = _ROOT / "templates" / "instruments"


def _detect_register_break(diarized_transcript: list) -> tuple[bool, list]:
    """Detect informal 'du' forms used by SPEAKER_00."""
    pattern = re.compile(
        r'\b(du|dich|dir|dein|deine[nrms]?)\b',
        flags=re.IGNORECASE
    )
    hits = []
    for seg in diarized_transcript:
        if seg.get("speaker") == "SPEAKER_00":
            for match in pattern.finditer(seg.get("text", "")):
                ts = seg.get("start", 0)
                mm = int(ts) // 60
                ss = int(ts) % 60
                hits.append({
                    "timestamp": f"{mm:02d}:{ss:02d}",
                    "text": seg["text"].strip(),
                    "match": match.group()
                })
    return bool(hits), hits


def _format_register_warning(hits: list) -> str:
    if not hits:
        return ""
    lines = ["REGISTER BREAK DETECTED (automatic scan)"]
    lines.append(
        "SPEAKER_00 uses informal 'du'-forms. Check whether each instance is "
        "directed at SPEAKER_01 (= register break, relevant for Item I) or "
        "refers to a third person (= not a register break)."
    )
    for h in hits[:5]:
        lines.append(f"  [{h['timestamp']}] \"{h['text']}\" (match: '{h['match']}')")
    if len(hits) > 5:
        lines.append(f"  ... ({len(hits) - 5} more matches)")
    return "\n".join(lines)


def _format_full_transcript(diarized_transcript: list) -> str:
    """Format transcript as [MM:SS] [SPEAKER_XX] text."""
    lines = []
    for seg in diarized_transcript:
        ts = seg.get("start", 0)
        mm = int(ts) // 60
        ss = int(ts) % 60
        lines.append(f"[{mm:02d}:{ss:02d}] [{seg['speaker']}] {seg['text']}")
    return "\n".join(lines)


def _format_video_summary(video_features: dict) -> str:
    if not video_features:
        return "Keine Videodaten verfuegbar."

    lines = []
    d1 = video_features.get("D1_eye_contact", {})
    gaze_rate = d1.get("gaze_on_target", {}).get("rate")
    d1_rel = d1.get("reliability", "unbekannt")
    data_avail = d1.get("data_availability_rate")
    if gaze_rate is not None:
        gaze_pct = round(gaze_rate * 100)
        lines.append(
            f"D1_Blickkontakt: gaze_on_target={gaze_pct}%, "
            f"data_availability={round(data_avail * 100) if data_avail else '?'}%, "
            f"reliability={d1_rel}"
        )

    d2 = video_features.get("D2_positioning", {})
    d2_rel = d2.get("reliability", "unbekannt")
    horizon_valid = d2.get("horizon_valid", False)
    at_rate = d2.get("at_patient_eye_level_rate", {}).get("rate")
    above_rate = d2.get("above_patient_eye_level_rate", {}).get("rate")
    if horizon_valid and at_rate is not None and above_rate is not None:
        at_pct = round(at_rate * 100)
        above_pct = round(above_rate * 100)
        lines.append(
            f"D2_Positionierung: horizon_valid=true, at_eye_level={at_pct}%, "
            f"above={above_pct}%, D2_reliability={d2_rel}"
        )
    else:
        lines.append(
            f"D2_Positionierung: horizon_valid={horizon_valid}, D2_reliability={d2_rel} "
            f"-> nicht auswertbar"
        )

    d3 = video_features.get("D3_posture", {})
    d3_rel = d3.get("reliability", "unbekannt")
    arm_dev = d3.get("baseline_arm_deviation", {}).get("median")
    if arm_dev is not None:
        lines.append(
            f"D3_Haltung: arm_deviation_median={arm_dev:.2f}, reliability={d3_rel}"
        )

    overall_rel = video_features.get(
        "I_professional_behaviour_demeanour", {}
    ).get("overall_reliability", "unbekannt")
    lines.append(f"Gesamtreliability: {overall_rel}")

    return "\n".join(lines)


def _format_interaction_metrics(verbal_features: dict) -> str:
    if not verbal_features:
        return "Keine Interaktionsmetriken verfuegbar."

    summary = verbal_features.get("summary", {})
    speakers = summary.get("speakers", {})
    lines = []

    s00 = speakers.get("SPEAKER_00", {})
    s01 = speakers.get("SPEAKER_01", {})

    ratio_00 = s00.get("speaking_ratio", 0)
    ratio_01 = s01.get("speaking_ratio", 0)
    lines.append(f"Sprechangteil SPEAKER_00: {ratio_00:.1%}")
    lines.append(f"Sprechangteil SPEAKER_01: {ratio_01:.1%}")
    lines.append(f"Turns SPEAKER_00: {s00.get('turn_count', '?')}")
    lines.append(f"Turns SPEAKER_01: {s01.get('turn_count', '?')}")
    lines.append(
        f"Avg Turndauer SPEAKER_00: {s00.get('avg_turn_duration_s', '?'):.1f}s"
        if isinstance(s00.get('avg_turn_duration_s'), float) else
        f"Avg Turndauer SPEAKER_00: ?"
    )
    lines.append(f"Gespraechsdauer gesamt: {summary.get('total_duration_s', '?'):.0f}s")
    lines.append(f"Bedeutungsvolle Pausen: {summary.get('meaningful_pauses', '?')}")
    lines.append(f"Unterbrechungen: {summary.get('interruptions', '?')}")

    return "\n".join(lines)


# ── main class ─────────────────────────────────────────────────────────────────

class InstrumentScorer:
    """
    Generic multi-pass scoring engine.

    One instance per instrument:
        scorer = InstrumentScorer("instruments/LUCAS.json", backend, cfg)
        result = scorer.score(context)
    """

    def __init__(
        self,
        instrument_path: str | Path,
        backend,
        cfg: dict,
        logger: logging.Logger = None,
    ):
        self.backend = backend
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.instrument = self._load_instrument(Path(instrument_path))
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )

    # ── public ────────────────────────────────────────────────────────────────

    def score(self, context: dict) -> dict:
        """
        Run all passes and return standardized ScoringResult dict.

        context keys used:
            diarized_transcript, verbal_features, video_nvb,
            conversation_phases, spikes_annotation (optional)
        """
        instr = self.instrument
        diarized = context.get("diarized_transcript", [])
        video_features = context.get("video_nvb") or {}
        verbal_features = context.get("verbal_features") or {}

        # Pre-compute helpers (used across passes)
        register_break, register_hits = _detect_register_break(diarized)
        precomputed = {
            "transcript_text": _format_full_transcript(diarized),
            "video_summary": _format_video_summary(video_features),
            "interaction_metrics": _format_interaction_metrics(verbal_features),
            "register_warning": _format_register_warning(register_hits),
            "register_break": register_break,
        }

        # Build item_def lookup
        item_defs: dict[str, dict] = {
            item["id"]: item for item in instr.get("items", [])
        }

        all_items: list[dict] = []
        overall_note = ""

        for pass_def in instr.get("passes", []):
            pass_type = pass_def.get("type", "standard")

            try:
                if pass_type == "aggregation":
                    overall_note = self._run_aggregation_pass(
                        pass_def, all_items
                    )

                elif pass_type == "sequence_check":
                    seq_result = self._run_sequence_check_pass(
                        pass_def, all_items, precomputed
                    )
                    # Inject sequence fields into the result
                    all_items.append({"_sequence_check": seq_result})

                else:
                    # Standard scoring pass
                    pass_items = [
                        item_defs[iid]
                        for iid in pass_def.get("item_ids", [])
                        if iid in item_defs
                    ]
                    if not pass_items:
                        self.logger.warning(
                            f"Pass {pass_def['id']}: no matching item_defs found"
                        )
                        continue

                    pass_context = self._prepare_context(
                        context, pass_def, precomputed
                    )
                    prompt = self._render_prompt(
                        pass_def["template"], pass_context, pass_items, instr
                    )

                    pass_cfg = {
                        **self.cfg,
                        "max_tokens": pass_def.get("max_tokens", 4000),
                    }

                    self.logger.info(
                        f"Instrument {instr['id']} — running {pass_def['id']} "
                        f"({len(pass_items)} items)"
                    )
                    raw = self.backend.generate(prompt, pass_cfg)
                    parsed_items = self._parse_response(raw)

                    if parsed_items is None:
                        self.logger.error(
                            f"Pass {pass_def['id']}: parse failed, skipping"
                        )
                        continue

                    for item in parsed_items:
                        item_id = item.get("id") or item.get("item")
                        if not item_id:
                            continue
                        item_def = item_defs.get(item_id, {})
                        validated = self._validate_item(
                            item, context, item_def
                        )
                        all_items.append(validated)

            except Exception as e:
                self.logger.error(
                    f"Pass {pass_def.get('id', '?')} failed: {e}", exc_info=True
                )

        return self._compute_result(all_items, instr, overall_note)

    # ── private ───────────────────────────────────────────────────────────────

    def _load_instrument(self, path: Path) -> dict:
        if not path.is_absolute():
            path = _ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Instrument file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _prepare_context(
        self,
        context: dict,
        pass_def: dict,
        precomputed: dict,
    ) -> dict:
        """Build template variables from context_requirements."""
        req = pass_def.get("context_requirements", {})
        tc = {}

        # Transcript
        transcript_mode = req.get("transcript", "full")
        if transcript_mode == "video_only":
            tc["transcript_text"] = None
        else:
            tc["transcript_text"] = precomputed["transcript_text"]

        # Optional feature blocks
        if req.get("video_features"):
            tc["video_summary"] = precomputed["video_summary"]
        if req.get("verbal_features"):
            tc["interaction_metrics"] = precomputed["interaction_metrics"]
        if req.get("register_warning"):
            tc["register_warning"] = precomputed["register_warning"]
        if req.get("spikes_annotation"):
            spikes = context.get("spikes_annotation", "")
            if isinstance(spikes, dict):
                spikes = json.dumps(spikes, indent=2, ensure_ascii=False)
            tc["spikes_annotation"] = spikes
        if req.get("emotion_scan"):
            tc["emotion_scan"] = True
        if req.get("phase_focus"):
            tc["phase_focus"] = req["phase_focus"]

        # Video summary always available for video_only passes
        tc.setdefault("video_summary", precomputed["video_summary"])

        return tc

    def _render_prompt(
        self,
        template_name: str,
        pass_context: dict,
        pass_items: list[dict],
        instrument: dict,
    ) -> str:
        try:
            template = self._jinja_env.get_template(template_name)
        except Exception as e:
            raise RuntimeError(
                f"Template '{template_name}' not found in {_TEMPLATES_DIR}: {e}"
            )
        return template.render(
            items=pass_items,
            scoring_preamble=instrument.get("scoring_preamble", ""),
            **pass_context,
        )

    def _parse_response(self, raw: str) -> list[dict] | None:
        """Parse LLM JSON response; returns items list or None on failure."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        parsed = None

        # Try direct parse
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try brace extract
        if parsed is None:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                candidate = text[start:end + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        repaired = repair_unescaped_quotes(candidate)
                        parsed = json.loads(repaired)
                    except Exception:
                        pass

        if parsed is None:
            self.logger.error(
                f"JSON parse failed. Raw (first 400): {raw[:400]!r}"
            )
            return None

        # Extract items list
        if isinstance(parsed, list):
            return parsed
        if "items" in parsed:
            return parsed["items"]
        return None

    def _validate_item(
        self,
        item: dict,
        context: dict,
        item_def: dict,
    ) -> dict:
        """Apply validation_rules from item_def."""
        flags = item.setdefault("validation_flags", [])
        rating = item.get("rating")
        video_features = context.get("video_nvb") or {}

        for rule in item_def.get("validation_rules", []):
            rule_type = rule.get("type")

            if rule_type == "video_threshold":
                field_path = rule.get("field", "").split(".")
                value = video_features
                for key in field_path:
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        value = None
                        break
                if value is None:
                    continue
                op = rule.get("operator", "<")
                threshold = rule.get("threshold", 0.75)
                blocked_score = rule.get("blocked_score")
                forced_score = rule.get("forced_score")

                triggered = (
                    (op == "<" and value < threshold) or
                    (op == "<=" and value <= threshold) or
                    (op == ">" and value > threshold) or
                    (op == ">=" and value >= threshold)
                )
                if triggered and rating == blocked_score:
                    msg = rule.get("message", f"video_threshold violated")
                    msg = msg.replace("{value}", f"{value:.2f}")
                    flags.append(msg)
                    item["rating"] = forced_score
                    item["justification"] = (
                        item.get("justification", "") +
                        f" [VALIDATOR: {msg}]"
                    )
                    self.logger.warning(
                        f"Item {item.get('id', '?')}: {msg}"
                    )

            elif rule_type == "reliability_gate":
                field_path = rule.get("field", "").split(".")
                value = video_features
                for key in field_path:
                    if isinstance(value, dict):
                        value = value.get(key)
                    else:
                        value = None
                        break
                blocked_score = rule.get("blocked_score")
                forced_score = rule.get("forced_score")
                gate_value = rule.get("value", "low")
                if value == gate_value and rating == blocked_score:
                    msg = rule.get("message", f"reliability_gate: {value}")
                    flags.append(msg)
                    item["rating"] = forced_score
                    item["justification"] = (
                        item.get("justification", "") +
                        f" [VALIDATOR: {msg}]"
                    )
                    self.logger.warning(
                        f"Item {item.get('id', '?')}: {msg}"
                    )

            elif rule_type == "disallow_score":
                disallowed = rule.get("disallowed")
                forced_score = rule.get("forced_score", 0)
                if rating == disallowed:
                    msg = rule.get(
                        "message",
                        f"disallowed score {disallowed} → forced to {forced_score}"
                    )
                    flags.append(msg)
                    item["rating"] = forced_score
                    item["justification"] = (
                        item.get("justification", "") +
                        f" [VALIDATOR: {msg}]"
                    )
                    self.logger.warning(
                        f"Item {item.get('id', '?')}: {msg}"
                    )

        return item

    def _run_aggregation_pass(
        self,
        pass_def: dict,
        all_items: list[dict],
    ) -> str:
        """Run aggregation pass (LUCAS pass 7): overall summary."""
        scorable = [
            i for i in all_items if not i.get("_sequence_check")
        ]
        total = sum(
            i.get("rating", 0) for i in scorable
            if isinstance(i.get("rating"), (int, float))
        )
        items_json = json.dumps(scorable, ensure_ascii=False, indent=2)

        try:
            template = self._jinja_env.get_template(pass_def["template"])
        except Exception as e:
            self.logger.error(f"Aggregation template load failed: {e}")
            return ""

        prompt = template.render(
            items_json=items_json,
            total_score=total,
            scoring_preamble=self.instrument.get("scoring_preamble", ""),
        )
        pass_cfg = {**self.cfg, "max_tokens": pass_def.get("max_tokens", 2000)}
        raw = self.backend.generate(prompt, pass_cfg)

        # Parse minimal response
        text = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

        return parsed.get("overall_summary", "")

    def _run_sequence_check_pass(
        self,
        pass_def: dict,
        all_items: list[dict],
        precomputed: dict,
    ) -> dict:
        """Run sequence check pass (SPIKES pass 7)."""
        scorable = [
            i for i in all_items if not i.get("_sequence_check")
        ]
        items_json = json.dumps(scorable, ensure_ascii=False, indent=2)

        try:
            template = self._jinja_env.get_template(pass_def["template"])
        except Exception as e:
            self.logger.error(f"Sequence check template load failed: {e}")
            return {"sequence_correct": None, "sequence_note": "template error"}

        prompt = template.render(
            items_json=items_json,
            scoring_preamble=self.instrument.get("scoring_preamble", ""),
        )
        pass_cfg = {**self.cfg, "max_tokens": pass_def.get("max_tokens", 2000)}
        raw = self.backend.generate(prompt, pass_cfg)

        text = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {"sequence_correct": None, "sequence_note": "parse error"}

    def _compute_result(
        self,
        all_items: list[dict],
        instrument: dict,
        overall_note: str,
    ) -> dict:
        """Build standardized ScoringResult dict."""
        # Extract sequence check if present
        sequence_check = None
        scored_items = []
        for item in all_items:
            if item.get("_sequence_check"):
                sequence_check = item["_sequence_check"]
            else:
                scored_items.append(item)

        # Build item_def lookup for max_score and metadata
        item_defs: dict[str, dict] = {
            d["id"]: d for d in instrument.get("items", [])
        }

        result_items = []
        raw_score = 0
        max_possible = 0
        by_category: dict[str, dict] = {}
        critical_misses = []
        critical_false_positives = []

        for item in scored_items:
            item_id = item.get("id") or item.get("item")
            if not item_id:
                continue

            idef = item_defs.get(item_id, {})
            scale = idef.get("scale", [0, 1, 2])
            numeric_scale = [v for v in scale if isinstance(v, (int, float))]
            item_max = int(max(numeric_scale)) if numeric_scale else 2

            rating = item.get("rating")
            rating_numeric = None
            if isinstance(rating, (int, float)):
                rating_numeric = int(rating)
            elif rating == "NA":
                rating_numeric = None

            # Scale labels
            scale_labels = idef.get("scale_labels", {})
            if rating_numeric is not None:
                rating_label = scale_labels.get(
                    str(rating_numeric),
                    scale_labels.get(rating_numeric, "")
                )
            else:
                rating_label = "NA"

            category = idef.get("category", item.get("category", ""))
            critical = idef.get("critical", False)

            # Accumulate scores (skip NA items)
            if rating_numeric is not None:
                raw_score += rating_numeric
                max_possible += item_max

                if category:
                    if category not in by_category:
                        by_category[category] = {"raw": 0, "max": 0}
                    by_category[category]["raw"] += rating_numeric
                    by_category[category]["max"] += item_max

            # Critical item checks
            if critical and rating_numeric == 0:
                critical_misses.append({
                    "id": item_id,
                    "name": idef.get("name", item.get("name", "")),
                    "category": category,
                    "justification": item.get("justification", ""),
                })
            if critical and rating_numeric == item_max and item_max > 0:
                # Not a false positive — just noting full score on critical
                pass

            result_items.append({
                "id": item_id,
                "name": idef.get("name", item.get("name", "")),
                "category": category,
                "rating": rating if rating == "NA" else rating_numeric,
                "max_score": item_max if rating != "NA" else None,
                "rating_label": rating_label,
                "justification": item.get("justification", ""),
                "evidence": item.get("evidence", []),
                "validation_flags": item.get("validation_flags", []),
            })

        # Compute by_category percentages
        for cat_data in by_category.values():
            cat_data["pct"] = (
                round(cat_data["raw"] / cat_data["max"] * 100, 1)
                if cat_data["max"] > 0 else 0.0
            )

        normalised = (
            round(raw_score / max_possible * 100, 1)
            if max_possible > 0 else 0.0
        )

        result = {
            "instrument_id": instrument.get("id", ""),
            "instrument_name": instrument.get("name", ""),
            "domain": instrument.get("domain", ""),
            "items": result_items,
            "summary": {
                "raw_score": raw_score,
                "max_possible_score": max_possible,
                "normalised_score_pct": normalised,
                "by_category": by_category,
            },
            "critical_misses": critical_misses,
            "overall_note": overall_note,
        }

        # Inject sequence_check fields if present (SPIKES)
        if sequence_check:
            result["sequence_correct"] = sequence_check.get("sequence_correct")
            result["sequence_note"] = sequence_check.get("sequence_note", "")

        return result
