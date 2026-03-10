"""
Stage 7: Translation
======================
Translates the analysis JSON (and optionally clinical_content JSON) from German
to English, producing parallel *_en.json files with identical structure and all
numeric/boolean/ID fields untouched.

Strategy:
  1. Walk the analysis dict and extract all translatable string values with a
     stable path key.
  2. Send all strings in a single batched API call as a numbered list.
  3. Parse the numbered responses and substitute back into a deep copy of the
     original dict.
  4. Write <report_id>_en.json alongside the original.

Config (pipeline_config.yaml → stages → translate):
  enabled: true
  fields_to_skip:           # exact key names whose values are never translated
    - session_id
    - report_id
    - generated_at
    - id                    # item IDs like CC01, CS02, S1 etc.
    - item                  # LUCAS item letter
    - category              # already English
    - rating_label          # controlled vocabulary — kept in DE for now
    - reliability
    - method_note
  min_length: 10            # strings shorter than this are skipped
  model: "claude-opus-4-6"  # or claude-haiku-4-5-20251001 for speed/cost
  max_tokens: 4096
"""

import copy
import json
import re
from pathlib import Path
from typing import Any

from stages.base import BaseStage


# ── Keys whose values should never be translated ─────────────────────────────
_DEFAULT_SKIP_KEYS = {
    "session_id", "report_id", "generated_at",
    "id",           # CS01, CC03, S1, …
    "item",         # A, B, C, …
    "category",     # already English taxonomy labels
    "rating_label", # Unacceptable / Borderline / Competent
    "reliability",  # high / moderate / low
    "method_note",  # technical metadata
    "speaker",      # SPEAKER_00 labels
    "_case_module_used", "_salvaged",
    "sequence_correct",
    "has_critical_miss",
}

# ── Keys that ARE translatable free-text ──────────────────────────────────────
_TRANSLATE_KEYS = {
    "note", "evidence",        # SPIKES steps
    "justification",           # LUCAS + clinical
    "strengths", "gaps", "next_steps",
    "overall_summary",         # LUCAS summary
    "overall_spikes_note", "sequence_note",
    "overall_clinical_note",
    "name",                    # scenario module item names (CS01 etc.)
    "overall_summary",
}


class TranslationStage(BaseStage):
    """Translate German analysis output to English via a batched LLM call."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("translate")
        cfg = self._get_stage_config("llm")
        if not cfg.get("enabled", True):
            self.logger.info("Translation stage disabled — skipping.")
            return ctx

        output_dir = Path(ctx["output_base"]) / "06_translate"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Locate source analysis JSON ----
        analysis_path = self._find_analysis_json(ctx)
        if not analysis_path:
            self.logger.warning("No analysis.json found — skipping translation.")
            return ctx

        with open(analysis_path, encoding="utf-8") as f:
            source = json.load(f)

        self.logger.info(f"Translating {analysis_path.name} …")

        # ---- Extract all translatable strings ----
        skip_keys = set(cfg.get("fields_to_skip", [])) | _DEFAULT_SKIP_KEYS
        min_len   = cfg.get("min_length", 10)

        strings: list[str] = []
        paths:   list[tuple] = []        # parallel list of (path, index_in_list_if_list)

        self._collect(source, strings, paths, skip_keys, min_len)
        self.logger.info(f"Collected {len(strings)} translatable strings.")

        if not strings:
            self.logger.warning("Nothing to translate.")
            return ctx

        # ---- Batch translate ----
        translations = self._batch_translate(strings, cfg, ctx)

        if len(translations) != len(strings):
            self.logger.error(
                f"Translation count mismatch: got {len(translations)}, "
                f"expected {len(strings)}. Aborting translation."
            )
            return ctx

        # ---- Substitute back into deep copy ----
        translated = copy.deepcopy(source)
        self._substitute(translated, translations, skip_keys, min_len, [0])

        # ---- Write output ----
        report_id = source.get("report_id", ctx.get("session_id", "report"))
        out_path = output_dir / f"{report_id}_en.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, indent=2, ensure_ascii=False)

        # Also write to the global reports dir for easy access
        reports_dir = Path(ctx["config"]["paths"]["output_dir"])
        global_out = reports_dir / ctx["session_id"] / f"{report_id}_en.json"
        global_out.parent.mkdir(parents=True, exist_ok=True)
        with open(global_out, "w", encoding="utf-8") as f:
            json.dump(translated, f, indent=2, ensure_ascii=False)

        ctx["artifacts"]["analysis_en_path"] = str(out_path)
        self.logger.info(f"Translation written: {out_path.name}")
        return ctx

    # ──────────────────────────────────────────────────────────────────────────
    # String collection — walk the dict and gather all translatable leaf strings
    # ──────────────────────────────────────────────────────────────────────────

    def _collect(
        self,
        obj: Any,
        strings: list[str],
        paths: list,
        skip_keys: set,
        min_len: int,
        _path: tuple = (),
    ):
        """
        Recursively walk obj and append every translatable string to `strings`,
        with its address tuple appended to `paths`.

        Address tuples encode navigation steps:
          str  → dict key
          int  → list index
        """
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in skip_keys:
                    continue
                self._collect(v, strings, paths, skip_keys, min_len, _path + (k,))

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._collect(item, strings, paths, skip_keys, min_len, _path + (i,))

        elif isinstance(obj, str):
            if len(obj) >= min_len:
                strings.append(obj)
                paths.append(_path)

    def _substitute(self, obj: Any, translations: list[str], skip_keys: set, min_len: int, _idx: list):
        """
        Re-walk obj in the same traversal order as _collect and replace each
        translatable string with its translation. Uses a shared mutable counter
        _idx = [int] so the single pass through translations stays in sync.
        """
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in skip_keys:
                    continue
                obj[k] = self._substitute_value(obj[k], translations, skip_keys, min_len, _idx)
        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = self._substitute_value(obj[i], translations, skip_keys, min_len, _idx)

    def _substitute_value(self, val: Any, translations: list[str], skip_keys: set, min_len: int, _idx: list) -> Any:
        if isinstance(val, str):
            if len(val) >= min_len:
                result = translations[_idx[0]]
                _idx[0] += 1
                return result
            return val
        elif isinstance(val, (dict, list)):
            self._substitute(val, translations, skip_keys, min_len, _idx)
            return val
        return val

    # ──────────────────────────────────────────────────────────────────────────
    # Batched LLM translation
    # ──────────────────────────────────────────────────────────────────────────

    def _batch_translate(self, strings: list[str], cfg: dict, ctx: dict) -> list[str]:
        """
        Send all strings as a numbered list in a single LLM call
        using the same backend as Stage 5 (llama_cpp or vllm).
        """
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(strings))

        prompt = f"""You are translating a medical simulation feedback report from German to English.

    Translate each numbered item below from German to English.
    - Preserve all medical terminology accurately.
    - Preserve timestamps like [00:53.1-00:57.9] exactly as-is.
    - Preserve speaker labels like SPEAKER_00 exactly as-is.
    - Preserve any existing English phrases within a string.
    - Do NOT add explanations or commentary.
    - Return ONLY a numbered list in exactly the same format: "N. <translation>"
    - Every number from 1 to {len(strings)} must appear exactly once.

    STRINGS TO TRANSLATE:
    {numbered}
    """

        self.logger.info(
            f"Translating {len(strings)} strings using backend={cfg.get('backend', 'llama_cpp')}"
        )

        raw = self._run_llm(prompt, cfg)

        return self._parse_numbered_list(raw, len(strings))
    
    def _run_llm(self, prompt: str, cfg: dict) -> str:
        backend = cfg.get("backend", "llama_cpp")
        if backend == "llama_cpp":
            return self._run_llama_cpp(prompt, cfg)
        elif backend == "vllm":
            return self._run_vllm(prompt, cfg)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")
        
    def _run_llama_cpp(self, prompt: str, cfg: dict) -> str:
        from llama_cpp import Llama

        if not hasattr(self, "_llama_model") or self._llama_model is None:
            model_path = cfg["model_path"]
            self.logger.info(f"Loading llama model: {model_path}")
            self._llama_model = Llama(
                model_path=model_path,
                n_ctx=cfg.get("context_length", 8192),
                n_gpu_layers=-1,
                seed=cfg.get("seed", 42),
                verbose=False,
            )

        response = self._llama_model(
            prompt,
            max_tokens=cfg.get("max_tokens", 4096),
            temperature=0.0,   # IMPORTANT for deterministic translation
            top_p=1.0,
        )

        return response["choices"][0]["text"]


    def _parse_numbered_list(self, text: str, expected: int) -> list[str]:
        """
        Parse "N. <text>" lines from the LLM response.
        Falls back to the original index if a line is missing.
        """
        results = {}
        # Match lines like "1. some text" — text may span continuation lines
        pattern = re.compile(r"^(\d+)\.\s+(.+)", re.MULTILINE)
        for m in pattern.finditer(text):
            idx = int(m.group(1)) - 1   # 0-based
            results[idx] = m.group(2).strip()

        # Fill any gaps (shouldn't happen with good model output)
        output = []
        for i in range(expected):
            if i in results:
                output.append(results[i])
            else:
                self.logger.warning(f"Missing translation for string index {i+1}")
                output.append(f"[translation missing for item {i+1}]")

        return output

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _find_analysis_json(self, ctx: dict) -> Path | None:
        """Locate the analysis JSON — prefer the stage output, fall back to ctx artifact."""
        # From s5 output directory
        analysis_dir = Path(ctx["output_base"]) / "05_analysis"
        candidates = list(analysis_dir.glob("*.json")) if analysis_dir.exists() else []
        # Exclude checkpoints
        candidates = [p for p in candidates if not p.name.startswith(".")]
        if candidates:
            # Prefer the one named analysis.json or the most recent
            named = [p for p in candidates if p.name == "analysis.json"]
            return named[0] if named else sorted(candidates)[-1]

        # From ctx artifacts
        ap = ctx.get("artifacts", {}).get("analysis_path")
        if ap and Path(ap).exists():
            return Path(ap)

        return None