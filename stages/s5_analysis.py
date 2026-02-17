"""
Stage 4: LLM-Based Analysis & Rating
======================================
- Loads frozen prompt template
- Injects structured profile (transcript + features + scenario metadata)
- Runs local LLM (llama-cpp-python or vLLM) with deterministic settings
- Parses structured output: ordinal ratings + narrative per domain
- Validates output schema before passing downstream
"""

import json
import re
from pathlib import Path

from stages.base import BaseStage


class LLMAnalysisStage(BaseStage):
    """Run LLM-based assessment across all configured domains."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("llm")
        output_dir = Path(ctx["output_base"]) / "04_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        llm_profile = ctx["artifacts"]["llm_profile"]

        # Load prompt template
        prompt = self._build_prompt(cfg, llm_profile, ctx)
        self.logger.info(f"Prompt length: {len(prompt)} chars")

        # Save the assembled prompt (for audit / freeze verification)
        prompt_path = output_dir / "assembled_prompt.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt)

        # Run LLM inference
        backend = cfg.get("backend", "llama_cpp")
        self.logger.info(f"Running LLM inference (backend: {backend})")

        if backend == "llama_cpp":
            raw_output = self._run_llama_cpp(prompt, cfg)
        elif backend == "vllm":
            raw_output = self._run_vllm(prompt, cfg)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")

        # Save raw output
        raw_path = output_dir / "llm_raw_output.txt"
        with open(raw_path, "w") as f:
            f.write(raw_output)

        self.logger.info(f"LLM output length: {len(raw_output)} chars")

        # Parse structured response
        analysis = self._parse_output(raw_output, cfg)

        # Validate analysis
        self._validate_analysis(analysis, cfg)

        # Save parsed analysis
        analysis_path = output_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        ctx["artifacts"]["analysis"] = analysis
        ctx["artifacts"]["analysis_path"] = str(analysis_path)
        ctx["artifacts"]["llm_raw_output"] = raw_output

        return ctx

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(self, cfg: dict, profile: dict, ctx: dict) -> str:
        """Build the analysis prompt from the Jinja2 template."""
        template_path = cfg.get("prompt_template", "templates/analysis_prompt.j2")

        if Path(template_path).exists():
            try:
                import jinja2

                with open(template_path) as f:
                    template = jinja2.Template(f.read())

                return template.render(
                    profile=profile,
                    domains=cfg.get("domains", []),
                    session_id=ctx.get("session_id", "unknown"),
                    transcript=json.dumps(
                        profile.get("diarized_transcript", []),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    interaction=json.dumps(
                        profile.get("interaction", {}),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    conversation_phases=json.dumps(
                        profile.get("conversation_phases", []),
                        indent=2,
                        ensure_ascii=False,
                    ),
                )
            except ImportError:
                self.logger.warning("Jinja2 not installed, using fallback prompt")

        # Fallback: build prompt directly
        return self._build_fallback_prompt(cfg, profile)

    def _build_fallback_prompt(self, cfg: dict, profile: dict) -> str:
        """Construct prompt without Jinja2."""
        domains = cfg.get("domains", [])

        domain_instructions = ""
        for d in domains:
            scale = d.get("scale", {})
            labels = scale.get("labels", {})
            label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
            domain_instructions += (
                f"\n### {d['name']}\n"
                f"Description: {d['description']}\n"
                f"Scale: {scale.get('min', 1)}-{scale.get('max', 5)} ({label_str})\n"
            )

        transcript_text = ""
        for seg in profile.get("diarized_transcript", []):
            transcript_text += f"[{seg['speaker']}] ({seg['start']:.1f}-{seg['end']:.1f}s): {seg['text']}\n"

        interaction_summary = json.dumps(
            profile.get("interaction", {}), indent=2, ensure_ascii=False
        )

        prompt = f"""You are an expert medical education assessor evaluating a paediatric simulation scenario. Your task is to generate a structured feedback report based on the simulation recording transcript and interaction data provided below.

## Assessment Domains
{domain_instructions}

## Diarized Transcript
{transcript_text}

## Interaction Metrics
{interaction_summary}

## Instructions
For EACH domain listed above, provide:
1. **rating**: An integer rating on the specified scale
2. **strengths**: 2-4 specific strengths observed, with evidence from the transcript
3. **gaps**: 1-3 specific areas for improvement, with evidence
4. **next_steps**: 1-3 concrete, actionable recommendations for the learner

Respond ONLY with a valid JSON object in the following structure (no markdown, no extra text):
{{
  "domains": [
    {{
      "name": "<domain_name>",
      "rating": <integer>,
      "strengths": ["<strength 1>", "<strength 2>"],
      "gaps": ["<gap 1>"],
      "next_steps": ["<recommendation 1>"]
    }}
  ],
  "overall_summary": "<2-3 sentence overall assessment>"
}}
"""
        return prompt

    # ------------------------------------------------------------------
    # LLM backends
    # ------------------------------------------------------------------
    def _run_llama_cpp(self, prompt: str, cfg: dict) -> str:
        """Run inference via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            self.logger.error(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
            raise

        model_path = cfg["model_path"]
        self.logger.info(f"Loading model: {model_path}")

        llm = Llama(
            model_path=model_path,
            n_ctx=cfg.get("context_length", 8192),
            n_gpu_layers=-1,  # Offload all layers to GPU if available
            seed=cfg.get("seed", 42),
            verbose=False,
        )

        response = llm(
            prompt,
            max_tokens=cfg.get("max_tokens", 4096),
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            stop=None,
        )

        return response["choices"][0]["text"]

    def _run_vllm(self, prompt: str, cfg: dict) -> str:
        """Run inference via vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            self.logger.error(
                "vLLM not installed. "
                "Install with: pip install vllm"
            )
            raise

        model_name = cfg.get("model_name", cfg.get("model_path"))
        self.logger.info(f"Loading vLLM model: {model_name}")

        llm = LLM(
            model=model_name,
            max_model_len=cfg.get("context_length", 8192),
            seed=cfg.get("seed", 42),
        )

        params = SamplingParams(
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            max_tokens=cfg.get("max_tokens", 4096),
        )

        outputs = llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_output(self, raw: str, cfg: dict) -> dict:
        """Parse the LLM JSON response, handling common formatting issues."""
        # Try to extract JSON from the response
        text = raw.strip()

        # Remove markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.error("Failed to parse LLM output as JSON")
        return {
            "domains": [],
            "overall_summary": "",
            "parse_error": True,
            "raw_output": raw[:2000],
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_analysis(self, analysis: dict, cfg: dict):
        """Validate the parsed analysis against expected schema."""
        if analysis.get("parse_error"):
            self.logger.error("Analysis has parse errors — report may be incomplete")
            return

        domains_cfg = {d["name"]: d for d in cfg.get("domains", [])}
        domains_out = analysis.get("domains", [])

        if len(domains_out) != len(domains_cfg):
            self.logger.warning(
                f"Expected {len(domains_cfg)} domains, got {len(domains_out)}"
            )

        for d in domains_out:
            name = d.get("name", "unknown")
            dcfg = domains_cfg.get(name, {})
            scale = dcfg.get("scale", {})

            rating = d.get("rating")
            if rating is not None:
                min_val = scale.get("min", 1)
                max_val = scale.get("max", 5)
                if not (min_val <= rating <= max_val):
                    self.logger.warning(
                        f"Domain '{name}': rating {rating} outside scale "
                        f"[{min_val}, {max_val}]"
                    )

            if not d.get("strengths"):
                self.logger.warning(f"Domain '{name}': no strengths listed")

            if not d.get("gaps"):
                self.logger.warning(f"Domain '{name}': no gaps listed")

            if not d.get("next_steps"):
                self.logger.warning(f"Domain '{name}': no next_steps listed")
