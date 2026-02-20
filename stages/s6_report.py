"""
Stage 6: Standardized Report Generation
=========================================
Renders the LUCAS + SPIKES LLM analysis into the standardized feedback
report format.

Input (from ctx["artifacts"]["analysis"]):
  - lucas_items  list[dict]  — A-J items, each with rating/score,
                               rating_label, justification, evidence,
                               strengths, gaps, next_steps
  - total_score  int         — sum of all ratings (max 18)
  - overall_summary str      — 3-5 sentence holistic assessment

Supports both output schemas:
  - German template schema: "lucas_items" / "item" / "rating"
  - Fallback English schema: "items"      / "id"   / "score"

Output formats:
  - JSON  (primary, always)
  - HTML  (if "html"  in cfg.additional_formats)
  - PDF   (if "pdf"   in cfg.additional_formats, requires weasyprint)

Report IDs use a blinded label for evaluation (e.g. REPORT_session123).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from stages.base import BaseStage

# LUCAS section groupings for report layout
_LUCAS_SECTIONS = {
    "Introductions":               ["A", "B"],
    "General":                     ["C", "D", "E", "F", "G", "H"],
    "Professional Behaviour and Conduct": ["I", "J"],
}

# Mapping item id → max score (for display)
_ITEM_MAX_SCORE = {
    "A": 1, "B": 1,
    "C": 2, "D": 2, "E": 2, "F": 2, "G": 2, "H": 2,
    "I": 2, "J": 2,
}


def _normalise_items(analysis: dict) -> list[dict]:
    """
    Convert either schema into a uniform list of item dicts:
      {
        item_id, name, category, max_score,
        rating (int), rating_label,
        justification, evidence, strengths, gaps, next_steps
      }
    Returns items sorted A → J.
    """
    raw = analysis.get("lucas_items") or analysis.get("items", [])
    ORDER = list("ABCDEFGHIJ")
    normalised = []

    for item in raw:
        item_id = str(item.get("item") or item.get("id") or "?").upper()

        # Resolve score — may come back as string from LLM
        score_raw = (
            item.get("rating")
            if item.get("rating") is not None
            else item.get("score")
        )
        try:
            score = int(score_raw)
        except (TypeError, ValueError):
            score = None

        normalised.append({
            "item_id":      item_id,
            "name":         item.get("name", ""),
            "category":     item.get("category", ""),
            "max_score":    item.get("max_score") or _ITEM_MAX_SCORE.get(item_id, 2),
            "rating":       score,
            "rating_label": item.get("rating_label", ""),
            "justification": item.get("justification", ""),
            "evidence":     item.get("evidence", []),
            "strengths":    item.get("strengths", []),
            "gaps":         item.get("gaps", []),
            "next_steps":   item.get("next_steps", []),
        })

    normalised.sort(key=lambda x: ORDER.index(x["item_id"])
                    if x["item_id"] in ORDER else 99)
    return normalised


class ReportGenerationStage(BaseStage):
    """Generate the final standardized LUCAS feedback report."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("report")
        output_dir = Path(ctx["output_base"]) / "06_report"
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis   = ctx["artifacts"]["analysis"]
        session_id = ctx["session_id"]

        report = self._build_report(analysis, session_id, cfg, ctx)

        # ── JSON (always) ──
        json_path = output_dir / f"{report['report_id']}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"JSON report: {json_path}")

        additional = cfg.get("additional_formats", ["pdf"])

        # ── HTML ──
        html_path = None
        if "html" in additional:
            html_path = output_dir / f"{report['report_id']}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self._render_html(report, cfg))
            self.logger.info(f"HTML report: {html_path}")

        # ── PDF ──
        pdf_path = None
        if "pdf" in additional and html_path:
            try:
                pdf_path = output_dir / f"{report['report_id']}.pdf"
                self._render_pdf(str(html_path), str(pdf_path))
                self.logger.info(f"PDF report: {pdf_path}")
            except Exception as e:
                self.logger.warning(f"PDF generation failed: {e}")

        # Copy to top-level reports directory
        reports_dir = Path(ctx["config"]["paths"]["output_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        final_path = reports_dir / f"{report['report_id']}.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        ctx["artifacts"]["report"]           = report
        ctx["artifacts"]["report_path"]      = str(final_path)
        ctx["artifacts"]["report_json_path"] = str(json_path)
        if html_path:
            ctx["artifacts"]["report_html_path"] = str(html_path)
        if pdf_path:
            ctx["artifacts"]["report_pdf_path"]  = str(pdf_path)

        return ctx

    # ------------------------------------------------------------------
    # Report construction
    # ------------------------------------------------------------------
    def _build_report(
        self, analysis: dict, session_id: str, cfg: dict, ctx: dict
    ) -> dict:
        """
        analysis is the combined dict written by Stage 5:
          {
            "spikes_annotation": {...},
            "lucas_analysis":    {"lucas_items": [...], "total_score": 16, ...},
            "lucas_total_score": 16,
            "lucas_max_score":   18,
          }
        Unwrap lucas_analysis before passing to _normalise_items.
        Spikes is taken directly from the combined dict (or ctx fallback).
        """
        label_prefix = cfg.get("label_prefix", "REPORT")
        report_id    = f"{session_id}"

        # ── Unwrap the nested structure from Stage 5 ──
        # ctx["artifacts"]["lucas_analysis"] is the direct LUCAS output;
        # analysis["lucas_analysis"] is the same thing via the combined dict.
        # Support both paths so the stage works whether called from a live
        # pipeline run (ctx populated) or reconstructed from analysis.json.
        lucas = (
            ctx["artifacts"].get("lucas_analysis")       # preferred: direct key
            or analysis.get("lucas_analysis")            # fallback: unwrap combined
            or analysis                                  # last resort: flat schema
        )

        spikes = (
            ctx["artifacts"].get("spikes_annotation")    # preferred: direct key
            or analysis.get("spikes_annotation")         # fallback: unwrap combined
        )

        items = _normalise_items(lucas)

        # Recompute total from normalised items — always authoritative
        total_computed = sum(
            it["rating"] for it in items if isinstance(it["rating"], int)
        )

        # Cross-check against both the combined dict and the lucas dict
        claimed_raw = (
            analysis.get("lucas_total_score")   # combined dict key
            or lucas.get("total_score")          # flat lucas key
        )
        if claimed_raw is not None:
            try:
                claimed_int = int(claimed_raw)
            except (TypeError, ValueError):
                claimed_int = None
            if claimed_int is not None and claimed_int != total_computed:
                self.logger.warning(
                    f"Report: claimed total_score={claimed_int} "
                    f"but computed={total_computed}. Using computed."
                )

        report = {
            "report_id":            report_id,
            "generated_at":         datetime.now(timezone.utc).isoformat(),
            "pipeline_version":     ctx["manifest"].get("pipeline_version", "unknown"),
            "manifest_digest":      ctx["manifest"].get("manifest_digest", "unknown"),
            "lucas_total_score":    total_computed,
            "lucas_max_score":      18,
            "score_percentage":     round(total_computed / 18 * 100, 1),
            "overall_summary":      lucas.get("overall_summary", ""),
            "items":                items,
        }

        if cfg.get("include_timestamps", True):
            report["processing_times"] = ctx.get("timestamps", {})

        if spikes and cfg.get("include_spikes", True):
            report["spikes_annotation"] = spikes

        return report

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------
    def _render_html(self, report: dict, cfg: dict) -> str:
        template_path = cfg.get("template")
        if template_path and Path(template_path).exists():
            try:
                import jinja2
                with open(template_path, encoding="utf-8") as f:
                    template = jinja2.Template(f.read())
                return template.render(report=report)
            except ImportError:
                self.logger.warning(
                    "Jinja2 not available — using built-in HTML template"
                )
        return self._builtin_html(report)

    @staticmethod
    def _builtin_html(report: dict) -> str:
        """
        Professional built-in HTML report.

        Features
        --------
        - Institutional letterhead with logo placeholder
        - Editable student information panel (name, ID, semester, date,
          location, assessor) — fields are browser-editable, print-ready
        - LUCAS score dashboard with per-item progress bars
        - Grouped LUCAS item cards (evidence, strengths, gaps, next steps)
        - SPIKES protocol table
        - Assessor notes / observation log section (blank lines for hand-
          written annotation; also editable in browser before printing)
        - Clean print stylesheet (no backgrounds, no shadows)
        """

        # ── helpers ──────────────────────────────────────────────────────
        def _ul(lst: list, css: str = "") -> str:
            if not lst:
                return "<span class='empty'>—</span>"
            cls = f' class="{css}"' if css else ""
            return "<ul{}>".format(cls) + "".join(
                f"<li>{i}</li>" for i in lst
            ) + "</ul>"

        def _bar(score, max_score) -> str:
            if score is None or max_score == 0:
                return ""
            pct = round(score / max_score * 100)
            return (
                f'<div class="bar-wrap">'
                f'<div class="bar-fill" style="width:{pct}%"></div>'
                f'</div>'
            )

        def _lbl_cls(label: str) -> str:
            l = (label or "").lower()
            if "unacceptable" in l: return "lbl-bad"
            if "borderline"   in l: return "lbl-warn"
            if "competent"    in l: return "lbl-good"
            return "lbl-uk"

        def _scorecard_row(it: dict) -> str:
            score     = it["rating"]
            max_score = it["max_score"]
            label     = it.get("rating_label", "")
            score_str = f"{score}/{max_score}" if score is not None else "—"
            return (
                f'<div class="sc-row">'
                f'<span class="sc-id">{it["item_id"]}</span>'
                f'<span class="sc-name">{it["name"]}</span>'
                f'<span class="sc-bar">{_bar(score, max_score)}</span>'
                f'<span class="sc-score">{score_str}</span>'
                f'<span class="rating-lbl {_lbl_cls(label)}">{label}</span>'
                f'</div>'
            )

        # ── scorecard ────────────────────────────────────────────────────
        items_by_id   = {it["item_id"]: it for it in report.get("items", [])}
        scorecard_html = ""
        for section_name, ids in _LUCAS_SECTIONS.items():
            rows = "".join(
                _scorecard_row(items_by_id[i])
                for i in ids if i in items_by_id
            )
            if rows:
                scorecard_html += (
                    f'<div class="sc-section-label">{section_name}</div>'
                    + rows
                )

        # ── detailed item cards ──────────────────────────────────────────
        sections_html = ""
        for section_name, ids in _LUCAS_SECTIONS.items():
            section_items = [items_by_id[i] for i in ids if i in items_by_id]
            if not section_items:
                continue
            cards = ""
            for it in section_items:
                score     = it["rating"]
                max_score = it["max_score"]
                label     = it.get("rating_label", "")
                score_str = f"{score} / {max_score}" if score is not None else "—"

                ev_block = ""
                if it.get("evidence"):
                    ev_lis = "".join(
                        f'<li><code>{e}</code></li>' for e in it["evidence"]
                    )
                    ev_block = (
                        f'<div class="card-sub">'
                        f'<h5>Evidence</h5><ul class="ev-list">{ev_lis}</ul>'
                        f'</div>'
                    )

                cards += f"""
              <div class="item-card">
                <div class="card-head">
                  <div class="card-title">
                    <span class="card-id">{it["item_id"]}</span>
                    <div>
                      <div class="card-name">{it["name"]}</div>
                      <div class="card-cat">{it.get("category","")}</div>
                    </div>
                  </div>
                  <div class="card-score-block">
                    {_bar(score, max_score)}
                    <span class="card-score-num">{score_str}</span>
                    <span class="rating-lbl {_lbl_cls(label)}">{label}</span>
                  </div>
                </div>
                <p class="card-just">{it.get("justification","")}</p>
                {ev_block}
                <div class="card-cols">
                  <div class="card-col">
                    <h5>&#10003; Strengths</h5>
                    {_ul(it.get("strengths",[]))}
                  </div>
                  <div class="card-col">
                    <h5>&#9651; Areas for Improvement</h5>
                    {_ul(it.get("gaps",[]))}
                  </div>
                  <div class="card-col">
                    <h5>&#8594; Next Steps</h5>
                    {_ul(it.get("next_steps",[]))}
                  </div>
                </div>
              </div>"""

            sections_html += f"""
            <div class="section-block">
              <div class="section-header">
                <span class="section-title">{section_name}</span>
              </div>
              {cards}
            </div>"""

        # ── SPIKES table ─────────────────────────────────────────────────
        spikes_html = ""
        spikes = report.get("spikes_annotation")
        if spikes and not spikes.get("parse_error"):
            rows = ""
            for step in spikes.get("steps", []):
                present = step.get("present", False)
                icon    = "✓" if present else "✗"
                cls     = "sp-yes" if present else "sp-no"
                rows += (
                    f'<tr class="{cls}">'
                    f'<td class="sp-id"><strong>{step.get("id","")}</strong></td>'
                    f'<td>{step.get("name","")}</td>'
                    f'<td class="sp-icon">{icon}</td>'
                    f'<td class="sp-note">{step.get("note","")}</td>'
                    f'</tr>'
                )
            seq_ok   = spikes.get("sequence_correct", False)
            seq_note = spikes.get("sequence_note", "")
            overall  = spikes.get("overall_spikes_note", "")
            seq_cls  = "seq-ok" if seq_ok else "seq-bad"

            spikes_html = f"""
            <div class="section-block">
              <div class="section-header">
                <span class="section-title">SPIKES Protocol Analysis</span>
              </div>
              <div class="card-plain">
                <table class="sp-table">
                  <thead>
                    <tr>
                      <th>Step</th><th>Name</th>
                      <th style="width:80px;text-align:center">Present</th>
                      <th>Assessor Note</th>
                    </tr>
                  </thead>
                  <tbody>{rows}</tbody>
                </table>
                <div class="sp-seq {seq_cls}">
                  <strong>Sequence:</strong>
                  {"Correct &nbsp;✓" if seq_ok else f"Issues detected &mdash; {seq_note}"}
                </div>
                <p class="sp-overall">{overall}</p>
              </div>
            </div>"""

        # ── top-level data ────────────────────────────────────────────────
        total   = report.get("lucas_total_score", "—")
        max_tot = report.get("lucas_max_score", 18)
        pct     = report.get("score_percentage", "—")
        rid     = report.get("report_id", "")
        gen_at  = (report.get("generated_at", "") or "")[:19].replace("T", " ")
        summary = report.get("overall_summary", "")

        # ── blank log lines ───────────────────────────────────────────────
        log_lines = "".join(
            '<div class="log-line"></div>' for _ in range(12)
        )

        return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feedback Report &mdash; {rid}</title>
<style>
/* ═══════════════════════════════════════════════════════════════
   Base
═══════════════════════════════════════════════════════════════ */
*, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
:root {{
  /* ── RWTH Aachen University Official Corporate Design Palette ──────
     Source: RWTH CD guidelines (HKS / Pantone refs)
     Blue 100% = HKS-44K = Pantone 2945C — primary identity colour.
  ─────────────────────────────────────────────────────────────────── */
  --rwth-blue:        #00549F;   /* Blue 100%  */
  --rwth-blue-75:     #407FB7;   /* Blue  75%  */
  --rwth-blue-50:     #8EBAE5;   /* Blue  50%  */
  --rwth-blue-25:     #C7DDF2;   /* Blue  25%  */
  --rwth-blue-10:     #E8F1FA;   /* Blue  10%  */
  --rwth-petrol:      #006165;   /* Petrol     */
  --rwth-green:       #57AB27;   /* Green      */
  --rwth-green-25:    #DDEBCE;   /* Green  25% */
  --rwth-bordeaux:    #A11035;   /* Bordeaux   */
  --rwth-bordeaux-25: #F5C9D0;   /* Bordeaux 25%*/
  --rwth-orange:      #F6A800;   /* Orange     */
  --rwth-orange-25:   #FDEEBF;   /* Orange 25% */

  /* ── Semantic aliases (used throughout) ─────────────────────────── */
  --navy:   #00549F;             /* RWTH Blue 100%                      */
  --accent: #00549F;             /* RWTH Blue 100%                      */
  --gold:   #F6A800;             /* RWTH Orange — accent bar            */
  --good:   #006165;             /* RWTH Petrol — competent             */
  --warn:   #7a5000;             /* Orange darkened for text contrast   */
  --bad:    #A11035;             /* RWTH Bordeaux — unacceptable        */
  --bg:     #EEF3F9;             /* Blue-10 tinted page background      */
  --card:   #ffffff;
  --border: #C7DDF2;             /* Blue-25 — borders                   */
  --text:   #1a1a1a;
  --muted:  #4a5568;
  --radius: 6px;
}}
body {{
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 13.5px;
  line-height: 1.65;
}}
a {{ color: var(--accent); }}

/* ═══════════════════════════════════════════════════════════════
   Page wrapper
═══════════════════════════════════════════════════════════════ */
.page {{
  max-width: 960px;
  margin: 0 auto;
  padding: 28px 20px 72px;
}}

/* ═══════════════════════════════════════════════════════════════
   ① Letterhead / cover
═══════════════════════════════════════════════════════════════ */
.letterhead {{
  background: var(--rwth-blue);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
}}
.lh-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 32px 18px;
}}
.lh-titles {{
  flex: 1;
  padding: 0 24px;
}}
.lh-titles h1 {{
  font-size: 19px;
  font-weight: 700;
  color: #fff;
  letter-spacing: .03em;
  margin-bottom: 3px;
}}
.lh-titles .subtitle {{
  font-size: 12px;
  color: rgba(200,225,255,.75);
  letter-spacing: .05em;
  text-transform: uppercase;
}}
.lh-score-panel {{
  text-align: right;
  flex-shrink: 0;
}}
.lh-score-label {{
  font-size: 11px;
  color: rgba(190,220,255,.65);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 4px;
}}
.lh-score-num {{
  font-size: 48px;
  font-weight: 800;
  color: #fff;
  line-height: 1;
}}
.lh-score-denom {{
  font-size: 20px;
  color: rgba(190,220,255,.7);
}}
.lh-score-pct {{
  font-size: 13px;
  color: var(--rwth-orange);
  margin-top: 3px;
}}

/* ═══════════════════════════════════════════════════════════════
   ② Student / Session information panel
═══════════════════════════════════════════════════════════════ */
.info-panel {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 20px;
}}
.info-panel-title {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  margin-bottom: 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}
.info-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px 24px;
}}
.info-field {{
  display: flex;
  flex-direction: column;
  gap: 3px;
}}
.info-field label {{
  font-size: 10.5px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--muted);
}}
.info-field .field-val {{
  font-size: 13px;
  color: var(--text);
  border: none;
  border-bottom: 1.5px solid var(--border);
  padding: 3px 2px;
  background: transparent;
  width: 100%;
  outline: none;
  font-family: inherit;
  transition: border-color .2s;
}}
.info-field .field-val:focus {{
  border-bottom-color: var(--accent);
}}
/* Show placeholder styling when empty */
.info-field .field-val:placeholder-shown {{
  color: #b0b8c8;
  font-style: italic;
}}

/* ═══════════════════════════════════════════════════════════════
   ③ Score dashboard / scorecard
═══════════════════════════════════════════════════════════════ */
.scorecard {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-bottom: 20px;
}}
.scorecard-title {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  margin-bottom: 14px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}
.sc-section-label {{
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--rwth-blue);
  margin: 12px 0 6px;
}}
.sc-row {{
  display: grid;
  grid-template-columns: 24px 1fr 120px 52px 110px;
  align-items: center;
  gap: 10px;
  padding: 5px 0;
  border-bottom: 1px solid #f0f2f5;
}}
.sc-row:last-child {{ border-bottom: none; }}
.sc-id {{
  font-weight: 800;
  font-size: 15px;
  color: var(--rwth-blue);
  text-align: center;
}}
.sc-name {{ font-size: 13px; color: var(--text); }}
.sc-bar  {{ /* bar rendered inline */ }}
.sc-score {{
  font-weight: 700;
  font-size: 13px;
  color: var(--rwth-blue);
  text-align: right;
}}

/* ═══════════════════════════════════════════════════════════════
   Shared: progress bars
═══════════════════════════════════════════════════════════════ */
.bar-wrap {{
  height: 7px;
  background: #e8eaf0;
  border-radius: 4px;
  overflow: hidden;
  width: 100%;
}}
.bar-fill {{
  height: 100%;
  background: linear-gradient(90deg, var(--rwth-blue), var(--rwth-blue-75));
  border-radius: 4px;
}}

/* ═══════════════════════════════════════════════════════════════
   Shared: rating label badges
═══════════════════════════════════════════════════════════════ */
.rating-lbl {{
  font-size: 10.5px;
  font-weight: 700;
  padding: 2px 9px;
  border-radius: 12px;
  text-transform: uppercase;
  letter-spacing: .04em;
  white-space: nowrap;
}}
.lbl-good {{ background: var(--rwth-green-25); color: var(--rwth-petrol); }}
.lbl-warn {{ background: var(--rwth-orange-25); color: var(--warn); }}
.lbl-bad  {{ background: var(--rwth-bordeaux-25); color: var(--rwth-bordeaux); }}
.lbl-uk   {{ background: #f3f4f6; color: var(--muted); }}

/* ═══════════════════════════════════════════════════════════════
   ④ Overall summary
═══════════════════════════════════════════════════════════════ */
.summary-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--rwth-blue);
  border-radius: var(--radius);
  padding: 18px 22px;
  margin-bottom: 24px;
}}
.summary-card h3 {{
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  margin-bottom: 8px;
}}
.summary-card p {{
  font-size: 13.5px;
  color: #374151;
  line-height: 1.7;
}}

/* ═══════════════════════════════════════════════════════════════
   ⑤ Section headings
═══════════════════════════════════════════════════════════════ */
.section-block {{ margin-bottom: 28px; }}
.section-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}}
.section-title {{
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--muted);
}}
.section-header::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}}

/* ═══════════════════════════════════════════════════════════════
   ⑥ Item cards
═══════════════════════════════════════════════════════════════ */
.item-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 22px;
  margin-bottom: 12px;
  page-break-inside: avoid;
}}
.card-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 12px;
}}
.card-title {{
  display: flex;
  align-items: flex-start;
  gap: 14px;
  flex: 1;
}}
.card-id {{
  font-size: 26px;
  font-weight: 900;
  color: var(--rwth-blue);
  line-height: 1;
  min-width: 28px;
}}
.card-name {{
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  line-height: 1.2;
}}
.card-cat {{
  font-size: 11px;
  color: var(--muted);
  margin-top: 2px;
}}
.card-score-block {{
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 5px;
  min-width: 130px;
}}
.card-score-block .bar-wrap {{ width: 110px; }}
.card-score-num {{
  font-size: 16px;
  font-weight: 800;
  color: var(--rwth-blue);
}}
.card-just {{
  font-size: 13px;
  color: #4b5563;
  font-style: italic;
  margin-bottom: 14px;
  padding-left: 3px;
  border-left: 3px solid var(--rwth-blue-25);
  padding-left: 10px;
  line-height: 1.6;
}}
.card-sub {{ margin-bottom: 14px; }}
.card-sub h5, .card-col h5 {{
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--muted);
  margin-bottom: 6px;
}}
.ev-list {{
  list-style: none;
  padding: 0;
}}
.ev-list li {{
  font-size: 12px;
  color: #374151;
  padding: 4px 10px;
  margin-bottom: 4px;
  background: #f8f9fc;
  border-radius: 4px;
  border-left: 3px solid var(--rwth-blue-50);
}}
.ev-list code {{
  font-family: 'SFMono-Regular', Consolas, monospace;
  font-size: 11.5px;
  color: var(--rwth-blue);
}}
.card-cols {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
  margin-top: 10px;
}}
.card-col ul {{
  list-style: disc;
  padding-left: 16px;
  font-size: 13px;
  color: #374151;
}}
.card-col li {{ margin-bottom: 4px; line-height: 1.5; }}
.empty {{ font-size: 12px; color: #c0c8d8; font-style: italic; }}

/* plain card (no heavy shadow) */
.card-plain {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 22px;
}}

/* ═══════════════════════════════════════════════════════════════
   ⑦ SPIKES table
═══════════════════════════════════════════════════════════════ */
.sp-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin-bottom: 12px;
}}
.sp-table th {{
  background: var(--rwth-blue-10);
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: var(--muted);
  padding: 8px 12px;
  text-align: left;
  border-bottom: 2px solid var(--border);
}}
.sp-table td {{ padding: 9px 12px; border-bottom: 1px solid #f0f2f5; }}
.sp-id {{ font-weight: 800; font-size: 13px; }}
.sp-icon {{ text-align: center; font-size: 16px; font-weight: 800; }}
.sp-note {{ font-size: 12px; color: #4b5563; }}
.sp-yes td {{ color: var(--rwth-petrol); }}
.sp-no  td {{ color: var(--rwth-bordeaux); }}
.sp-yes .sp-note {{ color: #374151; }}
.sp-no  .sp-note {{ color: #374151; }}
.sp-seq {{
  font-size: 13px;
  font-weight: 600;
  padding: 10px 14px;
  border-radius: 6px;
  margin-bottom: 10px;
}}
.seq-ok  {{ background: var(--rwth-green-25); color: var(--rwth-petrol); }}
.seq-bad {{ background: var(--rwth-bordeaux-25); color: var(--rwth-bordeaux); }}
.sp-overall {{
  font-size: 13px;
  color: #4b5563;
  font-style: italic;
  line-height: 1.6;
}}

/* ═══════════════════════════════════════════════════════════════
   ⑧ Assessor Notes / Observation Log
═══════════════════════════════════════════════════════════════ */
.notes-panel {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin-top: 32px;
  page-break-inside: avoid;
}}
.notes-title {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.notes-title::before {{
  content: '✎';
  font-size: 14px;
  color: var(--rwth-blue);
}}
.notes-textarea {{
  width: 100%;
  min-height: 160px;
  border: 1.5px solid var(--border);
  border-radius: 6px;
  padding: 12px 14px;
  font-family: inherit;
  font-size: 13px;
  color: var(--text);
  background: #fafbfd;
  resize: vertical;
  outline: none;
  line-height: 1.7;
  transition: border-color .2s;
}}
.notes-textarea:focus {{ border-color: var(--accent); background: #fff; }}
.notes-textarea::placeholder {{ color: #b0b8c8; font-style: italic; }}
/* Ruled lines for print */
.log-line {{
  border-bottom: 1px solid #dde3ef;
  height: 30px;
  margin-bottom: 0;
}}

/* ═══════════════════════════════════════════════════════════════
   ⑨ Footer
═══════════════════════════════════════════════════════════════ */
.report-footer {{
  margin-top: 36px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--muted);
}}

/* ═══════════════════════════════════════════════════════════════
   Responsive
═══════════════════════════════════════════════════════════════ */
@media (max-width: 680px) {{
  .lh-top        {{ flex-direction: column; gap: 16px; text-align: center; }}
  .info-grid     {{ grid-template-columns: 1fr 1fr; }}
  .sc-row        {{ grid-template-columns: 24px 1fr 80px 40px 90px; }}
  .card-cols     {{ grid-template-columns: 1fr; }}
  .card-head     {{ flex-direction: column; }}
  .card-score-block {{ align-items: flex-start; }}
  .report-footer {{ flex-direction: column; gap: 4px; }}
}}

/* ═══════════════════════════════════════════════════════════════
   ⑩ PDF / Print toolbar  (screen only)
═══════════════════════════════════════════════════════════════ */
.pdf-toolbar {{
  position: fixed;
  bottom: 28px;
  right: 28px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  z-index: 999;
}}
.pdf-btn {{
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 11px 20px;
  border: none;
  border-radius: 50px;
  font-family: inherit;
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 4px 14px rgba(0,0,0,.22);
  transition: transform .15s, box-shadow .15s;
  white-space: nowrap;
}}
.pdf-btn:hover {{
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0,0,0,.28);
}}
.pdf-btn:active {{ transform: translateY(0); }}
.pdf-btn-primary {{
  background: var(--accent);
  color: #fff;
}}
.pdf-btn-secondary {{
  background: #fff;
  color: var(--navy);
  border: 1.5px solid var(--border);
}}
.pdf-btn svg {{
  flex-shrink: 0;
}}
/* toast notification */
.pdf-toast {{
  position: fixed;
  bottom: 110px;
  right: 28px;
  background: var(--navy);
  color: #fff;
  font-size: 12.5px;
  font-weight: 600;
  padding: 10px 18px;
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(0,0,0,.25);
  opacity: 0;
  pointer-events: none;
  transition: opacity .3s;
  z-index: 1000;
}}
.pdf-toast.show {{ opacity: 1; }}

/* ═══════════════════════════════════════════════════════════════
   Print
═══════════════════════════════════════════════════════════════ */
@media print {{
  @page {{
    size: A4;
    margin: 15mm 12mm 18mm 12mm;
    /* Suppress browser-injected header/footer */
    margin-top: 10mm;
  }}
  body {{ background: #fff; font-size: 11.5px; }}
  .page {{ padding: 0; max-width: 100%; }}
  .pdf-toolbar {{ display: none !important; }}
  .letterhead {{ border-radius: 0; break-inside: avoid; }}
  .lh-top {{ padding: 14px 18px; }}
  .lh-score-num {{ font-size: 34px; }}
  .info-panel {{ break-inside: avoid; }}
  .info-field .field-val {{
    border: none;
    border-bottom: 1px solid #aaa;
    padding: 1px 2px;
  }}
  .scorecard {{ break-inside: avoid; }}
  .summary-card {{ break-inside: avoid; }}
  .item-card {{ break-inside: avoid; page-break-inside: avoid; }}
  .notes-textarea {{ display: none !important; }}
  .log-lines {{ display: block !important; }}
  .item-card, .card-plain, .notes-panel, .scorecard,
  .info-panel, .summary-card {{
    border: 1px solid #ccc;
    box-shadow: none;
  }}
  .bar-fill {{
    print-color-adjust: exact;
    -webkit-print-color-adjust: exact;
  }}
  .lbl-good, .lbl-warn, .lbl-bad, .seq-ok, .seq-bad {{
    print-color-adjust: exact;
    -webkit-print-color-adjust: exact;
  }}
  .sp-yes td, .sp-no td {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
  .seq-ok, .seq-bad {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
  a {{ color: inherit; text-decoration: none; }}
  .report-footer {{ border-top: 1px solid #ccc; }}
}}
@media screen {{
  .log-lines {{ display: none; }}
}}
</style>
</head>
<body>
<div class="page">

  <!-- ① LETTERHEAD -->
  <div class="letterhead">
    <div class="lh-top">
      <div class="lh-titles">
        <h1>LUCAS & Spike Simulation Feedback Report</h1>
        <div class="subtitle">
          Paediatric Simulation
        </div>
      </div>
      <div class="lh-score-panel">
        <div class="lh-score-label">Total Score</div>
        <div>
          <span class="lh-score-num">{total}</span>
          <span class="lh-score-denom">&thinsp;/ {max_tot}</span>
        </div>
        <div class="lh-score-pct">{pct}%</div>
      </div>
    </div>
  </div>

  <!-- ② STUDENT / SESSION INFO -->
  <div class="info-panel">
    <div class="info-panel-title">Student &amp; Session Information</div>
    <div class="info-grid">
      <div class="info-field">
        <label>Student Name</label>
        <input class="field-val" type="text" placeholder="e.g. Anna Müller" />
      </div>
      <div class="info-field">
        <label>Student ID / Matriculation No.</label>
        <input class="field-val" type="text" placeholder="e.g. 12345678" />
      </div>
      <div class="info-field">
        <label>Semester / Year</label>
        <input class="field-val" type="text" placeholder="e.g. 4th Semester" />
      </div>
      <div class="info-field">
        <label>Simulation Date</label>
        <input class="field-val" type="text" placeholder="e.g. 19.02.2026" />
      </div>
      <div class="info-field">
        <label>Location / Station</label>
        <input class="field-val" type="text" placeholder="e.g. Skills Lab Room 3" />
      </div>
      <div class="info-field">
        <label>Assessor / Supervisor</label>
        <input class="field-val" type="text" placeholder="e.g. Dr. Weber" />
      </div>
      <div class="info-field">
        <label>Scenario</label>
        <input class="field-val" type="text" placeholder="e.g. Breaking bad news — meningitis" />
      </div>
      <div class="info-field">
        <label>Attempt</label>
        <input class="field-val" type="text" placeholder="e.g. 1st attempt" />
      </div>
      <div class="info-field">
        <label>Report ID</label>
        <input class="field-val" type="text" value="{rid}" readonly style="color:#6b7280;" />
      </div>
    </div>
  </div>

  <!-- ③ SCORE DASHBOARD -->
  <div class="scorecard">
    <div class="scorecard-title">Score Dashboard &mdash; LUCAS Items A &ndash; J</div>
    {scorecard_html}
  </div>

  <!-- ④ OVERALL SUMMARY -->
  <div class="summary-card">
    <h3>Overall Assessment Summary</h3>
    <p>{summary}</p>
  </div>

  <!-- ⑤ DETAILED ITEM FEEDBACK -->
  {sections_html}

  <!-- ⑥ SPIKES ANNOTATION -->
  {spikes_html}

  <!-- ⑦ ASSESSOR NOTES / OBSERVATION LOG -->
  <div class="notes-panel">
    <div class="notes-title">Assessor Notes &amp; Observation Log</div>
    <!-- screen: editable textarea -->
    <textarea class="notes-textarea"
      placeholder="Add additional observations, behavioural notes, context, or follow-up actions here. This field is editable before printing."
    ></textarea>
    <!-- print: ruled lines -->
    <div class="log-lines">
      {log_lines}
    </div>
  </div>

  <!-- ⑧ FOOTER -->
  <div class="report-footer">
    <span>MO &copy; 420 </span>
    <span>Report ID: {rid}</span>
  </div>

</div><!-- /page -->

<!-- ⑨ PDF / PRINT TOOLBAR -->
<div class="pdf-toolbar" id="pdfToolbar">
  <button class="pdf-btn pdf-btn-primary" onclick="exportPDF()" title="Save as PDF via browser print dialog">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="12" y1="18" x2="12" y2="12"/>
      <polyline points="9 15 12 18 15 15"/>
    </svg>
    Download PDF
  </button>
  <button class="pdf-btn pdf-btn-secondary" onclick="window.print()" title="Print report">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="6 9 6 2 18 2 18 9"/>
      <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
      <rect x="6" y="14" width="12" height="8"/>
    </svg>
    Print
  </button>
</div>

<div class="pdf-toast" id="pdfToast">Preparing PDF&hellip;</div>

<script>
/* ─────────────────────────────────────────────────────────────
   Before printing, copy the live values of all editable inputs
   and the notes textarea into print-visible <span> elements so
   they appear in the PDF even though <input> values are often
   stripped by browser print engines.
───────────────────────────────────────────────────────────── */

function prepareForPrint() {{
  // 1. Inline field values into data attributes read by CSS
  document.querySelectorAll('.field-val:not([readonly])').forEach(function(inp) {{
    var val = inp.value.trim();
    if (val) {{
      inp.setAttribute('data-print-val', val);
      // Insert a sibling <span> that prints instead of the input
      var existing = inp.parentNode.querySelector('.print-val');
      if (!existing) {{
        var span = document.createElement('span');
        span.className = 'print-val';
        inp.parentNode.appendChild(span);
      }}
      inp.parentNode.querySelector('.print-val').textContent = val;
    }}
  }});

  // 2. Copy notes textarea → log area text node
  var ta = document.querySelector('.notes-textarea');
  var logLines = document.querySelector('.log-lines');
  if (ta && logLines && ta.value.trim()) {{
    var existing = document.getElementById('notesText');
    if (!existing) {{
      var p = document.createElement('p');
      p.id = 'notesText';
      p.style.cssText = 'font-size:12px;color:#374151;white-space:pre-wrap;line-height:1.8;padding-top:6px;';
      logLines.innerHTML = '';
      logLines.appendChild(p);
    }}
    document.getElementById('notesText').textContent = ta.value;
  }}
}}

function showToast(msg) {{
  var t = document.getElementById('pdfToast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(function() {{ t.classList.remove('show'); }}, 2200);
}}

function exportPDF() {{
  showToast('Opening print dialog — choose "Save as PDF"');
  prepareForPrint();
  setTimeout(function() {{ window.print(); }}, 400);
}}

// Also prepare on the browser's native beforeprint event
window.addEventListener('beforeprint', prepareForPrint);

</script>

<!-- Extra print CSS for field values captured by JS -->
<style>
@media print {{
  .print-val {{
    display: block;
    font-size: 13px;
    color: var(--text);
    padding: 2px 0;
    border-bottom: 1px solid #aaa;
    min-height: 20px;
  }}
  .field-val {{ display: none !important; }}
  .pdf-toolbar {{ display: none !important; }}
  .pdf-toast  {{ display: none !important; }}
}}
@media screen {{
  .print-val {{ display: none; }}
}}
</style>

</body>
</html>"""

    # ------------------------------------------------------------------
    # PDF rendering  (Python-side, server / pipeline use)
    # ------------------------------------------------------------------
    @staticmethod
    def _render_pdf(html_path: str, pdf_path: str) -> None:
        """
        Convert an HTML report file to PDF.

        Tries three backends in order:

        1. WeasyPrint  — best CSS fidelity, pure Python.
           Install: pip install weasyprint
           (requires system libs: libpango, libcairo — see weasyprint docs)

        2. Chromium / Google Chrome headless  — matches browser rendering
           exactly, including all JS-generated content.
           Requires chromium-browser or google-chrome on PATH.

        3. wkhtmltopdf  — lightweight fallback, no JS support.
           Install: apt install wkhtmltopdf  /  brew install wkhtmltopdf

        Raises RuntimeError if none of the backends is available.
        """
        import subprocess, shutil

        # ── 1. WeasyPrint ────────────────────────────────────────────────
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration

            font_config = FontConfiguration()
            # Inject A4 page setup that WeasyPrint honours
            page_css = CSS(
                string="""
                @page {
                    size: A4;
                    margin: 15mm 12mm 18mm 12mm;
                }
                /* Hide screen-only controls */
                .pdf-toolbar, .pdf-toast { display: none !important; }
                @media print {
                    .log-lines { display: block !important; }
                    .notes-textarea { display: none !important; }
                }
                """,
                font_config=font_config,
            )
            HTML(filename=html_path).write_pdf(
                pdf_path,
                stylesheets=[page_css],
                font_config=font_config,
            )
            return
        except ImportError:
            pass
        except Exception as exc:
            # WeasyPrint is installed but failed (missing system libs, etc.)
            raise RuntimeError(
                f"WeasyPrint failed: {exc}\n"
                "Try: pip install weasyprint  and ensure libpango/libcairo are installed."
            ) from exc

        # ── 2. Chromium / Chrome headless ────────────────────────────────
        chrome_candidates = [
            "chromium-browser", "chromium",
            "google-chrome", "google-chrome-stable",
            "chrome",
        ]
        chrome_bin = next(
            (shutil.which(c) for c in chrome_candidates if shutil.which(c)),
            None,
        )
        if chrome_bin:
            cmd = [
                chrome_bin,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                f"--print-to-pdf={pdf_path}",
                "--print-to-pdf-no-header",
                "--run-all-compositor-stages-before-draw",
                f"file://{html_path}",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return
            raise RuntimeError(
                f"Chrome headless exited {result.returncode}:\n{result.stderr}"
            )

        # ── 3. wkhtmltopdf ───────────────────────────────────────────────
        wk_bin = shutil.which("wkhtmltopdf")
        if wk_bin:
            cmd = [
                wk_bin,
                "--page-size", "A4",
                "--margin-top",    "15mm",
                "--margin-bottom", "18mm",
                "--margin-left",   "12mm",
                "--margin-right",  "12mm",
                "--encoding", "UTF-8",
                "--no-background",
                html_path,
                pdf_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return
            raise RuntimeError(
                f"wkhtmltopdf exited {result.returncode}:\n{result.stderr}"
            )

        # ── Nothing available ────────────────────────────────────────────
        raise RuntimeError(
            "No PDF backend found. Install one of:\n"
            "  pip install weasyprint          (recommended)\n"
            "  apt install chromium-browser    (or google-chrome)\n"
            "  apt install wkhtmltopdf\n"
            "Alternatively, open the HTML file in a browser and use "
            "File → Print → Save as PDF."
        )