"""
Stage 5: Standardized Report Generation
=========================================
- Renders the LLM analysis into the standardized feedback report format
- Identical structure to human instructor reports (for blinded evaluation)
- Outputs JSON (primary), HTML, and optionally PDF
- Applies generic blinded labels (REPORT_001, etc.)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from stages.base import BaseStage


class ReportGenerationStage(BaseStage):
    """Generate the final standardized feedback report."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("report")
        output_dir = Path(ctx["output_base"]) / "05_report"
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis = ctx["artifacts"]["analysis"]
        session_id = ctx["session_id"]

        # Build the standardized report
        report = self._build_report(analysis, session_id, cfg, ctx)

        # ---- JSON output (primary) ----
        json_path = output_dir / f"{report['report_id']}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"JSON report: {json_path}")

        # ---- HTML output ----
        additional = cfg.get("additional_formats", [])
        html_path = None
        if "html" in additional:
            html_path = output_dir / f"{report['report_id']}.html"
            html_content = self._render_html(report, cfg)
            with open(html_path, "w") as f:
                f.write(html_content)
            self.logger.info(f"HTML report: {html_path}")

        # ---- PDF output ----
        pdf_path = None
        if "pdf" in additional and html_path:
            try:
                pdf_path = output_dir / f"{report['report_id']}.pdf"
                self._render_pdf(str(html_path), str(pdf_path))
                self.logger.info(f"PDF report: {pdf_path}")
            except Exception as e:
                self.logger.warning(f"PDF generation failed: {e}")

        # Also copy final report to the top-level reports directory
        reports_dir = Path(ctx["config"]["paths"]["output_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        final_path = reports_dir / f"{report['report_id']}.json"
        with open(final_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        ctx["artifacts"]["report"] = report
        ctx["artifacts"]["report_path"] = str(final_path)
        if html_path:
            ctx["artifacts"]["report_html_path"] = str(html_path)
        if pdf_path:
            ctx["artifacts"]["report_pdf_path"] = str(pdf_path)

        return ctx

    # ------------------------------------------------------------------
    # Report construction
    # ------------------------------------------------------------------
    def _build_report(
        self, analysis: dict, session_id: str, cfg: dict, ctx: dict
    ) -> dict:
        """Assemble the standardized report structure."""
        label_prefix = cfg.get("label_prefix", "REPORT")
        # Generate a blinded report ID (no AI/human identifier)
        report_id = f"{label_prefix}_{session_id}"

        report = {
            "report_id": report_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": ctx["manifest"].get("pipeline_version", "unknown"),
            "manifest_digest": ctx["manifest"].get("manifest_digest", "unknown"),
            "domains": [],
            "overall_summary": analysis.get("overall_summary", ""),
        }

        for domain in analysis.get("domains", []):
            report["domains"].append({
                "name": domain.get("name", ""),
                "rating": domain.get("rating"),
                "strengths": domain.get("strengths", []),
                "gaps": domain.get("gaps", []),
                "next_steps": domain.get("next_steps", []),
            })

        # Include timing metadata if configured
        if cfg.get("include_timestamps", True):
            report["processing_times"] = ctx.get("timestamps", {})

        return report

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------
    def _render_html(self, report: dict, cfg: dict) -> str:
        """Render the report as a standalone HTML page."""
        template_path = cfg.get("template")

        if template_path and Path(template_path).exists():
            try:
                import jinja2

                with open(template_path) as f:
                    template = jinja2.Template(f.read())
                return template.render(report=report)
            except ImportError:
                self.logger.warning("Jinja2 not available, using built-in HTML template")

        # Built-in HTML template
        return self._builtin_html(report)

    @staticmethod
    def _builtin_html(report: dict) -> str:
        """Minimal built-in HTML report template."""
        domains_html = ""
        for d in report.get("domains", []):
            rating = d.get("rating", "N/A")
            name = d.get("name", "Unknown").replace("_", " ").title()

            strengths = "".join(f"<li>{s}</li>" for s in d.get("strengths", []))
            gaps = "".join(f"<li>{g}</li>" for g in d.get("gaps", []))
            next_steps = "".join(f"<li>{n}</li>" for n in d.get("next_steps", []))

            domains_html += f"""
            <div class="domain">
                <div class="domain-header">
                    <h2>{name}</h2>
                    <span class="rating">{rating} / 5</span>
                </div>
                <div class="section">
                    <h3>Strengths</h3>
                    <ul>{strengths}</ul>
                </div>
                <div class="section">
                    <h3>Areas for Improvement</h3>
                    <ul>{gaps}</ul>
                </div>
                <div class="section">
                    <h3>Next Steps</h3>
                    <ul>{next_steps}</ul>
                </div>
            </div>
            """

        return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Feedback Report — {report.get('report_id', '')}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #fafafa;
        color: #1a1a1a;
        padding: 40px;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }}
    .header {{
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }}
    .header h1 {{ font-size: 22px; font-weight: 600; }}
    .header .meta {{
        font-size: 13px;
        color: #666;
        margin-top: 6px;
    }}
    .domain {{
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }}
    .domain-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }}
    .domain-header h2 {{ font-size: 18px; }}
    .rating {{
        background: #f0f0f0;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 14px;
    }}
    .section {{ margin-bottom: 14px; }}
    .section h3 {{
        font-size: 14px;
        font-weight: 600;
        color: #444;
        margin-bottom: 6px;
    }}
    .section ul {{
        padding-left: 20px;
        font-size: 14px;
    }}
    .section li {{ margin-bottom: 4px; }}
    .summary {{
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 24px;
        font-size: 14px;
    }}
    .summary h2 {{ font-size: 18px; margin-bottom: 10px; }}
</style>
</head>
<body>
    <div class="header">
        <h1>Simulation Feedback Report</h1>
        <div class="meta">
            Report ID: {report.get('report_id', '')} &middot;
            Generated: {report.get('generated_at', '')[:19]}
        </div>
    </div>
    {domains_html}
    <div class="summary">
        <h2>Overall Summary</h2>
        <p>{report.get('overall_summary', '')}</p>
    </div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # PDF rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _render_pdf(html_path: str, pdf_path: str):
        """Convert HTML report to PDF using weasyprint."""
        try:
            from weasyprint import HTML
            HTML(filename=html_path).write_pdf(pdf_path)
        except ImportError:
            raise ImportError(
                "weasyprint not installed. Install with: pip install weasyprint"
            )
