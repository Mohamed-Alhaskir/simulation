"""
Stage 5: LLM-Based Analysis & Rating
======================================
Two-pass LLM inference against LUCAS and SPIKES frameworks.

Pass 1 — SPIKES structural annotation
    Reads the transcript and identifies where each of the six SPIKES steps
    occurred, flags absent or mis-sequenced steps, and cites specific turns
    as evidence. Output: spikes_annotation.json.

Pass 2 — LUCAS scoring
    Scores all ten LUCAS items (A-J) using the transcript, verbal features,
    video NVB features, and the SPIKES annotation from Pass 1 as additional
    structured context. Output: lucas_analysis.json.

Both passes save their raw LLM output and parsed JSON independently.
assembled_context.json is the reproducibility ground truth.

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
"""

import json
import re
from pathlib import Path
from typing import Any

from stages.base import BaseStage


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
    """Two-pass LLM assessment: SPIKES annotation then LUCAS scoring."""

    def run(self, ctx: dict) -> dict:
        cfg = self._get_stage_config("llm")
        output_dir = Path(ctx["output_base"]) / "05_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # 1. Assemble and save context (audit record)
        # ----------------------------------------------------------------
        context = self._build_context(ctx)
        self.logger.info(
            f"Context assembled: "
            f"{len(context['diarized_transcript'])} transcript segments, "
            f"{context['verbal_features']['summary']['total_turns']} turns, "
            f"video_nvb={'present' if context.get('video_nvb') else 'absent'}"
        )

        context_path = output_dir / "assembled_context.json"
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Assembled context saved: {context_path}")

        # ----------------------------------------------------------------
        # 2. Pass 1 - SPIKES structural annotation
        # ----------------------------------------------------------------
        self.logger.info("Pass 1: SPIKES structural annotation")

        spikes_prompt = self._build_spikes_prompt(context)
        (output_dir / "spikes_prompt.txt").write_text(spikes_prompt, encoding="utf-8")

        spikes_raw = self._run_llm(spikes_prompt, cfg)
        (output_dir / "spikes_raw_output.txt").write_text(spikes_raw, encoding="utf-8")

        spikes_annotation = self._parse_output(spikes_raw, "spikes")
        self._validate_spikes(spikes_annotation)

        spikes_path = output_dir / "spikes_annotation.json"
        with open(spikes_path, "w", encoding="utf-8") as f:
            json.dump(spikes_annotation, f, indent=2, ensure_ascii=False)

        n_present = sum(
            1 for s in spikes_annotation.get("steps", []) if s.get("present")
        )
        self.logger.info(
            f"SPIKES annotation saved: {n_present}/{len(SPIKES_STEPS)} steps identified"
        )

        # ----------------------------------------------------------------
        # 3. Pass 2 - LUCAS scoring (informed by SPIKES annotation)
        # ----------------------------------------------------------------
        self.logger.info("Pass 2: LUCAS scoring")

        lucas_prompt = self._build_lucas_prompt(context, spikes_annotation)
        (output_dir / "lucas_prompt.txt").write_text(lucas_prompt, encoding="utf-8")

        lucas_raw = self._run_llm(lucas_prompt, cfg)
        (output_dir / "lucas_raw_output.txt").write_text(lucas_raw, encoding="utf-8")

        lucas_analysis = self._parse_output(lucas_raw, "lucas")
        self._validate_lucas(lucas_analysis)

        lucas_path = output_dir / "lucas_analysis.json"
        with open(lucas_path, "w", encoding="utf-8") as f:
            json.dump(lucas_analysis, f, indent=2, ensure_ascii=False)

        total = lucas_analysis.get("total_score", 0)
        self.logger.info(
            f"LUCAS scoring saved: total score {total}/{LUCAS_MAX_SCORE}"
        )

        # ----------------------------------------------------------------
        # 4. Combine into final analysis artifact
        # ----------------------------------------------------------------
        analysis = {
            "spikes_annotation": spikes_annotation,
            "lucas_analysis": lucas_analysis,
            "lucas_total_score": total,
            "lucas_max_score": LUCAS_MAX_SCORE,
        }

        analysis_path = output_dir / "analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        ctx["artifacts"]["assembled_context"] = context
        ctx["artifacts"]["assembled_context_path"] = str(context_path)
        ctx["artifacts"]["spikes_annotation"] = spikes_annotation
        ctx["artifacts"]["spikes_annotation_path"] = str(spikes_path)
        ctx["artifacts"]["lucas_analysis"] = lucas_analysis
        ctx["artifacts"]["lucas_analysis_path"] = str(lucas_path)
        ctx["artifacts"]["analysis"] = analysis
        ctx["artifacts"]["analysis_path"] = str(analysis_path)

        return ctx

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------
    def _build_context(self, ctx: dict) -> dict:
        def _resolve(artifact):
            if isinstance(artifact, str) and Path(artifact).exists():
                with open(artifact, encoding="utf-8") as f:
                    return json.load(f)
            return artifact

        transcript = _resolve(ctx["artifacts"]["transcript"])
        features   = _resolve(ctx["artifacts"]["features"])
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

        video_features = _resolve(ctx["artifacts"].get("video_features"))
        if video_features:
            context["video_nvb"] = video_features
        else:
            self.logger.info(
                "No video NVB features in ctx - video analysis stage "
                "was skipped or disabled."
            )
            context["video_nvb"] = None

        return context

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    # ---- SPIKES template (inlined — German, matching LUCAS template style) ----
    _SPIKES_TEMPLATE = """\
Du bist ein Experte für medizinische Ausbildung und analysierst eine pädiatrische \
Simulationsübung anhand des **SPIKES-Protokolls** (Baile et al., 2000) für die \
Übermittlung schlechter Nachrichten. Deine Aufgabe ist es, das Transkript strukturiert \
gegen die sechs SPIKES-Schritte zu annotieren.

## SPIKES-Protokoll: Sechs Schritte

**S1 – Setting up (Vorbereitung)**
Kliniker bereitet die Gesprächsumgebung **aktiv und nachweislich** vor. \
Erforderliche Belege im Transkript oder Video: explizite Sicherstellung von Privatsphäre \
(z.B. Tür schließen, Vorhang ziehen), Positionierung auf Augenhöhe des Patienten, \
Herstellen von Augenkontakt vor Gesprächsbeginn, Minimierung von Unterbrechungen \
(z.B. Handy stumm, Kollegen wegschicken), Kommunikation des Zeitrahmens.
⚠ WICHTIG: S1 gilt NUR als vorhanden, wenn mindestens eines dieser Verhaltensweisen \
im Transkript oder in zuverlässigen Videometriken (reliability ≥ moderate) positiv \
belegt ist. Die bloße Abwesenheit von Hinweisen auf eine schlechte Umgebung genügt \
NICHT. Eine Selbstvorstellung des Klinikers ist kein Beleg für S1.

**P – Patient's Perception (Patientenwahrnehmung)**
Vor der Mitteilung: offene Fragen, um zu verstehen, was der Patient bereits weiß oder \
erwartet (z.B. „Was wurde Ihnen bisher über Ihren Zustand mitgeteilt?"). Klärt \
Missverständnisse und Verleugnung.

**I – Invitation (Einladung)**
Kliniker holt die explizite oder implizite Erlaubnis des Patienten ein, Informationen \
zu teilen; erkundigt sich, wie viel Detail der Patient möchte (z.B. „Möchten Sie, dass \
ich Ihnen alle Details erkläre?").

**K – Knowledge (Wissensvermittlung)**
Ankündigung vor der schlechten Nachricht (z.B. „Leider muss ich Ihnen mitteilen…"); \
verständliche Sprache ohne Fachjargon; Information in kleinen Schritten; \
regelmäßige Verständniskontrolle; keine falsche Beruhigung.

**E – Exploring Emotions / Empathy (Empathie)**
Kliniker erkennt emotionale Äußerungen des Patienten (Angst, Trauer, Wut, Überwältigung), \
benennt sie explizit, identifiziert den Grund und macht eine empathische Aussage. \
Setzt validierende und explorative Reaktionen ein. Lässt Stille zu. Übergeht emotionale \
Reaktionen NICHT.
⚠ WICHTIG: Suche im Transkript aktiv nach allen Stellen, an denen der Patient \
Emotionen äußert (z.B. Angst vor Nadeln, Sorge um Kind, Überwältigung). Notiere für \
jede solche Stelle: a) den Zeitstempel und das Zitat, b) ob der Kliniker empathisch \
reagiert hat oder nicht. Nur wenn der Kliniker auf die Emotionen eingeht (benennt, \
validiert, exploriert) gilt E als vorhanden. Eine rein sachliche Antwort auf eine \
emotionale Äußerung gilt NICHT als E.

**S2 – Strategy and Summary (Strategie und Zusammenfassung)**
Prüft Bereitschaft des Patienten für die Besprechung nächster Schritte; präsentiert \
Behandlungsoptionen; überprüft auf Missverständnisse; lädt den Patienten ein, Fragen \
zu stellen; fasst das Gespräch zusammen.

---

## Diarisiertes Transkript

{transcript}

## Diarisierte Interaktionsmetriken

{interaction}

## Gesprächsphasen

{conversation_phases}

## Video-Analyse des non-verbalen Verhaltens

{video_nvb_section}

---

## Aufgabe

Für jeden der sechs SPIKES-Schritte:
1. Bestimme, ob er durchgeführt wurde (present: true/false)
2. Falls vorhanden: geschätzter Zeitbereich und konkrete Gesprächswendung(en) als Beleg
3. Konkretes Zitat aus dem Transkript mit Zeitstempel als Nachweis (oder null wenn fehlend)
4. Falls fehlend: Benenne die spezifischen Patientenäußerungen oder Situationen, \
   auf die der Kliniker hätte reagieren sollen (mit Zeitstempeln)
5. Beurteilung, ob die Schritte in korrekter Reihenfolge erfolgten (S1 → P → I → K → E → S2)

Kritische Regeln:
- **Beweispflicht**: present: true erfordert einen positiven Beleg im Transkript oder \
  in zuverlässigen Videometriken. Fehlen von Gegenbeweisen reicht NICHT aus.
- **S1**: Selbstvorstellung ≠ Setting up. Nur aktive Umgebungsvorbereitung zählt.
- **E**: Suche explizit nach allen emotionalen Patientenäußerungen und prüfe, ob \
  der Kliniker jeweils empathisch (nicht nur sachlich) reagiert hat. Zitiere alle \
  übersehenen emotionalen Momente im note-Feld.
- **overall_spikes_note**: Bewerte die Gesamtadhärenz realistisch. \
  Fehlende Schritte sollen klar benannt werden, ohne das Gespräch zu beschönigen.
- Unterscheide klar zwischen den Sprechern.
- Sei präzise: ein teilweise erfüllter Schritt gilt nicht als vollständig vorhanden.

Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt (kein Markdown, kein zusätzlicher Text):

{{
  "steps": [
    {{
      "id": "S1",
      "name": "Setting up",
      "present": <true|false>,
      "start_s": <Sekunden als Dezimalzahl oder null>,
      "end_s": <Sekunden als Dezimalzahl oder null>,
      "evidence": "<Zitat oder Paraphrase aus dem Transkript, oder null>",
      "note": "<Was wurde gut gemacht, was fehlte oder war unvollständig>"
    }},
    {{
      "id": "P",
      "name": "Patient's Perception",
      "present": <true|false>,
      "start_s": <float oder null>,
      "end_s": <float oder null>,
      "evidence": "<Zitat oder Paraphrase, oder null>",
      "note": "<Bewertung>"
    }},
    {{
      "id": "I",
      "name": "Invitation",
      "present": <true|false>,
      "start_s": <float oder null>,
      "end_s": <float oder null>,
      "evidence": "<Zitat oder Paraphrase, oder null>",
      "note": "<Bewertung>"
    }},
    {{
      "id": "K",
      "name": "Knowledge",
      "present": <true|false>,
      "start_s": <float oder null>,
      "end_s": <float oder null>,
      "evidence": "<Zitat oder Paraphrase, oder null>",
      "note": "<Bewertung>"
    }},
    {{
      "id": "E",
      "name": "Exploring Emotions / Empathy",
      "present": <true|false>,
      "start_s": <float oder null>,
      "end_s": <float oder null>,
      "evidence": "<Zitat oder Paraphrase, oder null>",
      "note": "<Bewertung>"
    }},
    {{
      "id": "S2",
      "name": "Strategy and Summary",
      "present": <true|false>,
      "start_s": <float oder null>,
      "end_s": <float oder null>,
      "evidence": "<Zitat oder Paraphrase, oder null>",
      "note": "<Bewertung>"
    }}
  ],
  "sequence_correct": <true|false>,
  "sequence_note": "<Beschreibung von Reihenfolgefehlern oder 'Korrekte Reihenfolge eingehalten'>",
  "overall_spikes_note": "<2-3 Sätze Gesamtbewertung der SPIKES-Adhärenz>"
}}
"""

    def _build_spikes_prompt(self, context: dict) -> str:
        transcript_text = self._format_transcript(context["diarized_transcript"])
        interaction_text = json.dumps(
            context["verbal_features"], indent=2, ensure_ascii=False
        )
        phases_text = json.dumps(
            context["conversation_phases"], indent=2, ensure_ascii=False
        )

        # Mirror the same video NVB section logic used by _build_lucas_prompt
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

        return self._SPIKES_TEMPLATE.format(
            transcript=transcript_text,
            interaction=interaction_text,
            conversation_phases=phases_text,
            video_nvb_section=video_nvb_section,
        )

    # ---- LUCAS template (inlined from analysis_prompt.j2, German) ----
    # Variables injected: {transcript}, {interaction}, {conversation_phases},
    # {spikes_annotation}, {video_nvb_section}
    # The JSON schema block uses {{ }} to escape literal braces since this
    # is rendered via str.format_map(), not Jinja2.
    _LUCAS_TEMPLATE = """\
Du bist ein Experte für medizinische Ausbildung und bewertest eine pädiatrische \
Simulationsübung anhand der **Liverpool Undergraduate Communication Assessment Scale \
(LUCAS)** (© University of Liverpool). Deine Aufgabe ist es, einen strukturierten \
Feedbackbericht auf Basis des Transkripts und der Interaktionsdaten zu erstellen.

---

## ⚠ GESPRÄCHSKONTEXT — BITTE ZUERST LESEN

Das Gespräch findet **NICHT** direkt mit dem Patienten statt.
Der Kliniker (SPEAKER_00) spricht ausschließlich mit einer **Bezugsperson des Patienten** \
(SPEAKER_01) — in der Regel einem Elternteil oder Angehörigen.
Der Patient selbst ist **abwesend** oder nicht gesprächsfähig.

Konsequenzen für die Bewertung:
- Alle LUCAS-Items, die sich auf „den Patienten" beziehen, sind auf die \
**Bezugsperson** (SPEAKER_01) zu beziehen.
- „Bedürfnisse, Gefühle und Sorgen des Patienten" in den Deskriptoren meinen die \
Bedürfnisse, Gefühle und Sorgen der **Bezugsperson**.
- Ausnahme: Item B (Identitätsprüfung) prüft, ob die Identität der Bezugsperson \
UND die Beziehung zum abwesenden Patienten geklärt wird.
- SPEAKER_00 = Kliniker (zu bewerten)
- SPEAKER_01 = Bezugsperson / Angehöriger (nicht zu bewerten)

---

## LUCAS-Bewertungsskala

Die LUCAS besteht aus 10 Items (A-J) in drei Kategorien. \
Die maximale Gesamtpunktzahl beträgt **18 Punkte**.

---

### KATEGORIE 1: INTRODUCTIONS (Items A-B)
Bewertung: 1 (Competent) oder 0 (Unacceptable)

**A) Greeting and Introduction**
- **Competent (1):** Alle vier Elemente vorhanden: \
i) Begrüßt die Bezugsperson, \
ii) nennt vollständigen eigenen Namen, \
iii) nennt Berufsbezeichnung/Rolle, \
iv) gibt eine kurze Erklärung, warum er/sie die Bezugsperson anspricht \
(z.B. Bezug zum Kind, Zweck des Gesprächs). \
Beispiel: „Guten Tag, ich bin Dr. Müller, Assistenzarzt hier auf der Station. \
Ich spreche mit Ihnen wegen Ihres Sohnes."
- **Unacceptable (0):** Auslassung eines oder mehrerer der Elemente
- **Bewertungspflicht:** Pruefe jedes der vier Elemente i)-iv) einzeln am Transkript. \\
In der justification MUSS fuer jedes Element explizit stehen ob es vorhanden \\
oder abwesend ist, bevor du das Gap benennst. \\
Format: 'i) Begrueszung: vorhanden | ii) Name: vorhanden | \\
iii) Rolle: vorhanden | iv) Zweck: fehlt -> rating 0.' \\
Falsche Attributionen (z.B. 'Rolle fehlt' obwohl Rolle klar genannt wurde) \\
sind ein kritischer Fehler und invalidieren die Bewertung. i) bis iv).
⚠ Die vier Elemente müssen NICHT im selben Satz oder Redezug vorkommen. \
Scanne die **ersten 3 Minuten** vollständig nach allen vier Elementen. \
Ein Satz wie „Wir sind ja hier wegen Ihres Kindes" zählt als Element iv), \
auch wenn er erst nach der eigentlichen Vorstellung kommt. \
Markiere ein Element erst als fehlend, wenn du das gesamte Eröffnungssegment \
geprüft hast.

**B) Identity Check**
- **Competent (1):** Beide Elemente vorhanden: \
i) Klärt die Identität der Bezugsperson (z.B. Name bestätigen oder Beziehung \
zum Patienten feststellen: „Sind Sie die Mutter von [Name]?"), \
ii) prüft einen weiteren Identifikator des Patienten \
(z.B. Name des Kindes, Geburtsdatum, Zimmernummer).
- **Unacceptable (0):** Auslassung von Element i) oder ii).
⚠ Da die Bezugsperson nicht der Patient ist, ist der Check zweigliedrig: \
Wer ist die Bezugsperson, und in welcher Beziehung steht sie zum Patienten? \
Ein vollständiger Check klärt beide Aspekte.

---

### KATEGORIE 2: GENERAL (Items C-H)
Bewertung: 2 (Competent), 1 (Borderline) oder 0 (Unacceptable)

**C) Audibility and Clarity of Speech**
- **Competent (2):** Sprache ist klar oder überwiegend klar; Fachbegriffe werden \
erklärt oder umformuliert; die Bezugsperson folgt den Kernpunkten ohne größere \
Verständnisprobleme; bei Verständnisfragen reagiert der Kliniker mit Umformulierung.
- **Borderline (1):** Sprache stellenweise unklar oder Fachbegriffe nicht immer \
erklärt; die Bezugsperson versteht möglicherweise einige Kernpunkte nicht; \
Versuche zur Klärung sind vorhanden aber unzureichend.
- **Unacceptable (0):** Sprache ist überwiegend unklar oder durchgehend fachsprachlich \
ohne Erklärung; kein Versuch zur Anpassung; die Bezugsperson kann nicht folgen.
⚠ **Stimmmodulation und Lautstärke können NICHT aus dem Transkript abgeleitet werden.** \
Bewerte für C ausschließlich: Verständlichkeit der Erklärungen (textuell belegbar), \
Reaktion auf Verständnisfragen, Einsatz von Umformulierungen und Wiederholungen \
(wenn im Transkript erkennbar). Mache KEINE Aussagen über Tonlage, Lautstärke \
oder Sprechgeschwindigkeit — diese sind ohne Audiodaten nicht beurteilbar.

**D) Non-verbal Behaviour**
- **Competent (2):** Nonverbales Verhalten fördert das Engagement (ruhig, selbstsicher) \
oder ist überwiegend förderlich (evtl. leicht unsicher, aber ohne negativen Einfluss \
auf die Konsultation). Umfasst: Augenkontakt, Positionierung, Körperhaltung, Mimik, Gestik.
- **Borderline (1):** Nonverbales Verhalten ist unbeholfen und wahrscheinlich ablenkend \
für die Bezugsperson; schränkt das Engagement zeitweise ein.
- **Unacceptable (0):** Nonverbales Verhalten ist unangemessen für den klinischen Kontext \
oder verstörend für die Bezugsperson; Engagement erheblich gestört oder verhindert.
⚠ **Bewertungsregel für D — Evidenz und Rating:**
- Verwende AUSSCHLIEßLICH die vorformatierten evidence-Strings aus dem Abschnitt \
"Vorformatierte evidence-Strings für Item D" (weiter unten in diesem Prompt). \
Transkriptzitate sind für Item D NICHT zulässig — nonverbales Verhalten \
steht nicht im Transkript und kann daraus nicht abgeleitet werden. \
Erfinde KEINE Transkriptzitate als Proxy für nonverbale Beobachtungen.
- Rating-Regel: Gesamteinschätzung NVB „Alle Hauptindikatoren positiv“ → D:2. \
Gesamteinschätzung „Mindestens ein Indikator auffällig“ → D:1 prüfen. \
Downgrade unter D:2 erfordert zusätzlich einen konkreten Hinweis im Transkript \
(z.B. Bezugsperson kommentiert Verhalten des Klinikers explizit).

**E) Questions, Prompts and/or Explanations**
_(Hinweis: Hier wird NICHT der medizinische Inhalt bewertet, sondern die \
kommunikative Qualität der Fragen und Erklärungen.)_
- **Competent (2):** Fragen und Erklärungen adressieren die wesentlichen Bedürfnisse, \
Sorgen und Verständnisfragen der Bezugsperson; Auslassungen gering; Fachbegriffe \
werden erklärt; der Kliniker reagiert auf Verständnisfragen der Bezugsperson.
- **Borderline (1):** Versucht, Bedürfnisse und Sorgen der Bezugsperson zu adressieren, \
aber Erklärungen unvollständig oder unzureichend; wesentliche Anliegen nur teilweise \
abgedeckt; Formulierungen teils schwer verständlich.
- **Unacceptable (0):** Versäumt, wesentliche Themen zu explorieren oder zu erklären; \
Erklärungen durchgehend schwer verständlich; keine Reaktion auf Verständnisfragen.
⚠ **Item E ≠ Item F.** \
E bewertet die QUALITÄT der Fragen und Erklärungen (Verständlichkeit, Vollständigkeit, \
Reaktion auf Sachfragen). \
F bewertet Empathie und emotionale Responsivität. \
Verwende **NIE** dieselbe Textstelle als Hauptbeleg für beide Items. \
Suche für E konkret nach: \
a) Stellen, an denen Fachbegriffe erklärt werden \
(z.B. „Das ist eine Blinddarmentzündung", „Das Gehirn schwimmt quasi im Wasser"), \
b) offenen Fragen zur Exploration des Wissensstands oder der Sorgen der Bezugsperson, \
c) Reaktionen auf direkte Verständnis- oder Sachfragen \
(z.B. „Was ist das?", „Kann da nichts passieren?", „Haben Sie das schon mal gemacht?").

**F) Empathy and Responsiveness**
- **Competent (2):** Responsiv und sensibel gegenüber Bedürfnissen, Sichtweisen und \
Gefühlen der Bezugsperson (auch wenn Verbesserungsspielraum besteht); guter oder \
angemessener Einsatz von Spiegelung/verbaler Bestätigung; erkennbare Fürsorge und \
Anteilnahme.
- **Borderline (1):** Versucht auf Gefühle einzugehen, aber Reaktionen sind oft \
oberflächlich oder flüchtig; kehrt nach emotionalen Momenten sofort zur sachlichen \
Erklärung zurück, ohne die Emotion der Bezugsperson zu validieren; wirkt zeitweise \
distanziert.
- **Unacceptable (0):** Kaum oder keine Anerkennung der Gefühle und Sorgen der \
Bezugsperson; ignoriert emotionale Äußerungen; wirkt kalt oder desinteressiert.
⚠ **Pflicht-Emotionsscan für Item F:** \
Identifiziere intern alle emotionalen Äußerungen der Bezugsperson \
(Angst, Schock, Sorge, Überwältigung) — nutze die SPIKES-E-Annotation \
als Ausgangspunkt. Wähle dann die **3 repräsentativsten Momente** aus: \
einen frühen, einen mittleren und den emotional intensivsten Moment. \
Prüfe für jeden dieser 3 Momente, ob der Kliniker empathisch reagiert hat \
oder sofort zur Sacherklarung zurückgekehrt ist. \
Das evidence-Feld für F enthält **maximal 3 Einträge** — \
NICHT das gesamte Transkript und KEINE vollständige Auflistung aller \
emotionalen Momente. Fasse die Gesamtbewertung in der justification zusammen, \
belege aber nur mit den 3 ausgewählten Schlüsselmomenten.

**G) Clarifying and Summarising; Elicitation of Queries**
- **Competent (2):** Zeigt gutes oder angemessenes Bewusstsein für Klärung, \
Zusammenfassung und Rückfragen; keine wichtigen Missverständnisse bleiben ungeklärt; \
die Bezugsperson erhält explizit Gelegenheit, Fragen zu stellen.
- **Borderline (1):** Versucht Klärung und Rückfragen einzusetzen, aber ineffektiv \
oder unvollständig; übersieht wichtige Missverständnisse; Technik wirkt aufdringlich \
oder künstlich.
- **Unacceptable (0):** Kaum oder kein Bewusstsein für angemessene Klärung; gibt der \
Bezugsperson keine Gelegenheit, Fragen zu stellen.

**H) Consulting Style and Organisation**
- **Competent (2):** Konsultation wirkt gesprächsartig und geordnet (gelegentliche \
unsichere Pausen möglich); offene und geschlossene Fragen angemessen eingesetzt; \
gutes oder angemessenes Zeitmanagement.
- **Borderline (1):** Konsultation wirkt leicht unorganisiert; unausgewogener Einsatz \
von Fragetypen; ineffektives Zeitmanagement.
- **Unacceptable (0):** Konsultation wirkt verhörartig oder planlos; sehr schlechtes \
Zeitmanagement; Konsultation endet abrupt.
⚠ **Evidence für H muss aus dem Transkript stammen** — konkrete Gesprächswendungen \
mit Zeitstempel. Zahlenwerte aus den Interaktionsmetriken (Sprechdauer, Wortanzahl) \
sind **kein eigenständiger Beleg** — sie können ergänzend in der justification \
erwähnt werden, ersetzen aber keine Transkriptzitate. \
Zitiere Stellen, an denen der Kliniker Übergänge schafft, das Gespräch strukturiert \
oder zur nächsten Phase wechselt.

---

### KATEGORIE 3: PROFESSIONAL BEHAVIOUR AND CONDUCT (Items I–J)
Bewertung: 2 (Competent) oder 0 (Unacceptable) — kein Borderline

**I) Professional Behaviour**
- **Competent (2):** Verhalten gegenüber der Bezugsperson ist höflich, rücksichtsvoll \
und freundlich; zeigt professionelles Engagement; die Würde und die Sorgen der \
Bezugsperson werden gewahrt und ernst genommen.
- **Unacceptable (0):** Verhalten ist unprofessionell: übermäßig lässig, desinteressiert, \
unhöflich oder gedankenlos; die Bezugsperson hat den Eindruck, nicht ernst genommen \
zu werden.

**J) Professional Spoken/Verbal Conduct**
- **Competent (2):** Verbale Kommunikation ist professionell; Aussagen vermeiden größere \
Ungenauigkeiten, sind respektvoll, bleiben im Rahmen der eigenen Kompetenz; \
Beruhigung ist angemessen und nicht verfrüht.
- **Unacceptable (0):** Verbale Kommunikation ist unprofessionell — z.B.: \
i) größere Sachfehler, \
ii) abwertende oder respektlose Bemerkungen, \
iii) Aussagen jenseits der eigenen Kompetenz, \
iv) falsche oder verfrühte Beruhigung \
(z.B. „Machen Sie sich keine Sorgen" oder „Alles wird gut").
⚠ Ehrliche Aussagen zur eigenen Erfahrung \
(z.B. „Ich habe das noch nicht selbst gemacht, aber ein erfahrener Arzt wird dabei \
sein") zählen als **kompetentes** Verhalten — sie zeigen Transparenz, keine \
Inkompetenz, und sind im Rahmen von J positiv zu werten.

---

## Diarisiertes Transkript

Sprecher-Legende:
- SPEAKER_00 = Kliniker (wird bewertet)
- SPEAKER_01 = Bezugsperson / Angehöriger des Patienten (nicht bewertet)
- UNKNOWN = nicht zuordenbarer Sprecher (ignorieren)

⚠ **Diarisierungshinweis:** Kurze Ein-Wort-Segmente wie „Genau", „Mhm", „Absolut", \
„Ja", „Okay" sind häufig falsch dem Sprecher zugeordnet (Diarisierungsfehler). \
Vertraue solchen Kurzäußerungen unter einem unerwarteten Speaker-Label nicht als \
eigenständige Aussagen. Prüfe immer den Kontext des umgebenden Dialogs.

{transcript}

## Interaktionsmetriken

{interaction}

## Gesprächsphasen

{conversation_phases}

## SPIKES-Strukturannotation (Durchlauf 1)

Die folgende Annotation identifiziert, welche SPIKES-Schritte erkannt wurden. \
Nutze sie insbesondere für: \
F (alle dort zitierten emotionalen Momente MÜSSEN berücksichtigt werden), \
G (Klärung/Zusammenfassung) und H (Gesprächsführung/Organisation).

{spikes_annotation}

{video_nvb_section}

---

## ⚠ Kritische Bewertungsregeln (haben Vorrang vor allen anderen Überlegungen)

1. **Gesprächskontext:** Der Kliniker spricht mit einer **Bezugsperson** (Angehöriger), \
nicht mit dem Patienten selbst. Alle Bewertungen beziehen sich auf die Interaktion \
mit SPEAKER_01.

2. **Beweispflicht für evidence-Felder:** Für alle Items außer D gilt: \
Jedes evidence-Element MUSS ein Transkriptzitat mit Zeitstempel enthalten \
(Format: [MM:SS] „Zitat“). Reine Metrikwerte sind kein Ersatz. \
Für **Item D gilt eine Ausnahme**: Verwende AUSSCHLIEßLICH die vorformatierten \
evidence-Strings aus dem Videoanalyse-Abschnitt. Transkriptzitate sind für D \
NICHT zulässig — nonverbales Verhalten steht nicht im Transkript.

3. **Item A — vollständiger Scan der Eröffnung:** Die vier Elemente müssen nicht \
im selben Satz stehen. Scanne die ersten 3 Minuten vollständig. \
Markiere ein Element erst als fehlend, wenn du das gesamte Eröffnungssegment \
geprüft hast.

4. **Item C — keine Audioannahmen:** Mache keine Aussagen über Stimmmodulation, \
Lautstärke oder Sprechgeschwindigkeit — diese sind aus dem Transkript nicht \
ableitbar. Bewerte nur, was textuell belegbar ist.

5. **Item D — Videometriken korrekt interpretieren:** \
Wenn gaze_on_target ≥ 75% UND Haltung offen UND kein Fidgeting → D:2. \
Downgrade auf Borderline erfordert konkrete Transkriptbelege für ablenkende \
nonverbale Signale. Halluziniere keine Verhaltensbeobachtungen aus dem Transkript.

6. **Item E ≠ Item F:** E = Qualität der Erklärungen/Fragen. \
F = Empathie/emotionale Responsivität. \
Nie dieselbe Haupttextstelle für beide Items als Primärbeleg verwenden.

7. **Item F — Pflicht-Emotionsscan:** Identifiziere ALLE emotionalen Äußerungen \
der Bezugsperson. Bewerte jede davon. Berücksichtige die Gesamtheit, \
nicht nur den ersten Moment.

8. **Item H — keine Metrikwerte als evidence:** Nur Transkriptzitate mit Zeitstempel.

9. **Item J — Transparenz ≠ Inkompetenz:** Ehrliche Aussagen zur eigenen \
Erfahrung sind kompetentes Verhalten.

10. **Volle Bandbreite der Skala nutzen:** „Competent" bedeutet nicht automatisch \
Bestehen; „Borderline"/„Unacceptable" bedeutet nicht automatisch Durchfallen.

---

## Anweisungen

Bewerte die Simulation anhand **JEDES der 10 LUCAS-Items (A-J)**.

Für jedes Item gib an:
1. **rating**: Ganzzahlige Bewertung (A/B: 0-1; C-H: 0-2; I/J: 0 oder 2)
2. **justification**: Begründung mit direktem Bezug auf LUCAS-Deskriptoren \
und Gesprächskontext (Bezugsperson, nicht Patient)
3. **evidence**: Genau 1-3 Transkriptzitate mit Zeitstempel — KEINE reinen Metrikwerte. \
Für **Item D**: ausschließlich die vorformatierten Videometrik-Strings. \
Für **Item F**: genau 3 Einträge — früher / mittlerer / intensivster emotionaler Moment. \
Kein Item darf mehr als 3 evidence-Einträge haben. \
Mehr als 3 Einträge sind ein Fehler und machen den JSON-Output ungültig.
4. **strengths**: 1-3 spezifische Stärken
5. **gaps**: 0-2 Verbesserungsbereiche (leer lassen wenn Competent ohne Einschränkungen)
6. **next_steps**: 1-2 umsetzbare Empfehlungen

Weitere Bewertungsregeln:
- **Skalenpflicht:** Nutze die volle Bandbreite der Skala aktiv. Im Zweifel zwischen \
zwei Stufen wähle die niedrigere und begründe, welches Kriterium nicht vollständig \
erfüllt ist. Eine Competent-Bewertung ist nur gerechtfertigt, wenn der Kliniker \
das entsprechende Verhalten klar demonstriert hat — nicht nur vermieden hat, es \
falsch zu machen.
- **Competent-Prüfung:** Bevor du ein Item mit rating 2 abschließt, stelle dir \
die Frage: Welches konkrete Kriterium des Competent-Deskriptors ist im Transkript \
nachweisbar erfüllt? Wenn die Antwort vage ist oder nur auf Abwesenheit negativer \
Signale beruht, wähle Borderline (1).
- **Eindeutigkeitspflicht evidence:** Kein Transkriptzitat darf in mehr als zwei \
Items als Primärbeleg verwendet werden. Items E, G, H, I und J erfordern jeweils \
eigene, itemspezifische Belege aus unterschiedlichen Gesprächsmomenten. \
Erklärungs- und Wissenslieferungssequenzen der K-Phase (z.B. „Infektparameter \
steigen“, „Nervenwasser“, „Ultraschall am Bauch“) sind KEIN valider Beleg \
für E, G, H, I oder J. \
Für **Item E** sind ausschließlich folgende Belegtypen zulässig: \
a) offene Fragen zur Exploration des Wissensstands der Bezugsperson \
(z.B. „Haben Sie schon mal gehört, was wir vorhaben?“), \
b) Reaktionen auf direkte Verständnis- oder Sachfragen der Bezugsperson \
(z.B. nach „Was ist das?“ oder „Kann da nichts passieren?“), \
c) Umformulierungen bei Missverständnissen (z.B. „Entschuldigung, habe ich \
ein bisschen falsch formuliert“). \
Für **Item G** sind ausschließlich Kläärungs-, Zusammenfassungs- und \
Rückfragemomente zulässig — NICHT dieselben Stellen wie für E oder H.
- **Item B — strenge Auslegung:** Passive Bestätigungen wie „Das ist sie“ sind \
KEIN Identitätscheck. Competent erfordert, dass der Kliniker aktiv i) den Namen \
der Bezugsperson ODER ihre Beziehung zum Patienten erfragt/bestätigt UND \
ii) einen zweiten Identifikator des Patienten prüft (Name des Kindes, Geburtsdatum, \
Zimmernummer). Fehlt einer dieser Schritte → rating 0.
- **Item H — Sprechanteile:** Wenn SPEAKER_00 mehr als 75% der Gesprächszeit \
spricht, muss die Konsultation als Borderline-Kandidat (H:1) behandelt werden. \
H:2 ist nur zulässig, wenn in der justification explizit und \
transkriptgestützt begründet wird, warum die Gesprächsdominanz das Engagement \
der Bezugsperson NICHT eingeschränkt hat \
(z.B. weil die Bezugsperson aktiv und häufig eigene Fragen eingebracht hat). \
Prüfe die Interaktionsmetriken auf den Sprechangteil von SPEAKER_00 und nenne \
den Wert explizit in der justification.
- Items I und J haben kein Borderline — nur 0 oder 2.
- **Item I — evidence-Pflicht:** Alle evidence-Strings für I müssen \\
ausschließlich aus SPEAKER_00-Turns stammen und ein konkretes professionelles \\
Verhalten belegen: z.B. spontane Entschuldigung, Würdigung der Sorgen, \\
explizites Angebot von Unterstützung, respektvoller Umgang bei Ablehnung. \\
Unzulässig als I-Belege: (a) Äußerungen der Bezugsperson (SPEAKER_01); \\
(b) Metakommentare am Simulationsende (z.B. 'Oh Gott, ich bin so nervös' — \\
das ist SPEAKER_01, nicht SPEAKER_00); \\
(c) Backchannel-Signale ('Ja', 'Genau', 'Okay') ohne inhaltliche Substanz. \\
Prüfe jeden evidence-String: Gehört er zu SPEAKER_00? Zeigt er konkretes \\
professionelles Verhalten? Wenn nein → streichen und durch valide \\
SPEAKER_00-Belege ersetzen.
Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt (kein Markdown, kein Text davor/danach):

{{
  "lucas_items": [
    {{
      "item": "A",
      "name": "Greeting and Introduction",
      "category": "Introductions",
      "max_score": 1,
      "rating": "<0 oder 1>",
      "rating_label": "<Competent|Unacceptable>",
      "justification": "<Begründung mit Bezug auf LUCAS-Deskriptoren>",
      "evidence": ["<[MM:SS] 'Zitat aus Transkript'>"],
      "strengths": ["<Stärke>"],
      "gaps": ["<Verbesserungsbereich>"],
      "next_steps": ["<Empfehlung>"]
    }},
    {{
      "item": "B",
      "name": "Identity Check",
      "category": "Introductions",
      "max_score": 1,
      "rating": "<0 oder 1>",
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "C",
      "name": "Audibility and Clarity of Speech",
      "category": "General",
      "max_score": 2,
      "rating": "<0, 1 oder 2>",
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "D",
      "name": "Non-verbal Behaviour",
      "category": "General",
      "max_score": 2,
      "rating": "<0, 1 oder 2>",
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "E",
      "name": "Questions, Prompts and/or Explanations",
      "category": "General",
      "max_score": 2,
      "rating": "<0, 1 oder 2>",
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "F",
      "name": "Empathy and Responsiveness",
      "category": "General",
      "max_score": 2,
      "rating": "<0, 1 oder 2>",
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": [
        "<[MM:SS] Früher emotionaler Moment der Bezugsperson + Reaktion des Klinikers>",
        "<[MM:SS] Mittlerer emotionaler Moment der Bezugsperson + Reaktion des Klinikers>",
        "<[MM:SS] Intensivster emotionaler Moment der Bezugsperson + Reaktion des Klinikers>"
      ],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "G",
      "name": "Clarifying and Summarising",
      "category": "General",
      "max_score": 2,
      "rating": "<0, 1 oder 2>",
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "H",
      "name": "Consulting Style and Organisation",
      "category": "General",
      "max_score": 2,
      "rating": "<0, 1 oder 2>",
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "I",
      "name": "Professional Behaviour",
      "category": "Professional Behaviour and Conduct",
      "max_score": 2,
      "rating": "<0 oder 2>",
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }},
    {{
      "item": "J",
      "name": "Professional Spoken/Verbal Conduct",
      "category": "Professional Behaviour and Conduct",
      "max_score": 2,
      "rating": "<0 oder 2>",
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": ["..."],
      "gaps": ["..."],
      "next_steps": ["..."]
    }}
  ],
  "total_score": "<Summe aller Ratings (max. 18)>",
  "overall_summary": "<3-5 Sätze Gesamtbewertung: Hauptstärken, wichtigste Entwicklungsbereiche, Gesamteindruck>"
}}
"""

    # ------------------------------------------------------------------
    # Video summariser — converts raw metric JSON into LLM-readable prose
    # so the model doesn't misread distributions and hallucinate verdicts.
    # Called by _build_lucas_prompt before injecting into the template.
    # ------------------------------------------------------------------
    @staticmethod
    def _summarise_video_for_llm(video_features: dict) -> str:
        """
        Pre-interpret raw video metric dicts into concise German prose.
        Returns a formatted string ready to be injected into the prompt.
        This prevents the LLM from misreading raw distributions and avoids
        hallucinated nonverbal behaviour observations.
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

        # ── D1: Eye contact ──────────────────────────────────────────────
        d1 = video_features.get("D1_eye_contact", {})
        gaze = d1.get("gaze_on_target", {})
        gaze_rate = gaze.get("rate")
        gaze_rel = d1.get("reliability", "unbekannt")
        if gaze_rate is not None:
            pct = round(gaze_rate * 100)
            if gaze_rate >= 0.75:
                level = "gut"
            elif gaze_rate >= 0.50:
                level = "moderat"
            else:
                level = "niedrig"
            lines.append(
                f"**Augenkontakt (D1):** {pct}% der detektierten Frames auf "
                f"Gesprächspartner gerichtet → {level} "
                f"(Zuverlässigkeit der Messung: {gaze_rel})"
            )

        # ── D2: Positioning ──────────────────────────────────────────────
        d2 = video_features.get("D2_positioning", {})
        h2p = d2.get("Height_to_patient", {})
        h2p_mean = h2p.get("mean")
        d2_rel = d2.get("reliability", "unbekannt")
        if h2p_mean is not None:
            # Normalised eye-level Y: lower value = higher in frame = standing over patient
            # Values near 0.25-0.35 generally suggest seated/same level
            if h2p_mean <= 0.35:
                pos_interp = "ungefähr auf Augenhöhe der Bezugsperson (günstige Positionierung)"
            else:
                pos_interp = "tendenziell höher als die Bezugsperson positioniert"
            lines.append(
                f"**Positionierung (D2):** Augenhöhe normalisiert = {round(h2p_mean, 3)} "
                f"→ {pos_interp} (Zuverlässigkeit: {d2_rel})"
            )

        # ── D3: Posture ──────────────────────────────────────────────────
        d3 = video_features.get("D3_posture", {})
        arm_dev = d3.get("baseline_arm_deviation", {}).get("mean")
        d3_rel = d3.get("reliability", "unbekannt")
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

        # ── D4: Facial expressions ───────────────────────────────────────
        d4 = video_features.get("D4_facial_expressions", {})
        pos_expr = d4.get("positive_expression_rate", {})
        expr_rate = pos_expr.get("rate")
        d4_rel = d4.get("reliability", "unbekannt")
        if expr_rate is not None:
            pct_e = round(expr_rate * 100)
            if expr_rate >= 0.15:
                expr_level = "erkennbar positiv/freundlich"
            elif expr_rate >= 0.05:
                expr_level = "überwiegend neutral"
            else:
                expr_level = "kaum positive Mimik"
            lines.append(
                f"**Mimik (D4):** positive Gesichtsausdruck-Rate = {pct_e}% "
                f"→ {expr_level} (Zuverlässigkeit: {d4_rel})"
            )

        # ── D5: Gestures / fidgeting ─────────────────────────────────────
        d5 = video_features.get("D5_gestures_and_mannerisms", {})
        fidget = d5.get("hand_movement_periodicity", {})
        is_repetitive = fidget.get("is_repetitive", False)
        fidget_strength = fidget.get("periodicity_strength")
        d5_rel = d5.get("reliability", "unbekannt")
        fidget_str = (
            f"ja (Stärke: {round(fidget_strength, 2)})" if is_repetitive
            else "nein"
        )
        lines.append(
            f"**Wiederholende Handbewegungen / Fidgeting (D5):** {fidget_str} "
            f"(Zuverlässigkeit: {d5_rel})"
        )

        # ── Overall D summary ────────────────────────────────────────────
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


        # ── Phase-level fidgeting / posture flags ──────────────────────────
        phase_summaries = video_features.get("phase_summaries", [])
        phase_flags = []
        for ph in phase_summaries:
            ph_name = ph.get("phase", "?")
            ph_d5 = ph.get("D5_gestures_and_mannerisms", {})
            ph_hmp = ph_d5.get("hand_movement_periodicity", {})
            ph_pitch = ph_d5.get("head_movement", {}).get("pitch_periodicity", {})
            ph_rel = ph_d5.get("reliability", "unbekannt")
            ph_hand_rep = ph_hmp.get("is_repetitive", False)
            ph_hand_str = ph_hmp.get("periodicity_strength", 0)
            ph_pitch_rep = ph_pitch.get("is_repetitive", False)
            ph_pitch_str = ph_pitch.get("periodicity_strength", 0)
            ph_d3 = ph.get("D3_posture", {})
            ph_arm_dev = ph_d3.get("baseline_arm_deviation", {}).get("mean")
            phase_issues = []
            if ph_hand_rep:
                phase_issues.append(
                    f"Wiederholende Handbewegungen (St\u00e4rke: {round(ph_hand_str, 2)})"
                )
            if ph_pitch_rep:
                phase_issues.append(
                    f"Wiederholende Kopfbewegungen/Pitch (St\u00e4rke: {round(ph_pitch_str, 2)})"
                )
            if ph_arm_dev is not None and ph_arm_dev < -0.5:
                phase_issues.append(
                    f"Geschlossene K\u00f6rperhaltung "
                    f"(Armabweichung: {round(ph_arm_dev, 2)} SD)"
                )
            if phase_issues:
                phase_flags.append(
                    f"  \u26a0 Phase \u2018{ph_name}\u2019: "
                    + ", ".join(phase_issues)
                    + f" (Zuverl\u00e4ssigkeit: {ph_rel})"
                )
        if phase_flags:
            lines.append("")
            lines.append(
                "**Phasenspezifische NVB-Auff\u00e4lligkeiten** "
                "(auch wenn Globalwert unauff\u00e4llig ist):"
            )
            lines.extend(phase_flags)
            lines.append(
                "  \u2192 Phasenspezifische Signale in D-Bewertung ber\u00fccksichtigen; "
                "bei Dauer < 60s oder St\u00e4rke < 0.4 als Grenzfall behandeln."
            )

        # ── Pre-formatted evidence strings for Item D ───────────────────
        # The LLM copies these directly into the evidence array for Item D.
        # This prevents fabricated transcript quotes being used as D evidence.
        lines.append("")
        lines.append("## Vorformatierte evidence-Strings fur Item D")
        lines.append(
            "Kopiere die folgenden Strings DIREKT in das evidence-Feld von Item D. "
            "Ersetze sie NICHT durch Transkriptzitate. "
            "Transkriptzitate sind fur Item D UNGULTIG, da nonverbales Verhalten "
            "nicht im Transkript steht."
        )
        ev_lines = []
        if gaze_rate is not None:
            ev_lines.append(
                f'- "Augenkontakt (D1): {pct}% der Frames auf Gesprachspartner '
                f'gerichtet -> {level} (Zuverlassigkeit: {gaze_rel})"'
            )
        if arm_dev is not None:
            ev_lines.append(
                f'- "Korperhaltung (D3): {posture} '
                f'(Abweichung vom Ruhewert: {round(arm_dev, 2)} SD, '
                f'Zuverlassigkeit: {d3_rel})"'
            )
        ev_lines.append(
            f'- "Wiederholende Handbewegungen / Fidgeting (D5): {fidget_str} '
            f'(Zuverlassigkeit: {d5_rel})"'
        )
        if expr_rate is not None:
            ev_lines.append(
                f'- "Mimik (D4): positive Ausdrucksrate {pct_e}% '
                f'-> {expr_level} (Zuverlassigkeit: {d4_rel})"'
            )
        lines.extend(ev_lines)

        return "\n".join(lines) + "\n"

    def _build_lucas_prompt(self, context: dict, spikes_annotation: dict) -> str:
        transcript_text = self._format_transcript(context["diarized_transcript"])
        verbal_summary = json.dumps(
            context["verbal_features"], indent=2, ensure_ascii=False
        )
        phases_summary = json.dumps(
            context["conversation_phases"], indent=2, ensure_ascii=False
        )
        spikes_summary = json.dumps(spikes_annotation, indent=2, ensure_ascii=False)

        # Video NVB section — pre-interpreted prose replaces raw JSON dump.
        # _summarise_video_for_llm converts distributions into readable verdicts,
        # preventing the LLM from hallucinating nonverbal observations.
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

        return self._LUCAS_TEMPLATE.format(
            transcript=transcript_text,
            interaction=verbal_summary,
            conversation_phases=phases_summary,
            spikes_annotation=spikes_summary,
            video_nvb_section=video_nvb_section,
        )

    @staticmethod
    def _format_transcript(segments: list[dict]) -> str:
        return "\n".join(
            f"[{s['speaker']}] ({s['start']:.1f}-{s['end']:.1f}s): {s['text']}"
            for s in segments
        )

    # ------------------------------------------------------------------
    # LLM backends (model loaded once, reused for both passes)
    # ------------------------------------------------------------------
    def _run_llm(self, prompt: str, cfg: dict) -> str:
        backend = cfg.get("backend", "llama_cpp")
        if backend == "llama_cpp":
            return self._run_llama_cpp(prompt, cfg)
        elif backend == "vllm":
            return self._run_vllm(prompt, cfg)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")

    def _run_llama_cpp(self, prompt: str, cfg: dict) -> str:
        try:
            from llama_cpp import Llama
        except ImportError:
            self.logger.error(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
            raise

        if not hasattr(self, "_llama_model"):
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
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            stop=None,
        )
        return response["choices"][0]["text"]

    def _run_vllm(self, prompt: str, cfg: dict) -> str:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            self.logger.error("vLLM not installed. Install with: pip install vllm")
            raise

        if not hasattr(self, "_vllm_model"):
            model_name = cfg.get("model_name", cfg.get("model_path"))
            self.logger.info(f"Loading vLLM model: {model_name}")
            self._vllm_model = LLM(
                model=model_name,
                max_model_len=cfg.get("context_length", 8192),
                seed=cfg.get("seed", 42),
            )

        params = SamplingParams(
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            max_tokens=cfg.get("max_tokens", 4096),
        )
        outputs = self._vllm_model.generate([prompt], params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Parsing (shared by both passes)
    # ------------------------------------------------------------------
    @staticmethod
    def _cap_evidence(result: dict) -> dict:
        """Truncate evidence arrays to max 3 entries per item."""
        for key in ("lucas_items", "items"):
            items = result.get(key, [])
            for item in items:
                ev = item.get("evidence")
                if isinstance(ev, list) and len(ev) > 3:
                    item["evidence"] = ev[:3]
                    item.setdefault("_evidence_truncated", True)
        return result

    def _parse_output(self, raw: str, pass_name: str) -> dict:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # If model output two JSON objects concatenated, take only the first
        text = self._take_first_json_object(text)

        try:
            return self._cap_evidence(json.loads(text))
        except json.JSONDecodeError:
            pass

        # Try extracting the largest valid JSON object from the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return self._cap_evidence(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

        # Try salvaging: strip corrupt items, close open structures
        try:
            salvaged = self._salvage_corrupt_json(text, pass_name)
            if salvaged:
                return self._cap_evidence(salvaged)
        except Exception:
            pass

        self.logger.error(f"Failed to parse {pass_name} LLM output as JSON")
        return {
            "parse_error": True,
            "pass": pass_name,
            "raw_output": raw,  # full output preserved, not truncated
        }

    @staticmethod
    def _take_first_json_object(text: str) -> str:
        """If text contains two JSON objects concatenated, return only the first."""
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
                    return text[:i + 1]
        return text

    def _salvage_corrupt_json(self, text: str, pass_name: str) -> dict | None:
        """
        Three-strategy salvage for broken LLM JSON output.
        Operates on the already-extracted first JSON object (via _take_first_json_object).

        Strategy 1 — Corrupt item: an item block contains invalid syntax
          (e.g. bare `0,` before a key). Split the items array on object
          boundaries, parse each individually, discard corrupt ones.

        Strategy 2 — Truncation: the object cuts off before closing brackets.
          Find the last complete item (ends with `},`) and close the structure.

        Strategy 3 — last resort: try to extract any valid item dicts from the
          text using a broad pattern, regardless of overall structure.
        """
        import re as _re

        def _extract_array_content(txt: str, key: str) -> str | None:
            """Extract array content using bracket counting to handle nested []."""
            start = txt.find(f'"{key}"')
            if start == -1:
                return None
            bracket_start = txt.find('[', start)
            if bracket_start == -1:
                return None
            depth = 0
            in_str = False
            esc = False
            for i in range(bracket_start, len(txt)):
                ch = txt[i]
                if esc: esc = False; continue
                if ch == '\\' and in_str: esc = True; continue
                if ch == '"' and not esc: in_str = not in_str; continue
                if in_str: continue
                if ch == '[': depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return txt[bracket_start + 1:i]
            return None

        def _extract_good_items(arr_text: str) -> list:
            good = []
            parts = _re.split(r'(?<=\}),\s*(?=\{)', arr_text.strip())
            for part in parts:
                part = part.strip().rstrip(",")
                try:
                    obj = json.loads(part)
                    if isinstance(obj, dict) and ("item" in obj or "id" in obj):
                        good.append(obj)
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"{pass_name}: Dropped corrupt item block: {part[:80]!r}"
                    )
            return good

        # Strategy 1: extract items array using bracket-depth counting
        # (regex non-greedy fails because evidence:[] contains ] chars)
        arr_content = (_extract_array_content(text, "lucas_items")
                       or _extract_array_content(text, "items"))
        if arr_content:
            good_items = _extract_good_items(arr_content)
            if good_items:
                self.logger.warning(
                    f"{pass_name}: Salvaged {len(good_items)}/10 items "
                    f"(Strategy 1 — corrupt item removal)"
                )
                return {
                    "lucas_items": good_items,
                    "total_score": sum(
                        i.get("rating", i.get("score", 0)) for i in good_items
                    ),
                    "overall_summary": (
                        "[Salvaged output — one or more corrupt items removed]"
                    ),
                    "_salvaged": True,
                }

        # Strategy 2: truncated output — find last complete item and close
        candidates = [m.end() for m in _re.finditer(r'\}\s*,\s*\n', text)]
        if candidates:
            cut = candidates[-1]
            truncated = text[:cut].rstrip().rstrip(",")
            closed = truncated + "\n  ]\n}"
            try:
                result = json.loads(closed)
                n = len(result.get("lucas_items", result.get("items", [])))
                self.logger.warning(
                    f"{pass_name}: Salvaged {n} items (Strategy 2 — truncation)"
                )
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 3: extract any valid item-shaped dicts from anywhere in text
        item_matches = _re.findall(
            r'\{\s*"item"\s*:\s*"[A-J]"[\s\S]{20,500}?\}', text
        )
        good_items = []
        for raw in item_matches:
            try:
                obj = json.loads(raw)
                if "item" in obj and "rating" in obj:
                    good_items.append(obj)
            except json.JSONDecodeError:
                pass
        if good_items:
            self.logger.warning(
                f"{pass_name}: Salvaged {len(good_items)}/10 items "
                f"(Strategy 3 — pattern extraction)"
            )
            return {
                "lucas_items": good_items,
                "total_score": sum(i.get("rating", 0) for i in good_items),
                "overall_summary": "[Salvaged output — extracted from corrupt JSON]",
                "_salvaged": True,
            }

        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_spikes(self, annotation: dict) -> None:
        if annotation.get("parse_error"):
            self.logger.error(
                "SPIKES annotation has parse errors - LUCAS Pass 2 will "
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
            self.logger.info(f"SPIKES: steps not identified in recording: {absent}")

        if not annotation.get("sequence_correct"):
            self.logger.warning(
                f"SPIKES sequencing issue: {annotation.get('sequence_note', '')}"
            )

    def _validate_lucas(self, analysis: dict) -> None:
        if analysis.get("parse_error"):
            self.logger.error(
                "LUCAS analysis has parse errors - report may be incomplete."
            )
            return

        # The German template outputs "lucas_items" with key "item" and "rating".
        # Support both schemas so the fallback prompt (which uses "items"/"id"/"score")
        # also validates correctly.
        raw_items = analysis.get("lucas_items") or analysis.get("items", [])
        if not raw_items:
            self.logger.warning("LUCAS: no items found in output (tried 'lucas_items' and 'items')")
            return

        # Normalise to a common internal shape: {item_id: out_dict}
        # Primary schema: {"item": "A", "rating": 1, ...}
        # Fallback schema: {"id": "A", "score": 1, ...}
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
            # Primary schema uses "rating"; fallback uses "score"
            score_raw = out.get("rating") if out.get("rating") is not None else out.get("score")
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
                    f"LUCAS item '{item_id}': score {score} not in allowed {allowed}"
                )
                valid = False
            else:
                total_computed += score

            if not out.get("evidence"):
                self.logger.warning(f"LUCAS item '{item_id}': no evidence provided")

        # Correct total_score if the model miscalculated (value may be a string)
        claimed_raw = analysis.get("total_score")
        try:
            claimed = int(claimed_raw)
        except (TypeError, ValueError):
            claimed = None

        if claimed is not None and claimed != total_computed:
            self.logger.warning(
                f"LUCAS: claimed total {claimed} != computed {total_computed}. "
                "Overwriting with computed value."
            )
        analysis["total_score"] = total_computed

        if valid:
            self.logger.info(
                f"LUCAS validation passed: {total_computed}/{LUCAS_MAX_SCORE}"
            )