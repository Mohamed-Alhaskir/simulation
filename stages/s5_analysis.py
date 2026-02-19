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
        transcript = ctx["artifacts"]["transcript"]
        features = ctx["artifacts"]["features"]
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

        video_features = ctx["artifacts"].get("video_features")
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
Kliniker sorgt für eine angemessene Umgebung: Privatsphäre, sitzt auf Augenhöhe, \
Augenkontakt, lädt ggf. Angehörige ein, minimiert Unterbrechungen und kommuniziert \
zeitliche Rahmenbedingungen.

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
Beobachtet Patientenemotionen, benennt sie, identifiziert den Grund, macht eine \
empathische Aussage. Setzt validierende und explorative Reaktionen ein. Lässt \
Stille zu. Übergeht emotionale Reaktionen nicht.

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
3. Kurzes Zitat oder Paraphrase aus dem Transkript als Nachweis (oder null wenn fehlend)
4. Falls fehlend oder unvollständig: Was hat gefehlt?
5. Beurteilung, ob die Schritte in korrekter Reihenfolge erfolgten (S1 → P → I → K → E → S2)

Zusätzliche Hinweise:
- Unterscheide klar zwischen den Sprechern.
- Zitiere konkrete Textstellen mit Zeitstempeln.
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

## LUCAS-Bewertungsskala

Die LUCAS besteht aus 10 Items (A-J) in drei Kategorien. Die maximale Gesamtpunktzahl \
beträgt **18 Punkte**.

---

### KATEGORIE 1: INTRODUCTIONS (Items A-B)
Bewertung: 1 (Competent) oder 0 (Unacceptable)

**A) Greeting and Introduction**
- **Competent (1):** Alle vier Elemente vorhanden: i) Begrüßt den Patienten, \
ii) nennt vollständigen Namen, iii) nennt Berufsbezeichnung/Rolle, iv) gibt kurze \
Erklärung, warum er/sie den Patienten anspricht. Beispiel: „Guten Tag, mein Name ist \
[Vor- und Nachname], ich bin Medizinstudent/in im [X]. Semester der Universität [X]. \
Der Arzt hat mich gebeten, mit Ihnen zu sprechen."
- **Unacceptable (0):** Auslassung eines oder mehrerer der Elemente i) bis iv).

**B) Identity Check**
- **Competent (1):** Beide Elemente vorhanden: i) Überprüft den vollständigen Namen des \
Patienten (bei Angehörigen/Betreuern: Name der Kontaktperson und Beziehung zum Patienten), \
ii) prüft einen weiteren Identifikator (z.B. Geburtsdatum, Adresse).
- **Unacceptable (0):** Auslassung von i) oder ii).

---

### KATEGORIE 2: GENERAL (Items C-H)
Bewertung: 2 (Competent), 1 (Borderline) oder 0 (Unacceptable)

**C) Audibility and Clarity of Speech**
- **Competent (2):** Sprache ist klar oder überwiegend klar; moduliert Stimme oder nutzt \
Wiederholung bei Bedarf; Patient hört wahrscheinlich alle Kernpunkte.
- **Borderline (1):** Sprache stellenweise etwas unklar; versucht Stimme anzupassen oder \
Wiederholungen einzusetzen, aber Patient hört möglicherweise einige Kernpunkte nicht.
- **Unacceptable (0):** Sprache ist überwiegend unklar (z.B. zu leise, zu schnell, \
undeutlich); keine Anpassung der Stimme oder Wiederholung, sodass Patient nicht folgen kann.

**D) Non-verbal Behaviour**
- **Competent (2):** Nonverbales Verhalten fördert das Engagement (ruhig, selbstsicher) \
oder überwiegend förderlich (evtl. leicht unsicher, aber ohne negativen Einfluss auf die \
Konsultation). Umfasst: Augenkontakt, Positionierung, Körperhaltung, Mimik, Gestik.
- **Borderline (1):** Nonverbales Verhalten ist unbeholfen und wahrscheinlich ablenkend \
für den Patienten; schränkt das Engagement zeitweise ein.
- **Unacceptable (0):** Nonverbales Verhalten ist unangemessen für den klinischen Kontext \
oder verstörend für den Patienten, sodass das Engagement erheblich gestört oder \
verhindert wird.

**E) Questions, Prompts and/or Explanations**
_(Hinweis: Hier wird NICHT der medizinische Inhalt der Anamnese bewertet, sondern die \
kommunikative Qualität.)_
- **Competent (2):** Fragen/Erklärungen adressieren die wesentlichen Bedürfnisse, Gefühle \
und Sorgen des Patienten; Auslassungen sind gering; Formulierungen sind überwiegend \
verständlich (Fachjargon wird sparsam und erklärt eingesetzt).
- **Borderline (1):** Versucht, Bedürfnisse und Gefühle des Patienten zu adressieren, aber \
Fragen/Erklärungen sind unvollständig oder unzureichend; wesentliche Bedürfnisse nur \
teilweise abgedeckt; Formulierungen teils schwer verständlich.
- **Unacceptable (0):** Versäumt es, für den Patienten wesentliche Themen zu explorieren \
oder zu erklären; Fragen/Erklärungen sind sehr schwer verständlich (z.B. schlechte \
Wortwahl, kein Versuch umzuformulieren), sodass wesentliche Bedürfnisse des Patienten \
nicht adressiert werden.

**F) Empathy and Responsiveness**
- **Competent (2):** Responsiv und sensibel gegenüber Bedürfnissen, Sichtweisen und \
Gefühlen des Patienten (auch wenn Verbesserungsspielraum besteht); guter oder angemessener \
Einsatz von Spiegelung/verbaler Bestätigung; erkennbare Fürsorge und Anteilnahme.
- **Borderline (1):** Versucht, auf Bedürfnisse und Gefühle des Patienten einzugehen, aber \
Reaktionen sind generell unvollständig oder oberflächlich (z.B. offensichtlich flüchtig \
oder nur an der Oberfläche); wirkt distanziert oder abgelenkt.
- **Unacceptable (0):** Kaum oder keine Anerkennung der Bedürfnisse und Gefühle des \
Patienten (z.B. ignoriert Hauptsorgen, reagiert gleichgültig auf geäußerte Sorgen); \
wirkt kalt oder desinteressiert.

**G) Clarifying and Summarising; Elicitation of Patient's Queries**
- **Competent (2):** Zeigt gutes oder angemessenes Bewusstsein und Einsatz von Klärung, \
Zusammenfassung und Rückfragen (keine wichtigen Missverständnisse bleiben ungeklärt); \
Technik evtl. leicht forciert, aber nicht so, dass das Engagement eingeschränkt wird.
- **Borderline (1):** Versucht Klärung, Zusammenfassung und Rückfragen einzusetzen, aber \
ineffektiv oder unvollständig (z.B. übersieht wichtige Missverständnisse; \
Klärungstechnik wirkt aufdringlich oder künstlich und schränkt das Engagement ein).
- **Unacceptable (0):** Zeigt sehr wenig oder kein Bewusstsein für angemessene Klärung \
und Zusammenfassung; gibt dem Patienten keine Gelegenheit, Fragen zu stellen.

**H) Consulting Style and Organisation**
- **Competent (2):** Konsultation wirkt gesprächsartig und geordnet (gelegentliche \
Ausrutscher möglich, z.B. unsichere Pausen); offene und geschlossene Fragen angemessen \
eingesetzt; gutes oder angemessenes Zeitmanagement.
- **Borderline (1):** Konsultation wirkt leicht unorganisiert; unausgewogener Einsatz von \
offenen und geschlossenen Fragen; ineffektives Zeitmanagement (z.B. muss den Abschluss \
überstürzen).
- **Unacceptable (0):** Konsultation wirkt verhörartig (z.B. Übermäßiger Einsatz \
geschlossener Fragen, wiederholte Mehrfachfragen) oder planlos und richtungslos \
(z.B. Übermäßiger Einsatz offener Fragen); sehr schlechtes Zeitmanagement \
(Konsultation endet abrupt).

---

### KATEGORIE 3: PROFESSIONAL BEHAVIOUR AND CONDUCT (Items I–J)
Bewertung: 2 (Competent) oder 0 (Unacceptable)

**I) Professional Behaviour**
- **Competent (2):** Verhalten gegenüber dem Patienten ist höflich, rücksichtsvoll, \
freundlich. Zeigt professionelles Engagement.
- **Unacceptable (0):** Verhalten ist unprofessionell: übermäßig lässig, desinteressiert, \
unhöflich oder gedankenlos (z.B. Patient hat den Eindruck, nicht ernst genommen zu werden; \
Würde des Patienten wird nicht gewahrt).

**J) Professional Spoken/Verbal Conduct**
- **Competent (2):** Verbale Kommunikation ist professionell. Aussagen vermeiden größere \
Ungenauigkeiten, sind respektvoll, bleiben im Rahmen der eigenen Kompetenz; \
Beruhigung ist angemessen.
- **Unacceptable (0):** Verbale Kommunikation ist unprofessionell. Dies kann beinhalten: \
i) größere Sachfehler, ii) abwertende, urteilende oder respektlose Bemerkungen, \
iii) Aussagen jenseits der eigenen Ausbildung/Kompetenz, iv) falsche oder verfrühte \
Beruhigung (z.B. unangemessen „Machen Sie sich keine Sorgen" oder „Alles wird gut").

---

## Diarisiertes Transkript

{transcript}

## Interaktionsmetriken

{interaction}

## Gesprächsphasen

{conversation_phases}

## SPIKES-Strukturannotation (Durchlauf 1)

Die folgende Annotation identifiziert, welche SPIKES-Schritte im Gespräch erkannt wurden \
und in welcher Reihenfolge. Nutze diese Information für die Bewertung von Item F \
(Empathie), G (Klärung/Zusammenfassung) und H (Gesprächsführung/Organisation).

{spikes_annotation}
{video_nvb_section}
## Anweisungen

Bewerte die Simulation anhand **JEDES der 10 LUCAS-Items (A-J)** gemäß der oben \
beschriebenen Skala und Deskriptoren.

Für jedes Item gib an:
1. **rating**: Ganzzahlige Bewertung gemäß der jeweiligen Skala \
(A/B: 0 oder 1; C-H: 0, 1 oder 2; I/J: 0 oder 2)
2. **justification**: Begründung der Bewertung unter direktem Bezug auf die LUCAS-Deskriptoren
3. **evidence**: 1-3 konkrete Belege aus dem Transkript (Zeitstempel und Zitat)
4. **strengths**: 1-3 spezifische Stärken in diesem Bereich
5. **gaps**: 0-2 konkrete Verbesserungsbereiche (leer lassen wenn Competent ohne Einschränkungen)
6. **next_steps**: 1-2 umsetzbare Empfehlungen für den Lernenden

Zusätzliche Hinweise:
- Nutze die **volle Bandbreite der Skala** – „Competent" bedeutet nicht automatisch \
Bestehen, „Borderline"/„Unacceptable" bedeutet nicht automatisch Durchfallen.
- Beziehe dich auf konkrete Stellen im Transkript (Zeitstempel, wörtliche Zitate).
- Unterscheide zwischen den verschiedenen Sprechern.
- Berücksichtige die Gesprächsdynamik (Pausen, Unterbrechungen, Sprechanteile).
- Items I und J haben kein Borderline – nur 0 oder 2.
- Bewerte fair und konstruktiv.
- Berechne den **Gesamtscore** (max. 18 Punkte).

Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt (kein Markdown, kein zusätzlicher Text):

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
      "evidence": ["..."],
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

    def _build_lucas_prompt(self, context: dict, spikes_annotation: dict) -> str:
        transcript_text = self._format_transcript(context["diarized_transcript"])
        verbal_summary = json.dumps(
            context["verbal_features"], indent=2, ensure_ascii=False
        )
        phases_summary = json.dumps(
            context["conversation_phases"], indent=2, ensure_ascii=False
        )
        spikes_summary = json.dumps(spikes_annotation, indent=2, ensure_ascii=False)

        # Video NVB section — inserted before the instructions block.
        # Item D is the primary consumer; F and I use it as supporting evidence.
        if context.get("video_nvb"):
            video_nvb_section = (
                "## Nonverbale Verhaltensmetriken (Videoanalyse)\n\n"
                "Nutze diese Metriken als Hauptbelege für Item D (Nonverbales Verhalten) "
                "und als ergänzende Belege für F (Empathie) und I (Professionelles "
                "Verhalten). Metriken umfassen: Blickkontaktrate, Körperhaltungsoffenheit, "
                "Lächeln-Score, Kopfbewegungen und Gestik.\n\n"
                + json.dumps(context["video_nvb"], indent=2, ensure_ascii=False)
                + "\n"
            )
        else:
            video_nvb_section = (
                "## Nonverbale Verhaltensmetriken\n\n"
                "_Videoanalyse nicht verfügbar. Item D auf Basis des Transkripts bewerten._\n"
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
    def _parse_output(self, raw: str, pass_name: str) -> dict:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        self.logger.error(f"Failed to parse {pass_name} LLM output as JSON")
        return {
            "parse_error": True,
            "pass": pass_name,
            "raw_output": raw,  # full output preserved, not truncated
        }

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