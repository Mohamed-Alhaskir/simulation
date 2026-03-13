"""
lucas_multipass.py

Drop-in replacement for the LUCAS single-pass inference in llm_analysis_stage.py.

Architecture:
  Pass 1 (A, B)    — Introductions only, first 3 min transcript
  Pass 2 (C, E)    — Clarity + Explorations scan, full transcript
  Pass 3 (D)       — Video metrics only, no transcript
  Pass 4 (F, G)    — Emotion scan + Summarising, full transcript
  Pass 5 (H)       — Style/Organisation, full transcript + metrics
  Pass 6 (I, J)    — Professional conduct, full transcript
  Pass 7           — Aggregation: total_score, summary, evidence dedup check

Each pass runs independently. All passes can be parallelised except Pass 7.

DESIGN PRINCIPLES (v2.1 — generalised, strictness-aware):
  - Prompts follow the official LUCAS rubric criteria faithfully.
  - No hardcoded counting thresholds — the LLM applies rubric descriptors
    holistically.
  - No hardcoded German phrase lists — validators use structural checks only.
  - Scenario-specific context (caregiver conversation, speaker roles) is
    injected once via SYSTEM_PREAMBLE.
  - Strictness is integrated INTO per-item criteria at decision boundaries,
    not as a disconnected preamble. It affects:
      * Binary items (A, B, I, J): what counts as acceptable evidence
      * 3-point items (C, D, E, F, G, H): where the 1-vs-2 boundary sits
  - Strictness does NOT affect output length.
"""

import json
import logging
import re
import concurrent.futures
from typing import Any

from utils.json_utils import repair_unescaped_quotes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-rule validators
# ---------------------------------------------------------------------------

def _validate_item(item: dict, video_features: dict, register_break: bool) -> dict:
    """
    Applies hard rules to a single scored item.
    Returns the (possibly corrected) item with a 'validation_flags' field.
    """
    flags = []
    label = item.get("item")
    rating = item.get("rating")

    if label == "D":
        gaze = (video_features.get("D1_eye_contact") or {}).get("gaze_on_target", {}).get("rate")
        overall_rel = (video_features.get(
            "I_professional_behaviour_demeanour"
        ) or {}).get("overall_reliability", "low")

        if gaze is not None and gaze < 0.75 and rating == 2:
            flags.append(f"D1 VIOLATION: gaze={gaze:.1%} < 75% but rating=2 -> forced to 1")
            item["rating"] = 1
            item["rating_label"] = "Borderline"
            item["justification"] += (
                f" [VALIDATOR: gaze={gaze:.1%} < 75% -> D:2 blocked, corrected to D:1]"
            )

        if overall_rel == "low" and rating == 2:
            flags.append("D RELIABILITY VIOLATION: reliability=low but rating=2 -> forced to 1")
            item["rating"] = 1
            item["rating_label"] = "Borderline"
            item["justification"] += " [VALIDATOR: reliability=low -> D:2 blocked, corrected to D:1]"

    elif label in ("I", "J"):
        if rating == 1:
            flags.append(f"{label} BORDERLINE VIOLATION: rating=1 not allowed -> forced to 0")
            item["rating"] = 0
            item["rating_label"] = "Unacceptable"
            item["justification"] += f" [VALIDATOR: {label} has no borderline -> 0 or 2]"

    if flags:
        item["validation_flags"] = flags
        logger.warning(f"Item {label} validation: {flags}")

    return item


def _detect_register_break(diarized_transcript: list) -> tuple[bool, list]:
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


def _get_opening_transcript(diarized_transcript: list, max_seconds: float = 180.0) -> str:
    lines = []
    for seg in diarized_transcript:
        if seg.get("start", 0) > max_seconds:
            break
        ts = seg.get("start", 0)
        mm = int(ts) // 60
        ss = int(ts) % 60
        lines.append(f"[{mm:02d}:{ss:02d}] [{seg['speaker']}] {seg['text']}")
    return "\n".join(lines)


def _format_full_transcript(diarized_transcript: list) -> str:
    lines = []
    for seg in diarized_transcript:
        ts = seg.get("start", 0)
        mm = int(ts) // 60
        ss = int(ts) % 60
        lines.append(f"[{mm:02d}:{ss:02d}] [{seg['speaker']}] {seg['text']}")
    return "\n".join(lines)


def _call_llm(backend, prompt: str, cfg: dict) -> dict | None:
    try:
        raw = backend.generate(prompt, cfg)
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(clean[start:end + 1])
            except json.JSONDecodeError:
                pass
            try:
                repaired = repair_unescaped_quotes(clean[start:end + 1])
                return json.loads(repaired)
            except (json.JSONDecodeError, Exception):
                pass
        logger.error(f"LLM call: JSON parse failed. Raw output (first 500 chars): {raw[:500]!r}")
        return None
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Strictness helpers
# ---------------------------------------------------------------------------

def _strictness_note_binary(level: int) -> str:
    """
    Strictness calibration for binary items (A, B, I, J).
    Affects what counts as acceptable evidence, not the rubric structure.
    """
    if level == 1:
        return (
            "\nKALIBRIERUNG (nachsichtig): Implizite oder teilweise Belege "
            "werden akzeptiert. Wenn ein Element sinngemäss erkennbar ist, auch "
            "ohne woertlich-explizite Formulierung, zaehlt es als vorhanden.\n"
        )
    if level == 3:
        return (
            "\nKALIBRIERUNG (streng): Nur explizite, woertlich belegbare "
            "Aussagen zaehlen. Wenn ein Element nur implizit oder sinngemäss "
            "erkennbar ist, zaehlt es als fehlend.\n"
        )
    return ""  # level 2 = standard, no note


def _strictness_note_graded(level: int) -> str:
    """
    Strictness calibration for 3-point items (C, D, E, F, G, H).
    Affects the 1-vs-2 boundary.
    """
    if level == 1:
        return (
            "\nKALIBRIERUNG (nachsichtig): Bei Grenzfaellen zwischen zwei "
            "Stufen waehle die hoehere. Teilweise oder implizite Belege reichen "
            "fuer die hoehere Stufe.\n"
        )
    if level == 3:
        return (
            "\nKALIBRIERUNG (streng): Bei Grenzfaellen zwischen zwei Stufen "
            "waehle die niedrigere. Nur klare, eindeutige Belege rechtfertigen "
            "die hoehere Stufe.\n"
        )
    return ""  # level 2 = standard, no note


# ---------------------------------------------------------------------------
# Per-pass prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PREAMBLE = """Du bist ein Experte fuer medizinische Ausbildung und bewertest eine klinische Simulationsuebung anhand der Liverpool Undergraduate Communication Assessment Scale (LUCAS).

GESPRAECHSKONTEXT:
- SPEAKER_00 = Kliniker (wird bewertet)
- SPEAKER_01 = Bezugsperson / Angehoeriger (wird NICHT bewertet)
- Das Gespraech findet zwischen dem Kliniker und der Bezugsperson statt, NICHT direkt mit dem Patienten.
- Bewerte ausschliesslich das Verhalten von SPEAKER_00.

EVIDENCE-REGELN (gelten fuer ALLE Items):

Jeder evidence-Eintrag ist ein woertliches Transkriptzitat:
  Format A (Einzelaussage):  "[MM:SS] SPEAKER_00: 'Zitat'"
  Format B (Interaktion):    "[MM:SS] SPEAKER_01: 'Stimulus' -> SPEAKER_00: 'Reaktion'"

VERBOTEN in evidence (gehoert in justification):
  - Analysen, Checklisten, Scan-Ergebnisse, Platzhalter

JSON-REGEL: Verwende in justification-Strings KEINE doppelten Anfuehrungszeichen.
Nutze einfache Anfuehrungszeichen (') fuer Zitate innerhalb von justification.

KUERZE-REGEL:
- justification: max 3 Saetze
- evidence: max 5 Eintraege

Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt. Kein Markdown, kein Text davor/danach.
"""


def build_pass1_prompt(opening_transcript: str, strictness: int = 2) -> str:
    cal = _strictness_note_binary(strictness)
    return SYSTEM_PREAMBLE + """
## PASS 1 -- Items A und B

### A) Greeting and Introduction (0 oder 1)

LUCAS-Kriterien:
- Competent (1): Alle vier Elemente vorhanden:
    i)   Begruesst die Bezugsperson
    ii)  Nennt den eigenen vollstaendigen Namen
    iii) Nennt die eigene Rolle / Berufsbezeichnung
    iv)  Gibt eine kurze Erklaerung, warum das Gespraech stattfindet
- Unacceptable (0): Mindestens ein Element fehlt
""" + cal + """
Bewertungsanleitung:
1. Scanne die Eroeffnungssequenz und pruefe jedes Element einzeln.
2. Dokumentiere in der justification, welche Elemente vorhanden/fehlend sind.
3. Fuer Element iv): Die Bezugsperson muss nach dem Satz verstehen koennen, WARUM der Kliniker mit ihr spricht. Blosse Nennung der Anwesenheit des Kindes ohne inhaltlichen Kontext reicht nicht.

### B) Identity Check (0 oder 1)

LUCAS-Kriterien:
- Competent (1): Beide Elemente vorhanden:
    i)  Ueberprueft den vollstaendigen Namen der Bezugsperson / des Patienten
    ii) Ueberprueft mindestens einen weiteren Identifikator (z.B. Geburtsdatum, Adresse, Zimmernummer, Patientenarmband, o.ae.)
- Unacceptable (0): Mindestens ein Element fehlt
""" + cal + """
Bewertungsanleitung:
1. Nur AKTIVE Verifikation durch SPEAKER_00 zaehlt -- wenn SPEAKER_01 sich ungefragt vorstellt, ist das kein Beleg fuer aktive Ueberpruefung.
2. Der zweite Identifikator muss geeignet sein, den Patienten von anderen Patienten zu unterscheiden. Beziehungsbeschreibungen ('Ihr Sohn', 'die Mutter von') allein sind keine eindeutigen Identifikatoren.

## Transkript (erste 3 Minuten)

""" + opening_transcript + """

Antworte mit:
{
  "items": [
    {
      "item": "A",
      "rating": <0 oder 1>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "<welche Elemente vorhanden/fehlend>",
      "evidence": ["<Zitate>"],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "B",
      "rating": <0 oder 1>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["<Zitate>"],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass2_prompt(full_transcript: str, strictness: int = 2) -> str:
    cal = _strictness_note_graded(strictness)
    return SYSTEM_PREAMBLE + """
## PASS 2 -- Items C und E

### C) Audibility and Clarity of Speech (0, 1 oder 2)

LUCAS-Kriterien:
- Competent (2): Kliniker spricht durchgehend klar und verstaendlich. Medizinische Fachbegriffe werden in laienverstaendliche Sprache uebersetzt, ueberwiegend bevor die Bezugsperson nachfragen muss.
- Borderline (1): Sprache grundsaetzlich verstaendlich, aber Fachbegriffe werden haeufig erst nach Rueckfrage erklaert, oder Erklaerungen sind teilweise unklar.
- Unacceptable (0): Sprache unverstaendlich, oder Fachbegriffe werden nicht erklaert.
""" + cal + """
Bewertungsanleitung:
1. Identifiziere alle medizinischen Fachbegriffe, die SPEAKER_00 verwendet.
2. Pruefe fuer jeden: Wurde er proaktiv erklaert (vor einer Rueckfrage) oder erst reaktiv (nach Rueckfrage von SPEAKER_01)?
3. Beurteile das Gesamtbild -- ueberwiegend proaktiv = C:2, ueberwiegend reaktiv oder gemischt = C:1, kaum Erklaerungen = C:0.

### E) Questions, Prompts and/or Explanations (0, 1 oder 2)

LUCAS-Kriterien:
- Competent (2): Kliniker exploriert aktiv die Beduerfnisse, Gefuehle und Sorgen der Bezugsperson. Fragen und Erklaerungen sind verstaendlich.
- Borderline (1): Einige Exploration vorhanden, aber lueckenhaft oder oberflaechlich.
- Unacceptable (0): Wesentliche Themen nicht exploriert; Bezugsperson wird nicht einbezogen.
""" + cal + """
Bewertungsanleitung:
1. Scanne das gesamte Transkript nach Fragen von SPEAKER_00, die die Perspektive, Sorgen, Vorgeschichte oder das Verstaendnis der Bezugsperson explorieren.
2. Unterscheide proaktive Exploration (SPEAKER_00 initiiert) von reaktiver (SPEAKER_00 antwortet nur auf SPEAKER_01-Fragen).
3. Beurteile Breite und Tiefe der Exploration insgesamt.

Hinweis: E bewertet Exploration; emotionale Reaktionen und Empathie gehoeren zu F.

## Vollstaendiges Transkript

""" + full_transcript + """

Antworte mit:
{
  "items": [
    {
      "item": "C",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Gesamtbild der Fachbegriff-Erklaerungen>",
      "evidence": ["<Zitate>"],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "E",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Gesamtbild der Exploration>",
      "evidence": ["<Zitate>"],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass3_prompt(video_summary: str, strictness: int = 2) -> str:
    cal = _strictness_note_graded(strictness)
    return SYSTEM_PREAMBLE + """
## PASS 3 -- Item D (Non-verbal Behaviour)

LUCAS-Kriterien (max 2 Punkte):
Beurteilt werden: Blickkontakt, Positionierung, Koerperhaltung, Gesichtsausdruck, Gesten und Manierismen.
- Competent (2): Angemessener Blickkontakt, offene Koerperhaltung, passende Positionierung
- Borderline (1): Teilweise auffaellig in einem oder mehreren Bereichen
- Unacceptable (0): Deutlich unangemessenes nonverbales Verhalten
""" + cal + """
Bewertungsanleitung anhand der Videometriken:

D1 -- Blickkontakt:
gaze_on_target >= 75% -> gut
gaze_on_target < 75% -> D:2 ist ausgeschlossen (hartes Kriterium)

D2 -- Positionierung (nur wenn horizon_valid=true UND reliability != low):
at_patient_eye_level_rate >= 50% -> gut
above_patient_eye_level_rate >= 60% -> problematisch
Sonst: D2 nicht auswertbar -- ignorieren

D3 -- Koerperhaltung:
arm_deviation_median > -0.5 -> offen/neutral
arm_deviation_median <= -0.5 -> verschraenkt/geschlossen

Rating-Logik:
- Alle auswertbaren Kriterien gut -> D:2 (aber D1 < 75% sperrt D:2 absolut)
- Ein Kriterium auffaellig -> D:1
- reliability=low -> D:1 maximum
- Keine Videodaten -> D:1, mit Vermerk 'nicht beurteilbar'

Nenne in der justification die konkreten Metrikwerte fuer jedes Kriterium.

## Videometriken

""" + video_summary + """

Antworte mit:
{
  "items": [
    {
      "item": "D",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Metrikwerte und Bewertung pro Kriterium>",
      "evidence": [
        "[video] D1: gaze=<Wert>%",
        "[video] D2: at_eye_level=<Wert>%, above=<Wert>%",
        "[video] D3: arm_deviation_median=<Wert>"
      ],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass4_prompt(full_transcript: str, spikes_annotation: str, strictness: int = 2) -> str:
    cal = _strictness_note_graded(strictness)
    return SYSTEM_PREAMBLE + """
## PASS 4 -- Items F und G

### F) Empathy and Responsiveness (0, 1 oder 2)

LUCAS-Kriterien:
Beinhaltet Anpassung und Sensibilitaet gegenueber den Beduerfnissen der Bezugsperson.
- Competent (2): Kliniker reagiert einfuehlsam auf emotionale Aeusserungen. Zeigt mindestens eines von: Perspektivuebernahme, explizite Emotionsanerkennung, Validierung -- und kehrt nicht sofort zur Sachinformation zurueck.
- Borderline (1): Emotionale Reaktionen vorhanden, aber durchgehend formelhaft oder oberflaechlich. Kliniker kehrt nach jeder emotionalen Aeusserung sofort zur Sachinformation zurueck.
- Unacceptable (0): Emotionen der Bezugsperson werden durchgehend ignoriert.
""" + cal + """
Bewertungsanleitung:
1. Identifiziere emotionale Aeusserungen von SPEAKER_01 (Angst, Sorge, Unsicherheit, Ablehnung, Ueberwaeltigung).
2. Pruefe jeweils die Reaktion von SPEAKER_00.
3. F:2 vs F:1 Entscheidung: Ein einziger genuiner empathischer Moment (z.B. echte Perspektivuebernahme, nicht-formelhafte Validierung) reicht fuer F:2, auch wenn andere Reaktionen routinierter ausfallen. F:1 nur wenn ALLE Reaktionen formelhaft bleiben.
4. Evidence MUSS als Interaktionspaar formatiert sein: "[MM:SS] SPEAKER_01: '...' -> SPEAKER_00: '...'"

### G) Clarification and Summarising (0, 1 oder 2)

LUCAS-Kriterien:
Beinhaltet das Einholen von Rueckfragen der Bezugsperson.
- Competent (2): Kliniker fasst aktiv zusammen ODER fuehrt explizites Verstaendnis-Checking durch.
- Borderline (1): Mindestens ein echter Klaerungsmoment -- SPEAKER_00 stellt aktiv sicher, dass SPEAKER_01 etwas verstanden hat.
- Unacceptable (0): Keine aktive Klaerung durch SPEAKER_00.
""" + cal + """
Bewertungsanleitung:
1. G bewertet AKTIVE Initiative von SPEAKER_00 -- nicht reaktive Antworten auf Fragen von SPEAKER_01 (das gehoert zu Item E).
2. Standard-Abschlussformeln ('Haben Sie noch Fragen?') allein begruenden kein G:1.
3. Wenn kein aktiver Klaerungsmoment gefunden wird, schreibe dies explizit in evidence.

## Emotionale Hinweise
""" + spikes_annotation + """

## Vollstaendiges Transkript

""" + full_transcript + """

Antworte mit:
{
  "items": [
    {
      "item": "F",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": [
        "<[MM:SS] SPEAKER_01: '...' -> SPEAKER_00: '...'>"
      ],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "G",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass5_prompt(full_transcript: str, interaction_metrics: str, strictness: int = 2) -> str:
    cal = _strictness_note_graded(strictness)
    return SYSTEM_PREAMBLE + """
## PASS 5 -- Item H (Consulting Style and Organisation)

LUCAS-Kriterien (max 2 Punkte):
Beinhaltet Ordnung der Konsultation, Balance offener und geschlossener Fragen, Zeitmanagement.
- Competent (2): Gespraech ist geordnet und gespraechsartig, Bezugsperson wird aktiv beteiligt.
- Borderline (1): Leicht unorganisiert oder unausgewogen; Bezugsperson wird wenig einbezogen.
- Unacceptable (0): Verhoer-artig, planlos, oder Gespraech endet abrupt.
""" + cal + """
Bewertungsanleitung:
1. Beachte den Sprechangteil von SPEAKER_00. Wenn SPEAKER_00 deutlich dominiert (> 75%), pruefe ob aktive Strukturierungsmomente (Gespraechsuebergaenge, Einladungen zum Nachfragen, Zusammenfassungen zwischen Themenbloecken) dies kompensieren.
2. Evidence: Nur Strukturierungsmomente (Uebergaenge, Themenmarkierungen, aktive Beteiligung). NICHT: Begruessung (->A), Abschlussformeln (->G), Explorationsfragen (->E).

## Interaktionsmetriken
""" + interaction_metrics + """

## Vollstaendiges Transkript

""" + full_transcript + """

Antworte mit:
{
  "items": [
    {
      "item": "H",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Sprechangteil nennen, Strukturierung beurteilen>",
      "evidence": ["<Strukturierungsmomente mit Zeitstempel>"],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass6_prompt(full_transcript: str, register_warning: str, strictness: int = 2) -> str:
    cal = _strictness_note_binary(strictness)
    return SYSTEM_PREAMBLE + """
## PASS 6 -- Items I und J

### I) Professional Behaviour (0 oder 2 -- kein Borderline)

LUCAS-Kriterien:
- Competent (2): Hoefliches, ruecksichtsvolles, wuerdevolles Verhalten
- Unacceptable (0): Uebermaessig laessig, desinteressiert, unhoeflich oder gedankenlos
""" + cal + """
Bewertungsanleitung:
1. Pruefe ob SPEAKER_00 durchgehend professionell und respektvoll kommuniziert.
2. Registerwechsel: Wenn SPEAKER_00 informelle 'du'-Formen DIREKT gegenueber SPEAKER_01 verwendet (nicht ueber Dritte sprechend), ist das ein Registerwechsel -> I:0.
3. Post-Simulation Metakommentare am Ende des Gespraechs (z.B. 'Simulationsende') liegen ausserhalb des Bewertungsfensters und sind kein I:0-Grund. Nur wenn solche Kommentare MITTEN im Gespraech das Rollenspiel unterbrechen -> I:0.

""" + (register_warning + "\n" if register_warning else "") + """

### J) Professional Spoken/Verbal Conduct (0 oder 2 -- kein Borderline)

LUCAS-Kriterien:
- Competent (2): Aussagen sind respektvoll, fachlich korrekt, im Rahmen der eigenen Kompetenz, Beruhigung ist angemessen.
- Unacceptable (0): Aussagen sind respektlos, enthalten grobe Sachfehler, ueberschreiten die eigene Kompetenz, oder Beruhigung ist unangemessen/verfruehrt.
""" + cal + """
Bewertungsanleitung:
1. Pruefe ob medizinische Aussagen sachlich korrekt sind.
2. Pruefe ob der Kliniker im Rahmen seiner Kompetenz bleibt.
3. Pruefe ob Beruhigung angemessen ist (keine falschen Versprechen, keine Bagatellisierung).
4. Ehrliche Aussagen ueber eigene Grenzen sind kompetentes Verhalten, kein Fehler.

## Vollstaendiges Transkript

""" + full_transcript + """

Antworte mit:
{
  "items": [
    {
      "item": "I",
      "rating": <0 oder 2>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["<Zitate>"],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "J",
      "rating": <0 oder 2>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["<Zitate>"],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass7_prompt(all_items: list) -> str:
    items_json = json.dumps(all_items, ensure_ascii=False, indent=2)
    total = sum(item.get("rating", 0) for item in all_items)
    return SYSTEM_PREAMBLE + f"""
## PASS 7 -- Aggregation

Du erhaeltst die bewerteten Items A-J. Deine Aufgabe:
1. Schreibe eine overall_summary (3-5 Saetze: Hauptstaerken, wichtigste Entwicklungsbereiche, Gesamteindruck)
2. Pruefe evidence-Duplikate: Kein Zitat darf in mehr als 2 Items als Primaerbeleg erscheinen. Melde Konflikte im Format: "'[Zitat]' erscheint in Items X, Y, Z"
3. total_score = {total} (bereits berechnet -- uebernimm diesen Wert exakt)

## Bewertete Items
{items_json}

Antworte mit:
{{
  "total_score": {total},
  "overall_summary": "<3-5 Saetze>",
  "evidence_conflicts": ["<Liste von Konflikten oder leer>"]
}}
"""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

_PASS_MAX_TOKENS = {
    "pass1": 3000,
    "pass2": 6000,
    "pass3": 2000,
    "pass4": 6000,
    "pass5": 4000,
    "pass6": 4000,
}


class LucasMultipassScorer:
    """
    Replaces the single-pass LUCAS inference in llm_analysis_stage.py.

    Usage:
        scorer = LucasMultipassScorer(backend, cfg)
        result = scorer.score(context)

    Where context is the pipeline context dict with keys:
        diarized_transcript, verbal_features, video_nvb,
        conversation_phases, spikes_annotation (optional)
    """

    def __init__(self, backend, cfg: dict, strictness: int = 2):
        self.backend = backend
        self.cfg = cfg
        self._strictness = strictness

    def score(self, context: dict) -> dict:
        diarized = context.get("diarized_transcript", [])
        video_features = context.get("video_nvb") or {}
        verbal_features = context.get("verbal_features") or {}

        # --- Pre-processing ---
        register_break, register_hits = _detect_register_break(diarized)
        register_warning = _format_register_warning(register_hits)

        opening_transcript = _get_opening_transcript(diarized, max_seconds=180)
        full_transcript = _format_full_transcript(diarized)
        video_summary = self._format_video_summary(video_features)
        interaction_metrics = self._format_interaction_metrics(verbal_features)
        spikes_annotation = context.get("spikes_annotation", "")
        if isinstance(spikes_annotation, dict):
            spikes_annotation = json.dumps(spikes_annotation, indent=2, ensure_ascii=False)
        if register_warning:
            spikes_annotation = spikes_annotation + "\n\n" + register_warning

        # --- Build prompts (strictness injected per-pass into criteria) ---
        s = self._strictness
        prompts = {
            "pass1": build_pass1_prompt(opening_transcript, s),
            "pass2": build_pass2_prompt(full_transcript, s),
            "pass3": build_pass3_prompt(video_summary, s),
            "pass4": build_pass4_prompt(full_transcript, spikes_annotation, s),
            "pass5": build_pass5_prompt(full_transcript, interaction_metrics, s),
            "pass6": build_pass6_prompt(full_transcript, register_warning, s),
        }

        # --- Run passes sequentially ---
        # llama-cpp-python (Llama class) is NOT thread-safe.
        pass_order_exec = ["pass1", "pass2", "pass3", "pass4", "pass5", "pass6"]
        pass_results = {}
        for name in pass_order_exec:
            try:
                pass_cfg = {**self.cfg, "max_tokens": _PASS_MAX_TOKENS.get(name, 4000)}
                pass_results[name] = self._run_pass(name, prompts[name], pass_cfg)
            except Exception as e:
                logger.error(f"Pass {name} failed: {e}")
                pass_results[name] = None

        # --- Collect and validate items ---
        all_items = []
        item_order = ["A", "B", "C", "E", "D", "F", "G", "H", "I", "J"]

        for pass_name in pass_order_exec:
            result = pass_results.get(pass_name)
            if result is None:
                logger.error(f"{pass_name} returned None -- skipping")
                continue
            for item in result.get("items", []):
                validated = _validate_item(item, video_features, register_break)
                all_items.append(validated)

        # Sort into canonical order A-J
        order_map = {label: i for i, label in enumerate(item_order)}
        all_items.sort(key=lambda x: order_map.get(x.get("item", "Z"), 99))

        # --- Pass 7: aggregation ---
        pass7_prompt = build_pass7_prompt(all_items)
        pass7_result = _call_llm(self.backend, pass7_prompt, self.cfg)

        total_score = sum(item.get("rating", 0) for item in all_items)
        overall_summary = ""
        evidence_conflicts = []

        if pass7_result:
            overall_summary = pass7_result.get("overall_summary", "")
            evidence_conflicts = pass7_result.get("evidence_conflicts", [])
            p7_total = pass7_result.get("total_score", total_score)
            if p7_total != total_score:
                logger.warning(
                    f"Pass7 total_score={p7_total} differs from computed {total_score} -- using computed"
                )

        return {
            "lucas_items": all_items,
            "total_score": total_score,
            "overall_summary": overall_summary,
            "evidence_conflicts": evidence_conflicts,
            "register_break_detected": register_break,
            "register_hits": register_hits,
        }

    def _run_pass(self, name: str, prompt: str, cfg: dict | None = None) -> dict:
        logger.info(f"Running {name}...")
        result = _call_llm(self.backend, prompt, cfg if cfg is not None else self.cfg)
        if result is None:
            logger.error(f"{name} failed to parse")
            return {"items": []}
        logger.info(f"{name} complete: {[i.get('item') for i in result.get('items', [])]}")
        return result

    def _format_video_summary(self, video_features: dict) -> str:
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

    def _format_interaction_metrics(self, verbal_features: dict) -> str:
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