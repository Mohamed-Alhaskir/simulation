"""
lucas_multipass.py

Drop-in replacement for the LUCAS single-pass inference in llm_analysis_stage.py.

Architecture:
  Pass 1 (A, B)    — Introductions only, first 3 min transcript
  Pass 2 (C, E)    — Fachbegriff + Explorations scan, full transcript
  Pass 3 (D)       — Video metrics only, no transcript
  Pass 4 (F, G)    — Emotion scan + Summarising, full transcript
  Pass 5 (H)       — Style/Organisation, full transcript + metrics
  Pass 6 (I, J)    — Professional conduct, full transcript
  Pass 7           — Aggregation: total_score, summary, evidence dedup check

Each pass runs independently. All passes can be parallelised except Pass 7.
"""

import json
import logging
import re
import concurrent.futures
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-rule validators
# Run AFTER each pass to catch threshold violations regardless of LLM output.
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
        gaze = video_features.get("D1_eye_contact", {}).get("gaze_on_target", {}).get("rate")
        d2_rel = video_features.get("D2_positioning", {}).get("reliability", "low")
        overall_rel = video_features.get("I_professional_behaviour_demeanour", {}).get("overall_reliability", "low")

        if gaze is not None and gaze < 0.75 and rating == 2:
            flags.append(f"D1 VIOLATION: gaze={gaze:.1%} < 75% but rating=2 → forced to 1")
            item["rating"] = 1
            item["rating_label"] = "Borderline"
            item["justification"] += f" [VALIDATOR: D1 nicht erfüllt ({gaze:.1%} < 75%) → D:2 gesperrt, korrigiert auf D:1]"

        if overall_rel == "low" and rating == 2:
            flags.append(f"D RELIABILITY VIOLATION: reliability=low but rating=2 → forced to 1")
            item["rating"] = 1
            item["rating_label"] = "Borderline"
            item["justification"] += " [VALIDATOR: reliability=low → D:2 gesperrt, korrigiert auf D:1]"

    elif label == "I":
        # Hard register-break override removed: the regex-based detector cannot
        # distinguish du-directed-at-SPEAKER_01 from du-about-third-person.
        # The Pass 6 prompt now instructs the LLM to make this distinction.
        # The Simulationsende hard rule is also removed: post-simulation meta
        # comments at the end of the session are outside the assessment window.
        pass

    elif label == "G":
        evidence = item.get("evidence", [])

        # Pattern 1: explicit "no clarification" marker → G:0
        no_clarification_marker = "Keine aktive Zusammenfassung oder Verständnis-Check im Transkript identifiziert."
        if any(e == no_clarification_marker for e in evidence) and item.get("rating", 0) > 0:
            flags.append("G MARKER VIOLATION: explicit no-clarification marker but rating > 0 → forced to 0")
            item["rating"] = 0
            item["rating_label"] = "Unacceptable"
            item["justification"] += " [VALIDATOR: Kein Klärungsmoment markiert → G:0 erzwungen]"

        # Pattern 2: all evidence entries are reactive Q&A pairs (SPEAKER_01 asks → SPEAKER_00 answers)
        # A reactive pair is detected when evidence contains "SPEAKER_01:" followed by a question
        # and "SPEAKER_00:" as a response — this is Item E material, not G.
        elif evidence and item.get("rating", 0) > 0:
            reactive_markers = [
                "SPEAKER_01:", "melden Sie sich", "jederzeit Fragen",
                "nicht verstehen", "durchlesen", "Haben Sie noch Fragen",
                "keine konkreten Zahlen", "leider gerade"
            ]
            non_reactive_count = sum(
                1 for e in evidence
                if not any(m in e for m in reactive_markers)
                   and "SPEAKER_00" not in e.split("→")[0]  # not a pair starting with SP01
            )
            # If every entry either starts with SPEAKER_01 or is on negativliste → no genuine G moment
            all_reactive = all(
                any(m in e for m in reactive_markers) or e.startswith("[") and "SPEAKER_01:" in e
                for e in evidence
            )
            if all_reactive:
                flags.append("G REACTIVE VIOLATION: all evidence is reactive Q&A (Item E material) → forced to 0")
                item["rating"] = 0
                item["rating_label"] = "Unacceptable"
                item["justification"] += " [VALIDATOR: Alle evidence-Einträge sind reaktive Antworten (→ Item E) — kein aktiver G-Klärungsmoment → G:0 erzwungen]"

    elif label in ("I", "J"):
        # No borderline allowed for I and J
        if rating == 1:
            flags.append(f"{label} BORDERLINE VIOLATION: rating=1 not allowed → forced to 0")
            item["rating"] = 0
            item["rating_label"] = "Unacceptable"
            item["justification"] += f" [VALIDATOR: {label} hat kein Borderline → 0 oder 2]"

    if flags:
        item["validation_flags"] = flags
        logger.warning(f"Item {label} validation: {flags}")

    return item


def _detect_register_break(diarized_transcript: list) -> tuple[bool, list]:
    """
    Scan SPEAKER_00 turns for informal du-forms.
    Returns (break_detected, list_of_hits).
    """
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
    lines = ["⚠ REGISTERWECHSEL-HINWEIS (automatisch erkannt)"]
    lines.append("SPEAKER_00 verwendet informelle du-Formen. Item I MUSS diese Stellen zitieren und mit I:0 bewerten:")
    for h in hits[:5]:
        lines.append(f"  [{h['timestamp']}] \"{h['text']}\" (Treffer: '{h['match']}')")
    if len(hits) > 5:
        lines.append(f"  ... ({len(hits) - 5} weitere Treffer)")
    return "\n".join(lines)


def _get_opening_transcript(diarized_transcript: list, max_seconds: float = 180.0) -> str:
    """Returns transcript text for first max_seconds only (for Pass 1)."""
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
    """Single LLM call. Returns parsed JSON or None on failure."""
    try:
        raw = backend.generate(prompt, cfg)
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-pass prompt builders
# Each returns a string prompt ready for the LLM.
# ---------------------------------------------------------------------------

SYSTEM_PREAMBLE = """Du bist ein Experte für medizinische Ausbildung und bewertest eine pädiatrische Simulationsübung anhand der Liverpool Undergraduate Communication Assessment Scale (LUCAS).

GESPRÄCHSKONTEXT:
- SPEAKER_00 = Kliniker (wird bewertet)
- SPEAKER_01 = Bezugsperson / Angehöriger (nicht bewertet)
- Das Gespräch findet mit der Bezugsperson statt, NICHT direkt mit dem Patienten.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE-FORMAT-REGEL — gilt für ALLE Items, keine Ausnahmen:

Jeder evidence-Eintrag ist ein wörtliches Transkriptzitat in einem der beiden Formate:

  Format A (Einzelaussage):
    "[MM:SS] SPEAKER_00: 'Zitat'"

  Format B (Interaktionspaar — bevorzugt bei F, G, H):
    "[MM:SS] SPEAKER_01: 'Stimulus' → SPEAKER_00: 'Reaktion'"

VERBOTEN in evidence-Einträgen (gehört ausschließlich in justification):
  ✗ Kriterium-Checklisten   (z.B. "Kriterium 1 — D1: gaze=...")
  ✗ Scan-Ergebnisse         (z.B. "Fachbegriff-Scan: X → proaktiv")
  ✗ Elementprüfungen        (z.B. "i) Begrüßung: vorhanden — Beleg: ...")
  ✗ Analytische Bewertungen (z.B. "fehlt — kein Identifikator")
  ✗ Platzhalter             (z.B. "[x]", "<Wert>")

Die justification enthält die Analyse; evidence enthält nur saubere Transkriptzitate.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt. Kein Markdown, kein Text davor/danach.
"""


def build_pass1_prompt(opening_transcript: str) -> str:
    return SYSTEM_PREAMBLE + """
## PASS 1 — Items A und B

---

### A) Greeting and Introduction (max 1 Punkt)
- Competent (1): Alle vier Elemente vorhanden: i) Begrüßung, ii) eigener Name, iii) Rolle, iv) inhaltlicher Gesprächsanlass
- Unacceptable (0): Mindestens ein Element fehlt

PFLICHT-SCAN Schritt 1-4:
Schritt 1: Scanne das Eröffnungssegment vollständig.
Schritt 2: Prüfe jedes Element einzeln.
Schritt 3: Dokumentiere im Format: `i) Begrüßung: [vorhanden/fehlt] — Beleg: "..." | ii) Name: ... | iii) Rolle: ... | iv) Zweck: ...`
Schritt 4: Rating nur 1 wenn ALLE vier vorhanden.

⚠ Negativliste für Element iv) — KEIN gültiger Beleg:
- Fragen nach dem Wissensstand (z.B. "Was ist Ihr Stand?", "Was wissen Sie schon?", "Was haben Sie bisher gehört?") — explorieren Wissensstand, benennen KEINEN Gesprächsanlass
- Aussagen die nur Anwesenheit des Kindes benennen ("wegen Ihres Kindes", "wegen Ihrem Sohn") — kein inhaltlicher Anlass, AUSSER wenn im selben Atemzug ein Fachgebiet oder Eingriff benannt wird

⚠ Gültige Zweck-Formulierungen — auch wenn kein expliziter Gesprächsanlasssatz vorliegt:
- Nennung des Fachgebiets mit inhaltlichem Bezug: "Ich bin Assistenzarzt in der Neuropädiatrie — wir sind wegen Ihres Sohnes hier" = iv) erfüllt (Fachgebiet = inhaltlicher Kontext)
- Explizite Eingriffs-/Themennennung: "Ich möchte kurz über die Lumbalpunktion sprechen" = iv) erfüllt
- Schockraum-Kontext: "Ich möchte Ihnen den aktuellen Stand erklären" = iv) erfüllt
Faustregel: Könnte die Bezugsperson nach diesem Satz verstehen, WARUM der Kliniker zu ihr gekommen ist und was das Thema ist? Ja → iv) erfüllt.

---

### B) Identity Check (max 1 Punkt)
- Competent (1): Element i) UND ii) belegt mit wörtlichem Zitat
- Unacceptable (0): Ein Element fehlt

Element i) = eigener Name der Bezugsperson aktiv erfragt/bestätigt durch SPEAKER_00
⚠ Ein Zitat von SPEAKER_01 zählt NICHT — wenn die Bezugsperson sich selbst vorstellt (z.B. „Böhm, hallo"), ist das kein Beleg für aktive Verifikation durch SPEAKER_00. Element i) fehlt in diesem Fall.
Element ii) = unabhängiger Patientenidentifikator. Gültig sind ausschließlich:
- Ein explizit genannter Eigenname des Patienten (aktiv bestätigt)
- Geburtsdatum
- Zimmernummer oder vergleichbarer objektiver Identifikator

⚠ Nicht gültig für Element ii) — kein Identifikator, egal wie formuliert:
- Jedes Pronomen oder Verwandtschaftswort das auf den Patienten verweist (z.B. "Ihr Sohn", "Ihre Tochter", "das Kind", "er", "sie")
- Jede Beziehungsbestätigung ("Sie sind die Mutter von...?")
- Jede klinische Aussage über den Patienten (Diagnose, Befund, Symptom)
- Jede Beschreibung ohne expliziten Eigennamen, Datum oder Nummer

Testfrage: Könnte dieser Identifikator eindeutig zwei verschiedene Patienten unterscheiden? Wenn nein → kein gültiger Identifikator → Element ii) fehlt → B:0.

⚠ Negativliste — zwingend B:0:
- Jede Frage nach Wissensstand oder Vorerkrankungen
- Jede Beziehungsbestätigung ("Sie sind die Mutter von...?")
- Jede klinische Aussage (Diagnose, Befund, Laborwert) — kein Identifikator
- Zitat von SPEAKER_01 zählt NICHT als Beleg für SPEAKER_00-Verhalten

GEGENCHECK vor B:1: Nenne wörtliches Zitat für i) UND ii). Fehlt eines → B:0.

---


---

## BEISPIEL — Korrekte Bewertung (Pass 1)

**Transkript-Ausschnitt (fiktiv, aber repräsentativ):**
[00:05] [SPEAKER_00] Guten Tag, ich bin Dr. Müller, Assistenzarzt auf der Kinderstation. Ich möchte kurz mit Ihnen über die geplante Lumbalpunktion sprechen.
[00:12] [SPEAKER_01] Böhm, hallo.
[00:14] [SPEAKER_00] Sind Sie Frau Schmidt?
[00:15] [SPEAKER_01] Ja, genau.
[00:16] [SPEAKER_00] Und Ihr Sohn ist der Leon, geboren am 3. März 2019?
[00:18] [SPEAKER_01] Ja, richtig.

**Korrekte Ausgabe:**
- A: rating=1 — alle vier Elemente vorhanden. iv) Zweck: "Ich möchte kurz mit Ihnen über die geplante Lumbalpunktion sprechen" = klarer inhaltlicher Anlass ✓
- B: rating=1 — Element i): SPEAKER_00 fragt "Sind Sie Frau Schmidt?" → aktive Namensverifikation ✓. Element ii): "geboren am 3. März 2019" = Geburtsdatum → gültiger Identifikator ✓

**Gegenbeispiel — häufige Fehler:**
[00:05] [SPEAKER_00] Guten Tag, Besselmann mein Name, Assistenzarzt. Wir sind ja hier wegen Ihres Kindes.
[00:08] [SPEAKER_01] Frau Böhm, hallo.

- A: rating=0 — iv) fehlt: "wegen Ihres Kindes" benennt nur Anwesenheit, kein Fachgebiet, kein Eingriff, kein Thema ✗
- B: rating=0 — Element i) fehlt: "Frau Böhm" sagt SPEAKER_01, nicht SPEAKER_00. Element ii) fehlt: kein Eigenname, kein Datum, keine Zimmernummer ✗

**Sonderfall Fachgebiet als Zweck:**
[00:06] [SPEAKER_00] Hallo, guten Tag. Mein Name ist Maximilian Gomez. Ich bin Assistenzarzt in der Neuropädiatrie. Wir sind ja hier wegen Ihrem Sohn.
- A: rating=1 — iv) erfüllt: "Neuropädiatrie" + "wegen Ihrem Sohn" zusammen = klarer inhaltlicher Kontext. Die Bezugsperson weiß, dass es um ihr Kind in einem neurologisch-pädiatrischen Kontext geht ✓

**Schockraum-Sonderfall:**
[00:10] [SPEAKER_00] Hallo, Johannes Weiberg, Diensthabender Arzt. Ich übernehme jetzt die Schicht und möchte mich kurz vorstellen und Ihnen den aktuellen Stand zu Ihrem Sohn erklären.
- A: rating=1 — iv) erfüllt: "den aktuellen Stand erklären" = klarer Schockraum-Anlass ✓
- NICHT: "Wir hatten gerade Übergabe, Schichtwechsel" allein wäre grenzwertig — erst wenn der Kliniker den Bezug zum Kind/Patienten herstellt, ist iv) erfüllt.

---
## Transkript (erste 3 Minuten)

""" + opening_transcript + """

---

Antworte mit:
{
  "items": [
    {
      "item": "A",
      "rating": <0 oder 1>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "<Vier-Elemente-Format zwingend>",
      "evidence": ["<[MM:SS] SPEAKER_00: 'Zitat'>"],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "B",
      "rating": <0 oder 1>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["<[MM:SS] SPEAKER_00: 'Zitat für i)'>", "<[MM:SS] SPEAKER_00: 'Zitat für ii)'>"],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass2_prompt(full_transcript: str) -> str:
    return SYSTEM_PREAMBLE + """
## PASS 2 — Items C und E

---

### C) Audibility and Clarity of Speech (max 2 Punkte)
- Competent (2): Fachbegriffe überwiegend proaktiv erklärt (mehr proaktiv als reaktiv, mind. 3 proaktiv)
- Borderline (1): Überwiegend reaktiv oder Gleichstand
- Unacceptable (0): Kaum/keine Erklärungen

PFLICHT Fachbegriff-Scan:
Schritt 1: Alle Fachbegriffe identifizieren
Schritt 2: Für jeden Begriff: vorangehender SPEAKER_01-Turn enthält Frage? → reaktiv. Nein → proaktiv.
Schritt 3: Format: `Fachbegriff-Scan: "[Begriff]" → proaktiv/reaktiv (SPEAKER_01 hat vorher gefragt: ja/nein) | ...`
Schritt 4: Zählen: X proaktiv, Y reaktiv
Schritt 5: Rating vergeben — C:2 wenn Mehrheit proaktiv, C:1 wenn Gleichstand oder Mehrheit reaktiv

⚠ Zentralterm-Regel (NUR bei Aufklärungsgesprächen zu einem Eingriff/Verfahren):
Wenn SPEAKER_01 nach dem zentralen Eingriffsbegriff fragt → Erklärung ist zwingend reaktiv.
Bei Schockraum-Übergabe, Erstgespräch, Anamnesegespräch gilt diese Regel NICHT.

⚠ Anzahl der reaktiven Erklärungen ist KEIN eigenständiges Downgrade-Kriterium.
C:2 hängt ausschließlich von der Mehrheitsregel ab (mehr proaktiv als reaktiv).

---

### E) Questions, Prompts and/or Explanations (max 2 Punkte)
- Competent (2): Mind. 3 proaktive Explorationsfragen ODER 2 proaktive Fragen + mindestens 1 explizite emotionale Exploration
- Borderline (1): 1-2 proaktive Fragen, oder keine aber reaktiv kompetent
- Unacceptable (0): Wesentliche Themen nicht exploriert

PFLICHT Explorations-Scan — EXHAUSTIV:
Scanne JEDEN SPEAKER_00-Turn im vollständigen Transkript auf Explorationsfragen.
Höre NICHT nach 2-3 Fragen auf — scanne das GESAMTE Transkript durch.
Format: `Explorations-Scan: "[Frage]" → proaktiv/reaktiv | ...`
Maximal 20 Einträge. Kein Zitat wiederholen.

⚠ Was als proaktive Explorationsfrage zählt:
- Offene Fragen zu Anamnese/Vorgeschichte die SPEAKER_00 ohne vorherigen SPEAKER_01-Impuls stellt
- Fragen zu Beschwerden, Erkrankungen, Medikamenten, Allergien, Kontakten, Impfungen
- Fragen zu Sorgen, Ängsten, Erwartungen der Bezugsperson
Beispiele: "Hat Ihr Sohn Zeckenkontakt gehabt?", "Gibt es bekannte Erkrankungen?", "Was beschäftigt Sie am meisten?"

⚠ E ≠ F: Explorationsfragen gehören zu E, emotionale Reaktionen zu F. Nie dieselbe Stelle als Hauptbeleg für beide.
⚠ Sprecherzuordnung: Nur SPEAKER_00-Fragen zählen als proaktive Exploration.

---


---

## BEISPIEL — Korrekte Bewertung (Pass 2)

**Beispiel C:2 (Competent):**
Fachbegriff-Scan: "Lumbalpunktion" → reaktiv (SPEAKER_01: "Eine was?") | "Liquor" → proaktiv | "Meningitis" → proaktiv | "Lokalanästhesie" → proaktiv | "Spinalkanal" → proaktiv
Zählung: 4 proaktiv, 1 reaktiv → Mehrheit proaktiv → C:2

**Beispiel C:2 (Competent) — viele reaktive Erklärungen, aber Mehrheit proaktiv:**
Fachbegriff-Scan: "Nervenwasser" → proaktiv | "Hirnhautentzündung" → proaktiv | "Sedierung" → reaktiv | "Antibiotika" → reaktiv | "CT" → proaktiv | "Infektion" → proaktiv | "Gerinnungsprobleme" → proaktiv
Zählung: 5 proaktiv, 2 reaktiv → Mehrheit proaktiv → C:2
(Die Anzahl reaktiver Erklärungen spielt keine Rolle — nur Mehrheitsregel entscheidet)

**Beispiel C:1 (Borderline):**
Fachbegriff-Scan: "Lumbalpunktion" → reaktiv | "Nervenwasser" → proaktiv | "Blinddarmentzündung" → reaktiv | "Infektparameter" → proaktiv | "MRT" → reaktiv
Zählung: 2 proaktiv, 3 reaktiv → Mehrheit reaktiv → C:1

**Beispiel E:2 (Competent) — 3+ proaktive Fragen:**
Explorations-Scan: "Was haben Sie bisher gehört?" → proaktiv | "Hat Ihr Sohn Zeckenkontakt gehabt?" → proaktiv | "Gibt es bekannte Erkrankungen oder Gerinnungsprobleme?" → proaktiv | "Ist er schon mal operiert worden?" → proaktiv | "Wie geht es Ihnen damit?" → proaktiv (emotionale Exploration)
Zählung: 5 proaktiv → E:2

**Beispiel E:2 (Competent) — 2 proaktive + emotionale Exploration:**
Explorations-Scan: "Was haben Sie bisher gehört?" → proaktiv | "Hat Ihr Sohn Gerinnungsprobleme?" → proaktiv | "Ich merke, dass Sie sich Sorgen machen — was beschäftigt Sie am meisten?" → proaktiv (emotionale Exploration)
Zählung: 2 proaktiv + 1 emotionale → E:2

**Beispiel E:1 (Borderline):**
Explorations-Scan: "Was haben Sie bisher gehört?" → proaktiv | "Eine was?" → reaktiv | "Kann da nichts passieren?" → reaktiv
Zählung: 1 proaktiv, keine emotionale Exploration → E:1

---
## Vollständiges Transkript

""" + full_transcript + """

---

Antworte mit:
{
  "items": [
    {
      "item": "C",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Fachbegriff-Scan zwingend mit Zählung>",
      "evidence": ["<1-3 Zitate>"],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "E",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Explorations-Scan zwingend>",
      "evidence": ["<1-3 Zitate>"],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass3_prompt(video_summary: str) -> str:
    return SYSTEM_PREAMBLE + """
## PASS 3 — Item D (Nonverbales Verhalten)

max 2 Punkte. Bewerte AUSSCHLIEßLICH anhand der Videometriken. Keine Transkriptzitate.

### Drei-Kriterien-Regel:

**D1 — Blickkontakt:**
gaze_on_target ≥ 75% → erfüllt
gaze_on_target < 75% → NICHT erfüllt
⚠ HARTES AUSSCHLUSSKRITERIUM: gaze < 75% → D:2 GESPERRT, unabhängig von D2/D3 und reliability.
Schreibe explizit: "D1 nicht erfüllt (gaze=X% < 75%) → D:2 gesperrt."

**D2 — Positionierung (NUR wenn horizon_valid=true UND D2_reliability ≠ low):**
at_patient_eye_level_rate ≥ 50% → erfüllt
above_patient_eye_level_rate ≥ 60% → D:2 ausgeschlossen
Sonst: "D2 nicht auswertbar — Kriterium ignoriert"

**D3 — Körperhaltung:**
arm_deviation_median > -0.5 → erfüllt
arm_deviation_median ≤ -0.5 → nicht erfüllt

**Rating:**
- Alle erfüllten Kriterien → D:2 (aber D1 < 75% sperrt D:2 absolut)
- Ein Kriterium nicht erfüllt → D:1
- reliability=low → D:1 maximum

PFLICHT-Justification-Format (ersetze alle Platzhalter durch tatsächliche Werte aus den NVB-Daten):
```
Kriterium 1 — D1: gaze=<konkreter Prozentwert>%, data_availability=<konkreter Prozentwert>% → <erfüllt ODER nicht erfüllt>
Kriterium 2 — D2: horizon_valid=<true/false>, D2_reliability=<Wert>, at_eye_level=<Wert>%, above=<Wert>% → <erfüllt / nicht erfüllt / nicht anwendbar>
Kriterium 3 — D3: arm_deviation_median=<Wert> → <erfüllt ODER nicht erfüllt>
Gesamtreliability: <Wert>
Rating-Entscheidung: <Begründung> → D:<0/1/2>
```

⚠ Wenn die NVB-Daten "Keine Videodaten verfügbar" sind: Schreibe in justification: "Keine Videodaten verfügbar — Item D nicht beurteilbar." und vergib D:1. Verwende NIEMALS Platzhalter wie [x] im Output.

---


---

## BEISPIEL — Korrekte Bewertung (Pass 3)

**Beispiel D:2 (Competent):**
D1: gaze=82%, reliability=high → erfüllt (≥75%)
D2: horizon_valid=true, D2_reliability=moderate, at_eye_level=65%, above=20% → erfüllt
D3: arm_deviation_median=-0.12 → erfüllt (>-0.5)
→ Alle Kriterien erfüllt → D:2

**Beispiel D:1 (Borderline) — D2 sperrt:**
D1: gaze=90%, reliability=moderate → erfüllt
D2: horizon_valid=true, D2_reliability=low, at_eye_level=0%, above=100% → D:2 gesperrt (above≥60%)
D3: arm_deviation_median=-0.22 → erfüllt
→ D2 sperrt trotz D1+D3 erfüllt → D:1

**Beispiel D:1 (Borderline) — D1 sperrt:**
D1: gaze=68%, reliability=high → NICHT erfüllt (<75%) → D:2 gesperrt ABSOLUT
D2: at_eye_level=54% → würde erfüllen, aber irrelevant
D3: arm_deviation_median=-0.24 → erfüllt
→ D1 sperrt → D:1 (auch wenn D2+D3 erfüllt wären)

**Beispiel D:1 — reliability=low:**
Gesamtreliability=low → D:2 gesperrt unabhängig von Metrikwerten → D:1

---
## Videometriken

""" + video_summary + """

---

Antworte mit:
{
  "items": [
    {
      "item": "D",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Kriterien-Analyse mit konkreten Metrikwerten — Pflicht-Format zwingend>",
      "evidence": [
        "[video] D1: gaze=<Wert>%, data_availability=<Wert>%, reliability=<Wert>",
        "[video] D2: at_eye_level=<Wert>%, above=<Wert>%, reliability=<Wert>",
        "[video] D3: arm_deviation_median=<Wert>, reliability=<Wert>"
      ],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass4_prompt(full_transcript: str, spikes_annotation: str) -> str:
    return SYSTEM_PREAMBLE + """
## PASS 4 — Items F und G

---

### F) Empathy and Responsiveness (max 2 Punkte)
- Competent (2): Mindestens EINER dieser F:2-Belege vorhanden UND Muster nicht durchgehend formelhaft:
  • Perspektivübernahme: Kliniker versetzt sich in Lage der Bezugsperson ("Wäre es mein Kind / mein Sohn, ich würde genauso reagieren")
  • Emotionsanerkennung + Raum: Emotion benannt/gespiegelt UND danach keine sofortige Sachinformation
  • Explizite Validierung: "Das ist eine verständliche Reaktion", "Das ist schwer zu hören"
- Borderline (1): Reaktionen vorhanden aber durchgehend formelhaft, kehrt sofort zu Sachinformation zurück — kein einziger F:2-Beleg
- Unacceptable (0): Emotionen durchgehend ignoriert

PFLICHT Emotionsscan: Identifiziere alle emotionalen Äußerungen von SPEAKER_01 (Angst, Sorge, Ablehnung, Überwältigung).
Wähle 3 Momente: früh / mittig / intensivst.

Evidence-Format — JEDER Eintrag MUSS ein Paar sein:
"[MM:SS] SPEAKER_01: '[Emotion]' → SPEAKER_00: '[Reaktion]'"
Alle 3 Einträge müssen UNTERSCHIEDLICHE Zeitstempel haben.
Ein SPEAKER_00-Zitat ohne vorangehendes SPEAKER_01-Zitat ist UNGÜLTIG.

GEGENCHECK vor F:2: Findet sich im Transkript MINDESTENS EIN F:2-Beleg (Perspektivübernahme ODER Emotionsanerkennung+Raum ODER explizite Validierung)? Ja → F:2. Nein → F:1.

---

### G) Clarifying and Summarising (max 2 Punkte)
- Competent (2): Aktive Zusammenfassung mit wörtlichem Zitat ODER explizites Verständnis-Checking
- Borderline (1): Mindestens ein echter (nicht formelhafter) Klärungsmoment

⚠ Was ist ein echter Klärungsmoment (G:1)?
Ein echter Klärungsmoment ist eine Stelle, an der SPEAKER_00 AKTIV sicherstellt, dass SPEAKER_01 etwas verstanden hat — z.B.:
- "Ist das soweit verständlich?" / "Können Sie das nachvollziehen?"
- Eine Zusammenfassung eines Teilthemas in eigenen Worten
- Explizites Angebot, etwas zu wiederholen oder zu erklären, nachdem Verwirrung erkennbar war

⚠ Was ist KEIN echter Klärungsmoment (→ nicht G:1, sondern auf Negativliste):
- SPEAKER_00 beantwortet eine direkte Frage von SPEAKER_01 (→ Item E, nicht G)
- SPEAKER_00 gibt zu, keine Zahlen zu haben ("Ich habe leider keine konkreten Zahlen")
- SPEAKER_00 erklärt medizinische Inhalte (→ Item C/E)
- Reaktive Antworten jeder Art — G erfordert eine AKTIVE Initiative von SPEAKER_00
- Unacceptable (0): Nur Negativliste-Kandidaten oder gar nichts

GEGENCHECK vor G:2: Finde wörtliches Zitat für aktive Zusammenfassung oder Verständnis-Check.
Kein Zitat gefunden → G:1 maximum.

⚠ G Negativliste — zählt NICHT:
- "Wenn Sie noch Fragen haben, melden Sie sich"
- "Wenn Sie irgendwas nicht verstehen, dann melden Sie sich nochmal"
- "Haben Sie noch Fragen?" als Abschlussformel
- Dokumentübergabe, Formulare, Erklärungen medizinischer Inhalte

⚠ G:0-Regel: Wenn ALLE Kandidaten auf der Negativliste stehen → G:0, nicht G:1.

Evidence bei fehlendem Befund: Schreibe "Keine aktive Zusammenfassung oder Verständnis-Check im Transkript identifiziert."

---


---

## BEISPIEL — Korrekte Bewertung (Pass 4)

**Beispiel F:2 (Competent) — Perspektivübernahme:**
[02:09] SPEAKER_01: "Das ist, das hätte man auch am ersten Tag machen können." → SPEAKER_00: "Ja. Wäre es mein Sohn, dann wäre ich auch genauso aufgeregt."
→ Perspektivübernahme ("Wäre es mein Sohn") = F:2-Beleg. Mindestens ein solcher Moment reicht für F:2.

**Beispiel F:2 (Competent) — Emotion benannt + Raum:**
[03:12] SPEAKER_01: "Das macht mir wirklich Angst." → SPEAKER_00: "Das klingt wirklich beängstigend. Was genau macht Ihnen am meisten Sorgen?" [benennt Emotion + lässt Raum]
→ Emotion benannt + Raum gelassen → F:2

**Beispiel F:1 (Borderline) — kein einziger F:2-Beleg:**
[05:24] SPEAKER_01: "Das klingt so dramatisch." → SPEAKER_00: "Ich kann mir vorstellen, dass es erstmal beängstigend wirkt. Aber es ist wichtig, dass wir das abklären."
[02:28] SPEAKER_01: "Da hat er schon Panik." → SPEAKER_00: "Ja, ich weiß. Aber wir würden das in lokaler Anästhesie machen."
→ Kein Moment der Perspektivübernahme, Emotionsanerkennung oder Validierung ohne sofortige Sachinformation → F:1

**Beispiel G:2 (Competent):**
[08:15] SPEAKER_00: "Lassen Sie mich kurz zusammenfassen was wir besprochen haben: Ihr Sohn braucht eine Lumbalpunktion weil wir Meningitis ausschließen wollen. Die Risiken sind gering. Haben Sie das so verstanden?"
→ Aktive Zusammenfassung mit Verständnis-Check → G:2

**Beispiel G:1 (Borderline):**
[09:16] SPEAKER_00: "Verstehen Sie das?" [einmaliger Verständnis-Check, keine Zusammenfassung]
→ Echter Klärungsmoment, aber keine Zusammenfassung → G:1

**Beispiel G:0 (Unacceptable):**
Alle evidence-Kandidaten: reaktive Antworten auf SPEAKER_01-Fragen, Abschlussformel "Wenn Sie Fragen haben, melden Sie sich"
→ Kein aktiver Klärungsmoment von SPEAKER_00 initiiert → G:0

---
## Emotionale Hinweise
""" + spikes_annotation + """

## Vollständiges Transkript

""" + full_transcript + """

---

Antworte mit:
{
  "items": [
    {
      "item": "F",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Gegencheck zwingend>",
      "evidence": [
        "<[MM:SS] SPEAKER_01: '...' → SPEAKER_00: '...'>",
        "<[MM:SS] SPEAKER_01: '...' → SPEAKER_00: '...'>",
        "<[MM:SS] SPEAKER_01: '...' → SPEAKER_00: '...'>"
      ],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "G",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "...",
      "evidence": ["..."],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass5_prompt(full_transcript: str, interaction_metrics: str) -> str:
    return SYSTEM_PREAMBLE + """
## PASS 5 — Item H (Consulting Style and Organisation)

max 2 Punkte.

- Competent (2): Gesprächsartig und geordnet, Bezugsperson aktiv beteiligt
- Borderline (1): Leicht unorganisiert, unausgewogener Fragetypen-Einsatz
- Unacceptable (0): Verhörartig oder planlos, Konsultation endet abrupt

⚠ Sprechanteile:
SPEAKER_00 > 75% → H:1 als Default, AUSSER wenn im Transkript mehrere explizite Strukturierungsmomente belegt sind, die zeigen dass der Kliniker aktiv das Gespräch steuert und der Bezugsperson Raum schafft.
SPEAKER_00 ≤ 75% → H:2 möglich wenn aktive Beteiligung belegt.
Nenne Sprechangteil von SPEAKER_00 explizit in der justification.

⚠ Strukturierungsmoment zählt für H:2-Override wenn es AKTIV ist:
- Explizite Gesprächsübergänge mit Bezug auf die Bezugsperson ("Trotzdem möchte ich mit Ihnen jetzt darüber reden, damit Sie Bescheid wissen")
- Aktives Einladen zum Nachfragen: "Wenn Sie irgendwas nicht verstehen, fragen Sie ruhig"
- Thematische Zusammenfassung vor Weitergehen
Faustformel: Wenn der Kliniker > 75% spricht ABER das Gespräch erkennbar strukturiert führt und Raum für die Bezugsperson schafft, ist H:2 möglich.

⚠ H-evidence — NUR zulässig:
- Explizite Gesprächsübergänge ("So, dann kommen wir jetzt zu...")
- Strukturierende Fragen vor Themenwechsel
- Zusammenfassungsmomente zwischen Phasen

⚠ NICHT zulässig als H-Beleg:
- Begrüßungssätze (→ Item A)
- Abschlussfragen (→ Item G)
- Explorationsfragen (→ Item E)
- Rein inhaltliche Erklärungen
- Eröffnungssequenz (Begrüßung, Name, Rolle)

---


---

## BEISPIEL — Korrekte Bewertung (Pass 5)

**Beispiel H:2 (Competent) — Sprechangteil 76%, aktive Strukturierung:**
SPEAKER_00 Sprechangteil: 76% → über 75%, aber aktive Strukturierungsmomente vorhanden.
[02:37] SPEAKER_00: "Trotzdem möchte ich mit Ihnen jetzt darüber reden, was wir machen wollen, damit Sie auch Bescheid wissen." [aktive Einbeziehung mit klarem Zweck]
[09:16] SPEAKER_00: "Verstehen Sie das?" [aktiver Verständnis-Check]
[17:44] SPEAKER_00: "Machen Sie das ruhig und dann kommen Sie möglichst zeitnah zu mir nochmal." [aktiver nächster Schritt für Bezugsperson]
→ Mehrere Strukturierungsmomente + Bezugsperson aktiv einbezogen trotz 76% → H:2

**Beispiel H:2 (Competent) — Sprechangteil 68%:**
SPEAKER_00 Sprechangteil: 68% → unter 75%, H:2 möglich wenn aktive Beteiligung belegt.
[02:10] SPEAKER_00: "Bevor ich weitermache — was ist Ihre größte Sorge gerade?" [aktive Einbeziehung]
[05:33] SPEAKER_00: "So, jetzt haben wir Risiken besprochen. Kommen wir zu den nächsten Schritten." [expliziter Übergang]
→ Mehrere Strukturierungsmomente + aktive Beteiligung → H:2

**Beispiel H:1 (Borderline) — Sprechangteil 84%, keine Strukturierung:**
SPEAKER_00 Sprechangteil: 84% → über 75% → H:1 als Default.
Keine expliziten Strukturierungsmomente, kein Raum für Bezugsperson → H:1

**Beispiel H:1 — Sprechangteil 73%, keine Strukturierung:**
Sprechangteil knapp unter 75%, aber keine expliziten Strukturierungsmomente und Bezugsperson kaum aktiv eingebunden → H:1
(Sprechangteil allein reicht nicht für H:2 — aktive Strukturierung muss im Transkript belegt sein)

---
## Interaktionsmetriken
""" + interaction_metrics + """

## Vollständiges Transkript

""" + full_transcript + """

---

Antworte mit:
{
  "items": [
    {
      "item": "H",
      "rating": <0, 1 oder 2>,
      "rating_label": "<Competent|Borderline|Unacceptable>",
      "justification": "<Sprechangteil zwingend nennen>",
      "evidence": ["<1-3 Strukturierungsmomente mit Zeitstempel>"],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass6_prompt(full_transcript: str, register_warning: str) -> str:
    return SYSTEM_PREAMBLE + """
## PASS 6 — Items I und J

---

### I) Professional Behaviour (0 oder 2 — kein Borderline)
- Competent (2): Durchgehend höflich, rücksichtsvoll, würdevoll
- Unacceptable (0): Unprofessionell, lässig, unhöflich

PFLICHT 1 — Simulationsende-Check:
Suche nach "Simulationsende", "Ende der Simulation" oder vergleichbaren Metakommentaren.
⚠ "Simulationsende, Dankeschön" am Ende des Gesprächs ist ein POST-SIMULATION Metakommentar und liegt AUSSERHALB des Bewertungsfensters.
Es wird NICHT als I:0 gewertet, da es kein Verhalten gegenüber der Bezugsperson ist.
Nur wenn ein solcher Metakommentar MITTEN im Gespräch erscheint und das Rollenspiel aktiv unterbricht → I:0.

PFLICHT 2 — Registerwechsel-Check:
Suche nach informellem 'du' von SPEAKER_00 gegenüber SPEAKER_01.
⚠ Bestätigtes 'du' gegenüber SPEAKER_01 (direkt angesprochen, nicht über Drittperson) → I:0 ZWINGEND.
⚠ WICHTIG: Prüfe ob das "du" tatsächlich an SPEAKER_01 gerichtet ist. Wenn SPEAKER_00 über eine dritte Person spricht ("du" = jemand anderes, z.B. über das Kind), ist es KEIN Registerwechsel.

""" + (register_warning if register_warning else "") + """

GEGENCHECK vor I:2:
1. Metakommentar MITTEN im Gespräch vorhanden (nicht am Ende)? → I:0
2. Informelles 'du' direkt an SPEAKER_01 gerichtet? → I:0
3. Frustrationsmomente würdevoll behandelt?
Schreibe Gegencheck-Befund explizit in justification.

---

### J) Professional Spoken/Verbal Conduct (0 oder 2 — kein Borderline)
- Competent (2): Professionell, keine Sachfehler, im Rahmen der Kompetenz
- Unacceptable (0): Sachfehler, abwertend, falsche/verfrühte Beruhigung, irreführende Häufigkeitsangaben

⚠ Ehrliche Aussagen zur eigenen Erfahrung sind kompetentes Verhalten.
⚠ J-evidence: Nur Risikoangaben, Beruhigungsformeln, Kompetenzaussagen — keine Fachbegriff-Erklärungen (→ C), keine Explorationsfragen (→ E).

---


---

## BEISPIEL — Korrekte Bewertung (Pass 6)

**Beispiel I:2 (Competent):**
Simulationsende-Check: "[18:15] Simulationsende, Dankeschön." → post-simulation Metakommentar am Ende, außerhalb Bewertungsfenster → kein I:0 ✓
Registerwechsel-Check: kein "du" von SPEAKER_00 gegenüber SPEAKER_01 gefunden ✓
Verhalten durchgehend höflich und professionell ✓
→ I:2

**Beispiel I:0 — Simulationsende MITTEN im Gespräch:**
[05:30] SPEAKER_00: "Simulationsende." [unterbricht aktiv das Rollenspiel mitten im Gespräch] → I:0

**Beispiel I:0 — Registerwechsel:**
[10:22] SPEAKER_00: "Du kennst ja nichts dafür." → informelles "du" direkt an SPEAKER_01 gerichtet → I:0
⚠ Wenn kein Metakommentar mitten im Gespräch UND kein "du" gegenüber SPEAKER_01 → I:2, nicht I:0.

**Beispiel J:2 (Competent):**
Keine Sachfehler, keine verfrühte Beruhigung, bleibt im Rahmen der Kompetenz, ehrliche Risikoangaben → J:2

**Beispiel J:0 (Unacceptable):**
"Das passiert eigentlich nie, machen Sie sich keine Sorgen." → falsche Häufigkeitsangabe + verfrühte Beruhigung → J:0

---
## Vollständiges Transkript

""" + full_transcript + """

---

Antworte mit:
{
  "items": [
    {
      "item": "I",
      "rating": <0 oder 2>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "<Simulationsende-Prüfung + Registerwechsel-Prüfung zwingend>",
      "evidence": ["<1-3 Zitate professionellen Verhaltens>"],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    },
    {
      "item": "J",
      "rating": <0 oder 2>,
      "rating_label": "<Competent|Unacceptable>",
      "justification": "...",
      "evidence": ["<1-3 Zitate>"],
      "strengths": [],
      "gaps": [],
      "next_steps": [],
      "confidence": "<high|medium|low>"
    }
  ]
}
"""


def build_pass7_prompt(all_items: list) -> str:
    items_json = json.dumps(all_items, ensure_ascii=False, indent=2)
    total = sum(item.get("rating", 0) for item in all_items)
    return SYSTEM_PREAMBLE + f"""
## PASS 7 — Aggregation

Du erhältst die bewerteten Items A-J. Deine Aufgabe:
1. Schreibe eine overall_summary (3-5 Sätze: Hauptstärken, wichtigste Entwicklungsbereiche, Gesamteindruck)
2. Prüfe evidence-Duplikate: Kein Zitat darf in mehr als 2 Items als Primärbeleg erscheinen. Melde Konflikte.
3. total_score = {total} (bereits berechnet — übernehme diesen Wert exakt)

## Bewertete Items
{items_json}

Antworte mit:
{{
  "total_score": {total},
  "overall_summary": "<3-5 Sätze>",
  "evidence_conflicts": ["<Liste von Konflikten oder leer>"]
}}
"""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

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

    @staticmethod
    def _make_strictness_preamble(level: int) -> str:
        if level == 1:
            return (
                "SCORING CALIBRATION — LENIENT (Level 1/3)\n"
                "Apply benefit of the doubt throughout.\n"
                "- Implied or inferable behaviour warrants credit.\n"
                "- Borderline cases: score UP.\n"
                "- Focus on what the clinician achieved.\n\n"
            )
        if level == 3:
            return (
                "SCORING CALIBRATION — STRICT (Level 3/3)\n"
                "Apply a high evidentiary standard throughout.\n"
                "- Only explicit, unambiguous, clearly observable behaviour receives credit.\n"
                "- No inference; ambiguous cases score DOWN.\n"
                "- This level reflects a high-stakes examination standard.\n\n"
            )
        return ""  # level 2 = standard, no preamble

    def score(self, context: dict) -> dict:
        diarized = context.get("diarized_transcript", [])
        video_features = context.get("video_nvb", {})
        verbal_features = context.get("verbal_features", {})

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

        # --- Build prompts ---
        _sp = LucasMultipassScorer._make_strictness_preamble(self._strictness)
        prompts = {
            "pass1": _sp + build_pass1_prompt(opening_transcript),
            "pass2": _sp + build_pass2_prompt(full_transcript),
            "pass3": _sp + build_pass3_prompt(video_summary),
            "pass4": _sp + build_pass4_prompt(full_transcript, spikes_annotation),
            "pass5": _sp + build_pass5_prompt(full_transcript, interaction_metrics),
            "pass6": _sp + build_pass6_prompt(full_transcript, register_warning),
        }

        # --- Run passes sequentially ---
        # llama-cpp-python (Llama class) is NOT thread-safe.
        # Concurrent calls corrupt state or silently drop results.
        # Passes run in order; each completes before the next starts.
        pass_order_exec = ["pass1", "pass2", "pass3", "pass4", "pass5", "pass6"]
        pass_results = {}
        for name in pass_order_exec:
            try:
                pass_results[name] = self._run_pass(name, prompts[name])
            except Exception as e:
                logger.error(f"Pass {name} failed: {e}")
                pass_results[name] = None

        # --- Collect and validate items ---
        all_items = []
        pass_order = ["pass1", "pass2", "pass3", "pass4", "pass5", "pass6"]
        item_order = ["A", "B", "C", "E", "D", "F", "G", "H", "I", "J"]

        for pass_name in pass_order:
            result = pass_results.get(pass_name)
            if result is None:
                logger.error(f"{pass_name} returned None — skipping")
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
            # Verify pass7 didn't compute a different total
            p7_total = pass7_result.get("total_score", total_score)
            if p7_total != total_score:
                logger.warning(
                    f"Pass7 total_score={p7_total} differs from computed {total_score} — using computed"
                )

        return {
            "lucas_items": all_items,
            "total_score": total_score,
            "overall_summary": overall_summary,
            "evidence_conflicts": evidence_conflicts,
            "register_break_detected": register_break,
            "register_hits": register_hits,
        }

    def _run_pass(self, name: str, prompt: str) -> dict:
        """Run a single pass and return parsed result."""
        logger.info(f"Running {name}...")
        result = _call_llm(self.backend, prompt, self.cfg)
        if result is None:
            logger.error(f"{name} failed to parse")
            return {"items": []}
        logger.info(f"{name} complete: {[i.get('item') for i in result.get('items', [])]}")
        return result

    def _format_video_summary(self, video_features: dict) -> str:
        """
        Formats video metrics as flat human-readable strings for Pass 3.
        Avoids passing raw nested JSON to the model.
        """
        if not video_features:
            return "Keine Videodaten verfügbar."

        lines = []

        d1 = video_features.get("D1_eye_contact", {})
        gaze_rate = d1.get("gaze_on_target", {}).get("rate")
        d1_rel = d1.get("reliability", "unbekannt")
        data_avail = d1.get("data_availability_rate")
        if gaze_rate is not None:
            gaze_pct = round(gaze_rate * 100)
            threshold_note = "< 75% → D:2 gesperrt" if gaze_pct < 75 else "≥ 75% → erfüllt"
            lines.append(
                f"D1_Blickkontakt: gaze_on_target={gaze_pct}% ({threshold_note}), "
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
            block_note = "D:2 gesperrt (above ≥ 60%)" if above_pct >= 60 else (
                "Downgrade auf D:1 prüfen" if above_pct >= 40 else "günstig"
            )
            lines.append(
                f"D2_Positionierung: horizon_valid=true, at_eye_level={at_pct}%, "
                f"above={above_pct}% → {block_note}, D2_reliability={d2_rel}"
            )
        else:
            lines.append(
                f"D2_Positionierung: horizon_valid={horizon_valid}, D2_reliability={d2_rel} "
                f"→ D2-Kriterium nicht anwendbar"
            )

        d3 = video_features.get("D3_posture", {})
        d3_rel = d3.get("reliability", "unbekannt")
        arm_dev = d3.get("baseline_arm_deviation", {}).get("median")
        if arm_dev is not None:
            arm_note = "erfüllt (offen/neutral)" if arm_dev > -0.5 else "nicht erfüllt (verschränkt)"
            lines.append(
                f"D3_Haltung: arm_deviation_median={arm_dev:.2f} → {arm_note}, "
                f"reliability={d3_rel}"
            )

        # Overall reliability from I_professional block
        overall_rel = video_features.get(
            "I_professional_behaviour_demeanour", {}
        ).get("overall_reliability", "unbekannt")
        lines.append(f"Gesamtreliability: {overall_rel}")

        return "\n".join(lines)

    def _format_interaction_metrics(self, verbal_features: dict) -> str:
        """Formats verbal/interaction metrics for Pass 5."""
        if not verbal_features:
            return "Keine Interaktionsmetriken verfügbar."

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
            f"Ø Turndauer SPEAKER_00: {s00.get('avg_turn_duration_s', '?'):.1f}s"
            if isinstance(s00.get('avg_turn_duration_s'), float) else
            f"Ø Turndauer SPEAKER_00: ?"
        )
        lines.append(f"Gesprächsdauer gesamt: {summary.get('total_duration_s', '?'):.0f}s")
        lines.append(f"Bedeutungsvolle Pausen: {summary.get('meaningful_pauses', '?')}")
        lines.append(f"Unterbrechungen: {summary.get('interruptions', '?')}")

        return "\n".join(lines)