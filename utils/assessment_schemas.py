"""
Assessment framework definitions and rubrics.

Defines the schemas, items, and scoring rules for:
- LUCAS: 10-item communication skills framework
- SPIKES: 6-step bad-news delivery protocol
- Clinical Content: Scenario-specific medical knowledge
"""


# ──────────────────────────────────────────────────────────────────────
# LUCAS (University of Liverpool Communication Assessment System)
# ──────────────────────────────────────────────────────────────────────

LUCAS_MAX_SCORE = 18

LUCAS_ITEMS: list[dict] = [
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
