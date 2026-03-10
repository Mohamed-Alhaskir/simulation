"""
utils/scenario_map.py
=====================
Resolves a session_id to a canonical scenario_id using the flat mapping
file at data/session_scenario_map.json.

Canonical scenario IDs (must match _SCENARIO_CONFIG keys in llm_analysis.py):
  "Diabetes"        — bad-news delivery, Diagnoseübermittlung DM Typ 1
  "LP_Aufklaerung"  — consent conversation, §630e BGB Lumbalpunktion
  "Bauchschmerzen"  — history-taking, akuter Bauchschmerz

The mapping file uses free-form casing (e.g. "diabetes", "LP_aufklaerung").
resolve_scenario_id() normalises to canonical form before returning so that
callers never need to handle case variants.

Usage
-----
    from utils.scenario_map import resolve_scenario_id

    scenario_id = resolve_scenario_id("session_006")
    # → "Diabetes"  (even if the file contains "diabetes")
"""

import json
import logging
from pathlib import Path

# Path to the mapping file — relative to the project root.
_DEFAULT_MAP_PATH = Path("data/session_scenario_map.json")

# Canonical form lookup: lowercase key → canonical scenario_id.
# Add entries here whenever a new scenario is introduced.
_CANONICAL: dict[str, str] = {
    "diabetes":       "Diabetes",
    "lp_aufklaerung": "LP_Aufklaerung",
    "bauchschmerzen": "Bauchschmerzen",
}

log = logging.getLogger(__name__)


def _load_map(map_path: Path) -> dict[str, str]:
    """Load and return the raw session→scenario mapping dict."""
    if not map_path.exists():
        log.warning(
            f"Session–scenario map not found at '{map_path}'. "
            "All scenario lookups will return empty string."
        )
        return {}
    try:
        with open(map_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.error(
                f"Session–scenario map at '{map_path}' is not a JSON object. "
                "Ignoring."
            )
            return {}
        log.debug(f"Session–scenario map loaded: {len(data)} entries from '{map_path}'")
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.error(f"Failed to load session–scenario map from '{map_path}': {exc}")
        return {}


def _canonicalise(raw_id: str) -> str:
    """
    Normalise a raw scenario ID string to its canonical form.

    Strips whitespace, lowercases, and looks up in _CANONICAL.
    Returns the canonical string if found, otherwise returns the
    stripped original (preserving case) so unknown IDs pass through
    and surface as "Unbekanntes Szenario" in the registry fallback.
    """
    normalised = raw_id.strip().lower()
    return _CANONICAL.get(normalised, raw_id.strip())


def resolve_scenario_id(
    session_id: str,
    map_path: Path | str = _DEFAULT_MAP_PATH,
) -> str:
    """
    Return the canonical scenario_id for a given session_id.

    Parameters
    ----------
    session_id : str
        The session identifier, e.g. "session_006".
    map_path : Path | str
        Path to the JSON mapping file. Defaults to data/session_scenario_map.json.

    Returns
    -------
    str
        Canonical scenario_id, e.g. "Diabetes".
        Returns "" if session_id is not found in the map.
    """
    if not session_id:
        log.debug("resolve_scenario_id called with empty session_id — returning ''")
        return ""

    mapping = _load_map(Path(map_path))
    raw = mapping.get(session_id, "")

    if not raw:
        log.warning(
            f"Session '{session_id}' not found in scenario map at '{map_path}'. "
            "Scenario will fall back to _DEFAULT_SCENARIO_CONFIG."
        )
        return ""

    canonical = _canonicalise(raw)
    if canonical != raw.strip():
        log.info(
            f"Session '{session_id}': scenario '{raw}' normalised → '{canonical}'"
        )
    else:
        log.info(f"Session '{session_id}': scenario resolved → '{canonical}'")

    return canonical