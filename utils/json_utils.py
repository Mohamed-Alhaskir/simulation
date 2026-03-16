"""
JSON serialization utilities for handling non-standard types.
"""

import json
import math
from pathlib import Path
from decimal import Decimal


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for non-standard types."""

    def default(self, obj):
        """Handle non-standard types."""
        # numpy types (convert to Python native via .item())
        if hasattr(obj, "item"):
            val = obj.item()
            # Handle NaN/Inf
            if isinstance(val, float):
                if math.isnan(val) or math.isinf(val):
                    return None
            return val

        # Handle plain float NaN/Inf
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None

        # numpy/pandas boolean
        if isinstance(obj, (bool, type(True))):
            return bool(obj)

        # numpy arrays
        if hasattr(obj, "tolist"):
            return obj.tolist()

        # Path objects
        if isinstance(obj, Path):
            return str(obj)

        # Decimal
        if isinstance(obj, Decimal):
            return float(obj)

        # Fallback to string representation
        return str(obj)


def sanitize_for_json(obj):
    """
    Recursively convert non-JSON-serializable objects to JSON-serializable equivalents.

    Handles:
    - numpy types (bool_, int64, float64, arrays, etc.)
    - NaN and Infinity values (converted to None)
    - Path objects
    - Decimal values
    - Nested structures (dicts, lists)

    Parameters
    ----------
    obj : any
        Object to sanitize

    Returns
    -------
    any
        JSON-serializable version of obj
    """
    # None, bool, int, str are already JSON-serializable
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    # Handle plain float (including NaN/Inf)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # numpy/pandas types
    if hasattr(obj, "item"):
        val = obj.item()
        # Recursively sanitize the extracted value
        return sanitize_for_json(val)

    # numpy arrays and similar
    if hasattr(obj, "tolist"):
        arr_list = obj.tolist()
        return sanitize_for_json(arr_list)

    # Path objects
    if isinstance(obj, Path):
        return str(obj)

    # Decimal
    if isinstance(obj, Decimal):
        return float(obj)

    # Dictionaries
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    # Lists and tuples
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]

    # Fallback: convert to string
    return str(obj)


def repair_unescaped_quotes(text: str) -> str:
    """
    Repair common LLM JSON output errors: unescaped quotes inside string values.

    The LLM may emit strings like:
        "field": "value with "embedded" quotes"

    This function detects when a " is inside a JSON string (between two unescaped ")
    and escapes the embedded quote with a backslash, so it becomes:
        "field": "value with \"embedded\" quotes"

    Parameters
    ----------
    text : str
        Raw JSON text with potential unescaped quotes

    Returns
    -------
    str
        Repaired JSON text
    """
    out = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(text):
        ch = text[i]

        if escape_next:
            out.append(ch)
            escape_next = False
            i += 1
            continue

        if ch == '\\':
            out.append(ch)
            escape_next = True
            i += 1
            continue

        if ch == '"':
            if not in_string:
                # Entering a string
                in_string = True
                out.append(ch)
            else:
                # Inside a string; check if this is the closing quote
                j = i + 1
                while j < len(text) and text[j] in ' \t\r\n':
                    j += 1
                next_ch = text[j] if j < len(text) else ''

                # Closing quote chars: structural JSON (: , } ] or EOF)
                if next_ch in (':', ',', '}', ']', ''):
                    in_string = False
                    out.append(ch)
                else:
                    # Embedded quote; escape it
                    out.append('\\"')
        else:
            out.append(ch)

        i += 1

    return ''.join(out)
