"""
JSON serialization utilities for handling non-standard types.
"""

import json
from pathlib import Path
from decimal import Decimal


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for non-standard types."""

    def default(self, obj):
        """Handle non-standard types."""
        # numpy types
        if hasattr(obj, "item"):
            return obj.item()
        
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
    # None, bool, int, float, str are already JSON-serializable
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    
    # numpy/pandas types
    if hasattr(obj, "item"):
        return obj.item()
    
    # numpy arrays and similar
    if hasattr(obj, "tolist"):
        return obj.tolist()
    
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
