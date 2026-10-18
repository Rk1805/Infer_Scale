"""Load and validate request data from various sources."""

import csv
import json
import os
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List

from inferscale.schemas.request import RequestSchema


def load_from_csv(filepath: str) -> Iterator[Dict[str, Any]]:
    """Load requests from a CSV file."""
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_from_json(filepath: str) -> Iterator[Dict[str, Any]]:
    """Load requests from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
        for item in data:
            yield item


def validate_request(row: Dict[str, Any]) -> bool:
    """Validate a single request record."""
    try:
        req = RequestSchema(**row)
        return True
    except Exception:
        return False


def filter_by_source(filepath: str, source: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Filter requests by source if specified."""
    for row in load_from_csv(filepath):
        if source is None or row.get("source") == source:
            yield row


def load_requests(filepath: str) -> List[RequestSchema]:
    """Load and validate requests from a CSV or JSON file."""
    ext = Path(filepath).suffix.lower()
    rows = []
    if ext == ".csv":
        rows = list(load_from_csv(filepath))
    elif ext == ".json":
        rows = list(load_from_json(filepath))
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return [RequestSchema(**r) for r in rows]
