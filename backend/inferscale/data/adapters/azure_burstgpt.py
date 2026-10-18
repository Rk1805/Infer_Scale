"""Data adapters for Azure LLM Inference traces."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from inferscale.data.adapters.base import DataAdapter
from inferscale.schemas.request import RequestSchema


class AzureLLMAdapter(DataAdapter):
    """Adapter for Azure LLM Inference Dataset (2024)."""

    @property
    def source_name(self) -> str:
        return "azurellm"

    def load(self, filepath: str) -> List[dict]:
        """Load Azure LLM inference trace CSV."""
        rows = []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def validate(self, row: dict) -> bool:
        """Validate a single Azure trace row."""
        required = ["request_id", "timestamp", "prompt_tokens", "completion_tokens"]
        return all(field in row and row[field] != "" for field in required)

    def transform(self, row: dict) -> Optional[RequestSchema]:
        """Transform Azure row to RequestSchema."""
        try:
            # Parse timestamp (ISO format or Unix timestamp)
            ts_str = row.get("timestamp", "")
            if "T" in ts_str:
                arrival_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
            else:
                arrival_time = float(ts_str)
            
            return RequestSchema(
                id=str(row.get("request_id", "")),
                arrival_time_seconds=arrival_time,
                prompt_tokens=int(row.get("prompt_tokens", 0)),
                output_tokens=int(row.get("completion_tokens", 0)),
                source=self.source_name,
            )
        except Exception:
            return None


class BurstGPTAdapter(DataAdapter):
    """Adapter for BurstGPT traces."""

    @property
    def source_name(self) -> str:
        return "burstgpt"

    def load(self, filepath: str) -> List[dict]:
        """Load BurstGPT trace CSV."""
        rows = []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def validate(self, row: dict) -> bool:
        """Validate a single BurstGPT trace row."""
        required = ["request_id", "arrival_time", "prompt_tokens", "completion_tokens"]
        return all(field in row and row[field] != "" for field in required)

    def transform(self, row: dict) -> Optional[RequestSchema]:
        """Transform BurstGPT row to RequestSchema."""
        try:
            arrival_time = float(row.get("arrival_time", 0))
            
            return RequestSchema(
                id=str(row.get("request_id", "")),
                arrival_time_seconds=arrival_time,
                prompt_tokens=int(row.get("prompt_tokens", 0)),
                output_tokens=int(row.get("completion_tokens", 0)),
                source=self.source_name,
            )
        except Exception:
            return None


# Registry of available adapters
ADAPTERS = {
    "azurellm": AzureLLMAdapter(),
    "burstgpt": BurstGPTAdapter(),
}


def get_adapter(source: str) -> Optional[DataAdapter]:
    """Get adapter by source name."""
    return ADAPTERS.get(source.lower())


def load_dataset(filepath: str, source: str) -> List[RequestSchema]:
    """Load and transform a dataset using the appropriate adapter."""
    adapter = get_adapter(source)
    if not adapter:
        raise ValueError(f"Unknown data source: {source}")
    
    raw_rows = adapter.load(filepath)
    requests = []
    
    for row in raw_rows:
        if adapter.validate(row):
            req = adapter.transform(row)
            if req:
                requests.append(req)
    
    # Sort by arrival time
    requests.sort(key=lambda r: r.arrival_time_seconds)
    return requests