from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union

REPO_ROOT = Path(__file__).resolve().parent


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve project-relative paths while preserving absolute paths."""
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def get_sql_label(record: dict[str, Any]) -> str:
    """Return the SQL target from either supported dataset schema."""
    for key in ("query", "answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Expected a non-empty 'query' or 'answer' field.")


def load_jsonl_records(path: Union[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with resolve_path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def safe_model_id(model_id: str) -> str:
    """Make a model identifier safe to use as one path segment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
