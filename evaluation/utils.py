import re
import json
from typing import Any, Optional
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_utils import get_sql_label, resolve_path


def connect_db(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    port: Optional[int] = None,
) -> Any:
    import mysql.connector

    connection = mysql.connector.connect(
        host=host or os.getenv("DB_HOST", "localhost"),
        port=port or int(os.getenv("DB_PORT", "3306")),
        user=user or os.getenv("DB_USER"),
        password=password if password is not None else os.getenv("DB_PASSWORD"),
        database=database or os.getenv("DB_NAME", "master_dummy"),
    )

    return connection


def load_generated_queries(file_path: str) -> list[str]:
    path = resolve_path(file_path)
    if path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        column = "Results" if "Results" in df.columns else df.columns[0]
        return ["" if pd.isna(value) else str(value) for value in df[column]]

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            return [_extract_generated_query(json.loads(line)) for line in stream if line.strip()]

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if isinstance(payload, list):
        return [_extract_generated_query(entry) for entry in payload]
    raise ValueError(f"Expected {file_path} to contain a list of generated queries.")


def _extract_generated_query(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("Results", "generated_query", "result", "query", "answer"):
            value = entry.get(key)
            if isinstance(value, str):
                return value
    raise ValueError(f"Could not extract generated query from entry: {entry!r}")


def load_true_queries(file_path: str = "data/test_data.jsonl") -> list[dict[str, str]]:
    path = resolve_path(file_path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            payload = [json.loads(line) for line in stream if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)

    return [
        {
            "question": entry["question"],
            "context": entry["context"],
            "query": get_sql_label(entry),
        }
        for entry in payload
    ]

def change_sql_mode(cursor: Any) -> None:
    change_full_group_sql = "SET SESSION sql_mode=(SELECT REPLACE(@@SESSION.sql_mode,'ONLY_FULL_GROUP_BY',''));"
    cursor.execute(change_full_group_sql)



def check_for_join(query: str) -> bool:
    if not query:
        return False
    join_pattern = re.compile(r'\b(JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN)\b', re.IGNORECASE)
    if join_pattern.search(query):
        return True
    return False

def check_for_tables(context:str) -> bool:
    if not context:
        return False
    # Use a regular expression to find all standalone occurrences of 'create table'
    create_table_count = len(re.findall(r'\bcreate table\b', context, re.IGNORECASE))
    return create_table_count > 1

