import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_utils import resolve_path, safe_model_id

try:
    from .commons import RESULTS_DIR
except ImportError:
    from commons import RESULTS_DIR


def save_queries(model_id, generated_queries, few_shot, output_dir=RESULTS_DIR):
    output_path = resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / f"query_{safe_model_id(model_id)}_few_shot={few_shot}.json"
    
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(generated_queries, f)

    print(f"Queries saved to {file_path}")
    return file_path


def extract_sql_query(text):
    if not text:
        return None

    code_block_pattern = re.compile(r"```(?:sql|mysql|sqlite)?\s*([\s\S]*?)```", re.IGNORECASE)
    for match in code_block_pattern.finditer(text):
        query = match.group(1).strip()
        if query:
            return query

    query_pattern = re.compile(r"\b(?:SELECT|WITH|INSERT|UPDATE|DELETE)\b[\s\S]*?(?:;|$)", re.IGNORECASE)
    match = query_pattern.search(text)
    if match:
        return match.group(0).strip()

    return None
