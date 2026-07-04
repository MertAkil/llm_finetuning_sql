import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_utils import get_sql_label, resolve_path

try:
    from .commons import FEW_SHOT_PATH
    from .templates import create_message
except ImportError:
    from commons import FEW_SHOT_PATH
    from templates import create_message


def few_shot_template(few_shot_path=FEW_SHOT_PATH):
    few_shots = []
    with resolve_path(few_shot_path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data_sample = json.loads(line)
            question = data_sample['question']
            context = data_sample['context']
            query = get_sql_label(data_sample)

            message_template = create_message(question=question, context=context)

            template = [
                {
                    "role": "user",
                    "content": message_template,
                },
                {
                    "role": "assistant",
                    "content": query,
                },
            ]
            few_shots += template
    
    return few_shots
