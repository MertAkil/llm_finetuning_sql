import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_utils import resolve_path

try:
    from .commons import DATA_PATH, FEW_SHOT_PATH
    from .templates import create_message, create_system_template, create_template
    from .prompts import SYSTEM_PROMPT
    from .few_shot_script import few_shot_template
    from .utils import extract_sql_query
except ImportError:
    from commons import DATA_PATH, FEW_SHOT_PATH
    from templates import create_message, create_system_template, create_template
    from prompts import SYSTEM_PROMPT
    from few_shot_script import few_shot_template
    from utils import extract_sql_query


def run_inference(model, tokenizer, few_shot, data_path=DATA_PATH, few_shot_path=FEW_SHOT_PATH):
    answer_list = []
 
    with resolve_path(data_path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data_sample = json.loads(line)
            
            question = data_sample['question']
            context = data_sample['context']

            user_message = create_message(question, context)
            system_template = create_system_template(SYSTEM_PROMPT)
            user_template = create_template(SYSTEM_PROMPT, user_message)
            
            if few_shot:
                few_shots = few_shot_template(few_shot_path=few_shot_path)
                message_template = system_template + few_shots + user_template

            else:
                message_template = system_template + user_template

            tokenized_chat = tokenizer.apply_chat_template(message_template, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

            outputs = model.generate(
                tokenized_chat, 
                max_new_tokens=128,
                temperature=0.,
                top_p=0.95
            )

            output_string = tokenizer.decode(outputs[0][tokenized_chat.shape[-1]:], skip_special_tokens=True)

            generated_sql = extract_sql_query(output_string)
            answer_list.append(generated_sql)
    
    return answer_list
