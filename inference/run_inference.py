import json
from templates import create_message, create_system_template, create_template
from prompts import SYSTEM_PROMPT
from few_shot_script import few_shot_template
from utils import extract_sql_query

def run_inference(model, tokenizer, few_shot):
    answer_list = []
 
    file_path = 'data.jsonl'

    # Open the file and read line-by-line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON object from the line
            data_sample = json.loads(line)
            
            # Accessing the fields of the sample
            question = data_sample['question']
            context = data_sample['context']
            query = data_sample['query']

            USER_TEMPLATE = create_message(question, context)
            system_template = create_system_template(SYSTEM_PROMPT)
            user_template = create_template(SYSTEM_PROMPT, USER_TEMPLATE)

            message_template = None
            
            if few_shot == True:
                few_shots = few_shot_template()
                print(few_shots)
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
