import json
from templates import create_message

def few_shot_template():
    few_shots = []
    TEMPLATE = []
    with open("/home/ec2-user/efs/mert/Experiments/few_shot.jsonl", 'r') as file:
        for line in file:
            # print(line)
            # Parse the JSON object from the line
            data_sample = json.loads(line)
            
            # Accessing the fields of the sample
            question = data_sample['question']
            context = data_sample['context']
            query = data_sample['query']

            message_template = create_message(question=question, context=context)


            TEMPLATE = [
                # {
                #     "role": "system",
                #     "content": SYSTEM_PROMPT,
                # },
                {
                    "role": "user",
                    "content": message_template,
                },
                {
                    "role": "assistant",
                    "content": query,
                },
            ]
            few_shots += TEMPLATE
    
    return few_shots
