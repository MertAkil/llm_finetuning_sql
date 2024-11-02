import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
from datetime import datetime
from commons import DATA_PATH, model_to_hf_name
from peft import PeftModel
import os

def get_args():
    parser = argparse.ArgumentParser(description="Model and Few-Shot Argument Parsing")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama2", "codellama", "tinyllama"],  # Available models
        required=True,
        help="Choose a model from the available options: modelA, modelB, modelC"
    )
    
    parser.add_argument(
        "--best_epoch",
        type=int,
        # choices=["llama2", "codellama", "tinyllama"],  # Available models
        required=False,
        help="Choose a model from the available options: modelA, modelB, modelC"
    )
    
    # Add few_shot argument (True/False flag)
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Enable few-shot learning if flag is present"
    )

    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Enable few-shot learning if flag is present"
    )

    return parser.parse_args()

def get_formatted_data(model, few_shot=False):    
    if few_shot:
        from commons import PROMPT_TEMPLATE_FS
        prompt = PROMPT_TEMPLATE_FS
    else:
        from commons import PROMPT_TEMPLATE
        prompt = PROMPT_TEMPLATE

    questions, contexts = get_template_pairs()

    formatted_prompts = [
        prompt.format(question=question, schema=context)
        for question, context in zip(questions, contexts)
    ]

    return formatted_prompts

def get_template_pairs():
    questions = []
    contexts = []
    with open(DATA_PATH, "r") as lines:
        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            questions.append(entry["question"])
            contexts.append(entry["context"])
    
    return questions, contexts

def save_results(results, model_name, few_shot, finetuned):
    current_date = datetime.now().strftime("%d:%m:%Y")

    df = pd.DataFrame(results, columns = ["Results"])
    name = model_name.split("/")[-1]
    if finetuned:
        finetune_flag = True
    else:
        finetune_flag = ""
    df.to_csv(f"../results/{name}_{few_shot}_{finetune_flag}_{current_date}.csv", index=False)

def datas(data):
    for data in datas:
        yield data

def infer_llm(model_name, data, few_shot=False, finetuned=False, best_epoch=-1):
    print(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if finetuned:
        checkpoint_files = os.listdir(f"../train/checkpoints_lora/{model_name}")
        best_checkpoint = checkpoint_files[best_epoch]
        print(f"best_checkpoint is considered to be: {best_checkpoint}")
        output_dir = f"../train/checkpoints_lora/{model_name}/{best_checkpoint}"

        model = PeftModel.from_pretrained(model, output_dir)


    print("HERE")
    pipe = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer,
        torch_dtype=torch.float16
        )

    generated_queries = []
    
    ctr = 0 
    for prompt in data:
        ans = pipe(prompt, max_new_tokens=100)
        # generated_queries.append(ans[0]['generated_text'])
        generated_queries.append(ans[0]['generated_text'].split('```')[2])
        print(ctr)
        print(ans[0]['generated_text'].split('```')[2])
        # print(ans[0]['generated_text'])
        print("-"*30)
        ctr +=1

    save_results(generated_queries, model_name, few_shot, finetuned)
    print(model.get_memory_footprint())
    return generated_queries


if __name__=="__main__":
    args = get_args()
    data = get_formatted_data(model=args.model, few_shot=args.few_shot)
    # print(data[1])
    # bla = bla
    model_name = model_to_hf_name[args.model]
    print(model_name)
    results = infer_llm(model_name=model_name, data=data, few_shot=args.few_shot, finetuned=args.finetuned, best_epoch=args.best_epoch)
    print("DONE")
    