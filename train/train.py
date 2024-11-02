import argparse
from datetime import datetime
import os
import sys

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import transformers

from datasets import load_dataset

from commons import model_to_hf_name

def parse_args():
    parser = argparse.ArgumentParser(description="Model and Few-Shot Argument Parsing")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama2", "codellama", "tinyllama"],  # Available models
        required=True,
        help="Choose a model from the available options: modelA, modelB, modelC"
    )
    
    # Add few_shot argument (True/False flag)
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Enable few-shot learning if flag is present"
    )
    return parser.parse_args()

def get_train_data():
    dataset = load_dataset("json", data_files = "./../data/train_data.jsonl")
    return dataset["train"]

def get_eval_data():
    dataset = load_dataset("json", data_files = "./../data/val_data.jsonl")
    return dataset["train"]


def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{data_point["answer"]}
"""
    return tokenize(full_prompt, tokenizer)

def setup_training(model_name, device_map):
    if model_name != "NousResearch/Llama-2-7b-hf":
        type=torch.float16
    else: 
        type = torch.bfloat16
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=type,
        device_map=device_map,
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer


def train(model_name, device_map):
    model, tokenizer = setup_training(model_name, device_map)

    train_dataset, eval_dataset = get_train_data(), get_eval_data()
    if model_name == "NousResearch/Llama-2-7b-hf":
        bfloat16 = True
        float16 = False
    else:
        bfloat16 = False
        float16 = True

    tokenized_train_dataset = train_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
    tokenized_val_dataset = eval_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))

    model.train()

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    output_dir = f"checkpoints_lora/{model_name}"

    training_args = TrainingArguments(
            auto_find_batch_size=True,
            warmup_steps=100,
            gradient_accumulation_steps=2,
            learning_rate=3e-4,
            fp16=float16,
            bf16=bfloat16,
            optim="adamw_torch",
            evaluation_strategy="epoch", # if val_set_size > 0 else "no",
            save_strategy="epoch",
            num_train_epochs=8,
            output_dir=output_dir,
            load_best_model_at_end=False,
            group_by_length=True, # group sequences of roughly the same length together to speed up training
        )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    tokenizer.pad_token = tokenizer.eos_token

    trainer.train()


if __name__=="__main__":
    args = parse_args()
    device_map = {"": 0}
    base_model = model_to_hf_name[args.model]
    
    train(base_model, device_map)
    print("DONE")

