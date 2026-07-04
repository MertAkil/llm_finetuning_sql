import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_utils import get_sql_label, resolve_path, safe_model_id

try:
    from .commons import model_to_hf_name
except ImportError:
    from commons import model_to_hf_name

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Text-to-SQL model with LoRA.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama2", "codellama", "tinyllama"],
        required=True,
        help="Base model alias to fine-tune.",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Retained for experiment parity; training data is controlled by --train-data.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train_data.jsonl",
        help="JSONL training split path. Records may use either 'query' or 'answer'.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/val_data.jsonl",
        help="JSONL validation split path. Records may use either 'query' or 'answer'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train/checkpoints_lora",
        help="Directory where LoRA checkpoints are written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenized prompt length.",
    )
    return parser.parse_args()

def get_train_data(path: str):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(resolve_path(path)))
    return dataset["train"]

def get_eval_data(path: str):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(resolve_path(path)))
    return dataset["train"]


def tokenize(prompt, tokenizer, max_length: int):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point, tokenizer, max_length: int):
    query = get_sql_label(data_point)
    full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
{data_point["question"]}

### Context:
{data_point["context"]}

### Response:
{query}
"""
    return tokenize(full_prompt, tokenizer, max_length=max_length)

def setup_training(model_name, device_map):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name != "NousResearch/Llama-2-7b-hf":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_cache=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_eos_token = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def train(
    model_name,
    device_map,
    train_data: str,
    val_data: str,
    output_dir: str,
    epochs: int,
    max_length: int,
):
    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

    model, tokenizer = setup_training(model_name, device_map)

    train_dataset = get_train_data(train_data)
    eval_dataset = get_eval_data(val_data)
    if model_name == "NousResearch/Llama-2-7b-hf":
        bfloat16 = True
        float16 = False
    else:
        bfloat16 = False
        float16 = True

    tokenized_train_dataset = train_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, max_length=max_length),
        remove_columns=train_dataset.column_names,
    )
    tokenized_val_dataset = eval_dataset.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, max_length=max_length),
        remove_columns=eval_dataset.column_names,
    )

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
    model_output_dir = resolve_path(output_dir) / safe_model_id(model_name)

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
            num_train_epochs=epochs,
            output_dir=str(model_output_dir),
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

    train(
        model_name=base_model,
        device_map=device_map,
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_length=args.max_length,
    )
    print("DONE")

