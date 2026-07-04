import argparse
from datetime import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_utils import get_sql_label, load_jsonl_records, resolve_path, safe_model_id

try:
    from .commons import CHECKPOINT_DIR, DATA_PATH, FEW_SHOT_PATH, RESULTS_DIR, model_to_hf_name
    from .utils import extract_sql_query
except ImportError:
    from commons import CHECKPOINT_DIR, DATA_PATH, FEW_SHOT_PATH, RESULTS_DIR, model_to_hf_name
    from utils import extract_sql_query

def get_args():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL inference.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama2", "codellama", "tinyllama"],
        required=True,
        help="Base model alias to run.",
    )
    parser.add_argument(
        "--best-epoch",
        "--best_epoch",
        type=int,
        required=False,
        default=-1,
        help="Checkpoint index from the sorted checkpoint directory. Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Use the few-shot prompt template.",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Load a LoRA adapter from --checkpoint-dir.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PATH),
        help="JSONL file containing questions and schemas.",
    )
    parser.add_argument(
        "--few-shot-path",
        type=str,
        default=str(FEW_SHOT_PATH),
        help="JSONL file containing few-shot examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory where generated query CSV files are written.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINT_DIR),
        help="Directory containing LoRA checkpoints.",
    )

    return parser.parse_args()

def _format_few_shot_examples(few_shot_path=FEW_SHOT_PATH):
    examples = []
    for idx, entry in enumerate(load_jsonl_records(few_shot_path), start=1):
        examples.append(
            "\n".join(
                [
                    f"Example {idx}:",
                    f"Question: {entry['question']}",
                    f"Schema: {entry['context']}",
                    f"SQL query: {get_sql_label(entry)}",
                ]
            )
        )
    return "\n\n".join(examples)


def get_formatted_data(model, few_shot=False, data_path=DATA_PATH, few_shot_path=FEW_SHOT_PATH):
    if few_shot:
        examples = _format_few_shot_examples(few_shot_path=few_shot_path)
        prompt = (
            "[INST] Write a SQL query to answer the following question given the database schema. "
            "Please wrap your code answer using ```.\n\n"
            "{examples}\n\n"
            "Schema: {schema}\nQuestion: {question}[/INST] SQL query: ``` "
        )
    else:
        try:
            from .commons import PROMPT_TEMPLATE as prompt
        except ImportError:
            from commons import PROMPT_TEMPLATE as prompt

    questions, contexts = get_template_pairs(data_path=data_path)

    formatted_prompts = [
        prompt.format(question=question, schema=context, examples=examples if few_shot else "")
        for question, context in zip(questions, contexts)
    ]

    return formatted_prompts

def get_template_pairs(data_path=DATA_PATH):
    records = load_jsonl_records(data_path)
    questions = [entry["question"] for entry in records]
    contexts = [entry["context"] for entry in records]

    return questions, contexts

def save_results(results, model_name, few_shot, finetuned, output_dir=RESULTS_DIR):
    import pandas as pd

    current_date = datetime.now().strftime("%Y-%m-%d")

    output_path = resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results, columns=["Results"])
    finetune_flag = "finetuned" if finetuned else "base"
    file_path = output_path / f"{safe_model_id(model_name)}_{few_shot}_{finetune_flag}_{current_date}.csv"
    df.to_csv(file_path, index=False)
    return file_path

def _checkpoint_parent(checkpoint_dir, model_name):
    checkpoint_root = resolve_path(checkpoint_dir)
    candidates = [
        checkpoint_root / safe_model_id(model_name),
        checkpoint_root / model_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find LoRA checkpoint directory for "
        f"{model_name}. Checked: {', '.join(str(path) for path in candidates)}"
    )


def _select_checkpoint(checkpoint_dir, model_name, best_epoch=-1):
    checkpoint_parent = _checkpoint_parent(checkpoint_dir, model_name)
    checkpoints = sorted(path for path in checkpoint_parent.iterdir() if path.is_dir())
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_parent}")
    if best_epoch == -1:
        return checkpoints[-1]
    try:
        return checkpoints[best_epoch]
    except IndexError as exc:
        raise IndexError(
            f"Checkpoint index {best_epoch} is out of range for {checkpoint_parent}."
        ) from exc

def infer_llm(
    model_name,
    data,
    few_shot=False,
    finetuned=False,
    best_epoch=-1,
    checkpoint_dir=CHECKPOINT_DIR,
    output_dir=RESULTS_DIR,
):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if finetuned:
        adapter_path = _select_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            best_epoch=best_epoch,
        )
        model = PeftModel.from_pretrained(model, str(adapter_path))


    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16
        )

    generated_queries = []

    ctr = 0
    for prompt in data:
        ans = pipe(prompt, max_new_tokens=100, do_sample=False, return_full_text=False)
        generated_text = ans[0]["generated_text"]
        generated_queries.append(extract_sql_query(generated_text) or generated_text.strip())
        if ctr % 50 == 0:
            print(f"Processed {ctr} prompts")
        ctr +=1

    file_path = save_results(generated_queries, model_name, few_shot, finetuned, output_dir=output_dir)
    print(f"Saved generated queries to {file_path}")
    return generated_queries


if __name__=="__main__":
    args = get_args()
    data = get_formatted_data(
        model=args.model,
        few_shot=args.few_shot,
        data_path=args.data_path,
        few_shot_path=args.few_shot_path,
    )
    model_name = model_to_hf_name[args.model]
    results = infer_llm(
        model_name=model_name,
        data=data,
        few_shot=args.few_shot,
        finetuned=args.finetuned,
        best_epoch=args.best_epoch,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )
    print("DONE")

