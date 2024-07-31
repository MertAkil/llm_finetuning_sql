import torch
from prompts import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from run_inference import run_inference
from utils import save_queries

def main(model_id, few_shot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")

    model.to(device)

    generate_queries = run_inference(model, tokenizer, few_shot)

    save_queries(model_id, generate_queries, few_shot)

    return generate_queries

def parse_args():
    parser = argparse.ArgumentParser(description="Script for running a model with optional few-shot learning.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model identifier to use for inference (e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')"  
    )

    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Flag indicating if few-shot learning should be used"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args.model, args.few_shot)