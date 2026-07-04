import sys
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .commons import DATA_PATH, FEW_SHOT_PATH, RESULTS_DIR
    from .run_inference import run_inference
    from .utils import save_queries
except ImportError:
    from commons import DATA_PATH, FEW_SHOT_PATH, RESULTS_DIR
    from run_inference import run_inference
    from utils import save_queries

def main(model_id, few_shot=False, data_path=DATA_PATH, few_shot_path=FEW_SHOT_PATH, output_dir=RESULTS_DIR):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")

    model.to(device)

    generate_queries = run_inference(
        model,
        tokenizer,
        few_shot,
        data_path=data_path,
        few_shot_path=few_shot_path,
    )

    save_queries(model_id, generate_queries, few_shot, output_dir=output_dir)

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
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DATA_PATH),
        help="JSONL file containing evaluation questions and schemas.",
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
        help="Directory for generated query JSON outputs.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(
        args.model,
        few_shot=args.few_shot,
        data_path=args.data_path,
        few_shot_path=args.few_shot_path,
        output_dir=args.output_dir,
    )
