# Fine-Tuning Open-Source LLMs for Text-to-SQL

This repository contains the code and curated data artifacts for my master's thesis on improving open-source language models for translating natural-language questions into executable SQL queries.

I fine-tuned and evaluated Llama 2, Code Llama, and TinyLlama across zero-shot and few-shot settings. The thesis found that fine-tuning substantially improved Text-to-SQL performance, especially without in-context examples, and that general-purpose language models could compete with or outperform code-pretrained models after task-specific fine-tuning. Few-shot prompting did not consistently improve accuracy in this setup.

## Repository Contents

- `data/`: cleaned train, validation, test, and few-shot splits in JSON and JSONL format.
- `dataset_utils/`: intermediate dataset-cleaning artifacts used to build the released Text-to-SQL dataset.
- `train/`: LoRA fine-tuning entry point for Llama 2, Code Llama, and TinyLlama.
- `inference/`: prompt construction and generation scripts for base and fine-tuned models.
- `evaluation/`: exact-match, string-similarity, and execution-based evaluation utilities.
- `results/` and `inference/infer_results/`: representative thesis-era outputs and reference evaluation artifacts.

## Setup

Use Python 3.9+ in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

For GPU runs that use 8-bit model loading, install the optional CUDA-oriented dependency:

```bash
pip install -r requirements-gpu.txt
```

Execution-based evaluation expects a MySQL database populated with the thesis dummy tables. Copy `.env.example` to `.env` or export the same variables in your shell.

## Data Format

The normalized JSONL splits use `question`, `context`, and `query`:

```json
{"question": "What DVD season/name for region 2 was released August 22, 2010?", "context": "create table ...", "query": "select dvd_name from ..."}
```

Some legacy JSON artifacts use `answer` instead of `query`. The training, inference, and evaluation utilities now support both field names, while new code treats `query` as the canonical SQL label.

## Usage

Fine-tune a model with LoRA:

```bash
python -m train.train \
  --model codellama \
  --train-data data/train_data.jsonl \
  --val-data data/val_data.jsonl \
  --output-dir train/checkpoints_lora \
  --epochs 8 \
  --max-length 512
```

Run inference with a base model:

```bash
python -m inference.inference \
  --model codellama \
  --data-path data/test_data.jsonl \
  --output-dir results
```

Run inference with a fine-tuned LoRA checkpoint:

```bash
python -m inference.inference \
  --model codellama \
  --finetuned \
  --checkpoint-dir train/checkpoints_lora \
  --data-path data/test_data.jsonl \
  --output-dir results
```

Evaluate without a database, useful for quick local checks:

```bash
python -m evaluation.run_eval \
  --generated-path results/example_generation.csv \
  --truth-path data/test_data.jsonl \
  --no-exec
```

Evaluate with MySQL execution:

```bash
python -m evaluation.run_eval \
  --generated-path results/example_generation.csv \
  --truth-path data/test_data.jsonl \
  --db-host "$DB_HOST" \
  --db-port "$DB_PORT" \
  --db-user "$DB_USER" \
  --db-password "$DB_PASSWORD" \
  --db-name "$DB_NAME"
```

## Evaluation Notes

The evaluation pipeline reports:

- exact query match after lightweight normalization,
- token-overlap string similarity,
- result-set similarity for executable generated queries,
- subgroup metrics for joins, multiple-table schemas, and single-table schemas.

The stored outputs are preserved as research artifacts. Some files are legacy experiment exports and may include raw generations, prompt leakage, or date-stamped filenames from thesis runs; see the directory-level notes for details.

## Reproducibility

Full training and inference require GPU hardware and access to the relevant Hugging Face model weights. The unit tests avoid model downloads and database access:

```bash
pytest
```

Lightweight syntax checks:

```bash
python -m compileall train inference evaluation project_utils.py
```

## Released Assets

- Fine-tuned models: TODO add Hugging Face model links.
- Cleaned Text-to-SQL dataset: TODO add dataset link.
- Thesis PDF: TODO add thesis/publication link.
- Personal website: TODO add project page link.

## License

TODO choose and document a license before treating this repository as reusable open-source software. Until then, the repository should be considered a public research/portfolio artifact rather than a formally licensed package.
