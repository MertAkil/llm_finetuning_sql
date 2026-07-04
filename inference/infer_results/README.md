# Legacy Inference Outputs

This directory contains preserved thesis-era model generations.

The files are useful for auditing experiment history, but they are raw outputs: some generations include explanatory text, prompt leakage, or malformed SQL from base-model and few-shot runs. New inference runs should generally be written to `results/` with the current CLI:

```bash
python -m inference.inference --model codellama --data-path data/test_data.jsonl --output-dir results
```
