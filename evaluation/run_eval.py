import argparse
import os
from typing import Optional

try:
    from .utils import load_generated_queries, load_true_queries
    from .evaluate import evaluate_generated_queries
    from .data_class import export_to_csv
except ImportError:
    from utils import load_generated_queries, load_true_queries
    from evaluate import evaluate_generated_queries
    from data_class import export_to_csv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated Text-to-SQL queries.")
    parser.add_argument(
        "--generated-path",
        "--file_path",
        dest="generated_path",
        type=str,
        required=True,
        help="CSV, JSON, or JSONL file with generated queries.",
    )
    parser.add_argument(
        "--truth-path",
        type=str,
        default="data/test_data.jsonl",
        help="JSON or JSONL file containing reference questions, schemas, and SQL labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where evaluation stats/metrics folders are written.",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Skip MySQL execution and compute exact/string metrics only.",
    )
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "3306")))
    parser.add_argument("--db-user", default=os.getenv("DB_USER"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD"))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "master_dummy"))

    return parser.parse_args()


def run_evaluation(
    path_to_generated_queries: str,
    truth_path: str = "data/test_data.jsonl",
    output_dir: str = "results",
    execute: bool = True,
    db_config: Optional[dict] = None,
) -> list:
    generated_queries = load_generated_queries(file_path=path_to_generated_queries)
    true_queries = load_true_queries(file_path=truth_path)

    res = evaluate_generated_queries(
        generated_queries=generated_queries,
        true_queries=true_queries,
        execute=execute,
        db_config=db_config,
    )
    stats_path, metrics_path = export_to_csv(res, path_to_generated_queries, output_dir=output_dir)
    print(f"Saved stats to {stats_path}")
    print(f"Saved metrics to {metrics_path}")
    return res

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        args.generated_path,
        truth_path=args.truth_path,
        output_dir=args.output_dir,
        execute=not args.no_exec,
        db_config={
            "host": args.db_host,
            "port": args.db_port,
            "user": args.db_user,
            "password": args.db_password,
            "database": args.db_name,
        },
    )
