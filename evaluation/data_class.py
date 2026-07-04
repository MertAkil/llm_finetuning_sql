import csv
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_utils import resolve_path

try:
    from .utils import check_for_join, check_for_tables
except ImportError:
    from utils import check_for_join, check_for_tables

@dataclass
class Metrics:
    generated_query: str
    true_query: str
    exact_match: bool
    query_similarity_measure: float
    string_similarity_measure: float
    has_join: bool
    has_multiple_tables: bool
    exec_gen_query: bool

def update_metrics(
    generated_query: str,
    true_query: str,
    true_context: str,
    exact_flag: bool,
    query_similarity_measure: float,
    string_similarity_measure: float,
    exec_gen_query: bool
) -> Metrics:
    has_join = check_for_join(true_query)
    has_multiple_tables = check_for_tables(true_context)

    metric = Metrics(
        generated_query = generated_query,
        true_query = true_query,
        exact_match = exact_flag,
        query_similarity_measure = query_similarity_measure,
        string_similarity_measure = string_similarity_measure,
        has_join = has_join,
        has_multiple_tables = has_multiple_tables,
        exec_gen_query = exec_gen_query
    )

    return metric

def calculate_metrics(comparisons: list[Metrics]) -> dict:
    total = len(comparisons)
    exact_match_count = sum(c.exact_match for c in comparisons)
    query_total_similarity = sum(c.query_similarity_measure for c in comparisons)
    string_total_similarity = sum(c.string_similarity_measure for c in comparisons)
    
    join_comparisons = [c for c in comparisons if c.has_join]
    join_exact_match_count = sum(c.exact_match for c in join_comparisons)
    join_query_total_similarity = sum(c.query_similarity_measure for c in join_comparisons)
    join_string_total_similarity = sum(c.string_similarity_measure for c in join_comparisons)

    mult_table_comparisons = [c for c in comparisons if c.has_multiple_tables]
    mult_table_exact_match_count = sum(c.exact_match for c in mult_table_comparisons)
    mult_table_query_total_similarity = sum(c.query_similarity_measure for c in mult_table_comparisons)
    mult_table_string_total_similarity = sum(c.string_similarity_measure for c in mult_table_comparisons)

    single_table_comparisons = [c for c in comparisons if not c.has_multiple_tables]
    single_table_exact_match_count = sum(c.exact_match for c in single_table_comparisons)
    single_table_query_total_similarity = sum(c.query_similarity_measure for c in single_table_comparisons)
    single_table_string_total_similarity = sum(c.string_similarity_measure for c in single_table_comparisons)

    executable = sum(c.exec_gen_query for c in comparisons)
    def safe_average(value, denominator):
        return value / denominator if denominator else 0

    metrics = {
        'overall_exec': safe_average(executable, total),
        'overall_accuracy': safe_average(exact_match_count, total),
        'overall_query_similarity': safe_average(query_total_similarity, total),
        'overall_string_similarity': safe_average(string_total_similarity, total),
        'join_accuracy': safe_average(join_exact_match_count, len(join_comparisons)),
        'join_query_similarity': safe_average(join_query_total_similarity, len(join_comparisons)),
        'join_string_similarity': safe_average(join_string_total_similarity, len(join_comparisons)),
        'mult_table_accuracy': safe_average(mult_table_exact_match_count, len(mult_table_comparisons)),
        'mult_table_query_similarity': safe_average(mult_table_query_total_similarity, len(mult_table_comparisons)),
        'mult_table_string_similarity': safe_average(mult_table_string_total_similarity, len(mult_table_comparisons)),
        'single_table_accuracy': safe_average(single_table_exact_match_count, len(single_table_comparisons)),
        'single_table_query_similarity': safe_average(single_table_query_total_similarity, len(single_table_comparisons)),
        'single_table_string_similarity': safe_average(single_table_string_total_similarity, len(single_table_comparisons)),
    }
    return metrics

def export_to_csv(
        comparisons: list[Metrics],
        base_file_path: str,
        output_dir: str = "results",
) -> tuple[Path, Path]:

    base_file = Path(base_file_path).stem
    dest_path = resolve_path(output_dir) / base_file

    dest_path.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(comparisons)

    stats_csv_path = dest_path / "stats.csv"
    metrics_csv_path = dest_path / "metrics.csv"

    with stats_csv_path.open('w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [field.name for field in Metrics.__dataclass_fields__.values()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(asdict(comparison))

    with metrics_csv_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    return stats_csv_path, metrics_csv_path
