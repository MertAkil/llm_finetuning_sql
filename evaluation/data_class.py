import csv
import os
from dataclasses import dataclass, asdict
from utils import check_for_join, check_for_tables

@dataclass
class Metrics:
    generated_query: str
    true_query: str
    exact_match: bool
    similarity_measure: float
    has_join: bool
    has_multiple_tables: bool
    exec_gen_query: bool

def update_metrics(
    generated_query: str,
    true_query: str,
    true_context: str,
    exact_flag: bool,
    similarity_measure: float,
    exec_gen_query: bool
) -> Metrics:
    has_join = check_for_join(true_query)
    has_multiple_tables = check_for_tables(true_context)

    metric = Metrics(
        generated_query = generated_query,
        true_query = true_query,
        exact_match = exact_flag,
        similarity_measure = similarity_measure,
        has_join = has_join,
        has_multiple_tables = has_multiple_tables,
        exec_gen_query = exec_gen_query
    )

    return metric

def calculate_metrics(comparisons: list[Metrics]) -> dict:
    total = len(comparisons)
    exact_match_count = sum(c.exact_match for c in comparisons)
    total_similarity = sum(c.similarity_measure for c in comparisons)
    
    join_comparisons = [c for c in comparisons if c.has_join]
    join_exact_match_count = sum(c.exact_match for c in join_comparisons)
    join_total_similarity = sum(c.similarity_measure for c in join_comparisons)
    
    mult_table_comparisons = [c for c in comparisons if c.has_multiple_tables]
    mult_table_exact_match_count = sum(c.exact_match for c in mult_table_comparisons)
    mult_table_total_similarity = sum(c.similarity_measure for c in mult_table_comparisons)
    
    single_table_comparisons = [c for c in comparisons if not c.has_multiple_tables]
    single_table_exact_match_count = sum(c.exact_match for c in single_table_comparisons)
    single_table_total_similarity = sum(c.similarity_measure for c in single_table_comparisons)
    
    metrics = {
        'overall_accuracy': exact_match_count / total if total > 0 else 0,
        'overall_similarity': total_similarity / total if total > 0 else 0,
        'join_accuracy': join_exact_match_count / len(join_comparisons) if join_comparisons else 0,
        'join_similarity': join_total_similarity / len(join_comparisons) if join_comparisons else 0,
        'mult_table_accuracy': mult_table_exact_match_count / len(mult_table_comparisons) if mult_table_comparisons else 0,
        'mult_table_similarity': mult_table_total_similarity / len(mult_table_comparisons) if mult_table_comparisons else 0,
        'single_table_accuracy': single_table_exact_match_count / len(single_table_comparisons) if single_table_comparisons else 0,
        'single_table_similarity': single_table_total_similarity / len(single_table_comparisons) if single_table_comparisons else 0,
    }
    return metrics

def export_to_csv(
        comparisons: list[Metrics], 
        base_file_path: str
) -> None:
    metrics = calculate_metrics(comparisons)

    # Extract directory and base file name from the provided file path
    base_dir = os.path.dirname(base_file_path)
    base_name = os.path.basename(base_file_path).split('.')[0]

    # Define paths for the two CSV files
    stats_csv_path = os.path.join(base_dir, f"{base_name}_stats.csv")
    metrics_csv_path = os.path.join(base_dir, f"{base_name}_metrics.csv")

    # Write the query generation statistics to the first CSV
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [field.name for field in Metrics.__dataclass_fields__.values()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(asdict(comparison))

    # Write the overall metrics to the second CSV
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metrics'])
        for key, value in metrics.items():
            writer.writerow([key, value])
