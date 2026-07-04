import csv

from evaluation.data_class import Metrics, calculate_metrics, export_to_csv
from evaluation.evaluate import evaluate_generated_queries


def test_calculate_metrics_handles_empty_input():
    metrics = calculate_metrics([])

    assert metrics["overall_exec"] == 0
    assert metrics["overall_accuracy"] == 0


def test_calculate_metrics_keeps_query_and_string_similarity_separate():
    comparisons = [
        Metrics("select 1", "select 1", True, 1.0, 1.0, False, False, True),
        Metrics("select a", "select b", False, 0.25, 0.75, True, True, False),
    ]

    metrics = calculate_metrics(comparisons)

    assert metrics["overall_query_similarity"] == 0.625
    assert metrics["overall_string_similarity"] == 0.875
    assert metrics["join_query_similarity"] == 0.25
    assert metrics["join_string_similarity"] == 0.75


def test_export_to_csv_writes_cross_platform_paths(tmp_path):
    comparisons = [
        Metrics("select 1", "select 1", True, 1.0, 1.0, False, False, True),
    ]

    stats_path, metrics_path = export_to_csv(
        comparisons,
        base_file_path="generated.csv",
        output_dir=str(tmp_path),
    )

    assert stats_path == tmp_path / "generated" / "stats.csv"
    assert metrics_path == tmp_path / "generated" / "metrics.csv"
    assert stats_path.exists()
    assert metrics_path.exists()

    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["metric", "value"]


def test_evaluate_without_database_does_not_mark_queries_executed():
    generated = ["select a from t", "select b from t"]
    truth = [
        {"query": "select a from t", "context": "create table t (a int)"},
        {"query": "select c from t", "context": "create table t (c int)"},
    ]

    metrics = evaluate_generated_queries(generated, truth, execute=False)

    assert metrics[0].exact_match is True
    assert metrics[0].exec_gen_query is False
    assert metrics[1].exact_match is False
    assert metrics[1].string_similarity_measure > 0
