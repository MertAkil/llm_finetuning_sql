import json

import pytest

from project_utils import get_sql_label, load_jsonl_records, resolve_path, safe_model_id


def test_get_sql_label_prefers_query_and_supports_answer():
    assert get_sql_label({"query": "select 1", "answer": "select 2"}) == "select 1"
    assert get_sql_label({"answer": "select 2"}) == "select 2"


def test_get_sql_label_rejects_missing_label():
    with pytest.raises(KeyError):
        get_sql_label({"question": "missing"})


def test_load_jsonl_records_resolves_project_relative_paths(tmp_path):
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(json.dumps({"query": "select 1"}) + "\n", encoding="utf-8")

    assert load_jsonl_records(jsonl) == [{"query": "select 1"}]
    assert resolve_path("data/test_data.jsonl").is_absolute()


def test_safe_model_id_is_single_path_segment():
    assert safe_model_id("codellama/CodeLlama-7b-instruct-hf") == "codellama_CodeLlama-7b-instruct-hf"
