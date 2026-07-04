import json

from inference.utils import extract_sql_query, save_queries


def test_extract_sql_query_prefers_fenced_sql():
    text = "Here is the query:\n```sql\nselect name from museum;\n```"

    assert extract_sql_query(text) == "select name from museum;"


def test_extract_sql_query_falls_back_to_plain_select():
    text = "The answer is select count(*) from museum;"

    assert extract_sql_query(text) == "select count(*) from museum;"


def test_extract_sql_query_returns_none_when_absent():
    assert extract_sql_query("No SQL here.") is None


def test_save_queries_uses_safe_model_filename(tmp_path):
    output = save_queries("org/model", ["select 1"], few_shot=False, output_dir=tmp_path)

    assert output.name == "query_org_model_few_shot=False.json"
    assert json.loads(output.read_text(encoding="utf-8")) == ["select 1"]
