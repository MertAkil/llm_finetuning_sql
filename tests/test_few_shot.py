import json

from inference.few_shot_script import few_shot_template


def test_few_shot_template_supports_query_and_answer_fields(tmp_path):
    examples = tmp_path / "few_shot.jsonl"
    records = [
        {"question": "Q1", "context": "create table t (a int)", "answer": "select a from t"},
        {"question": "Q2", "context": "create table u (b int)", "query": "select b from u"},
    ]
    examples.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    messages = few_shot_template(few_shot_path=examples)

    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["content"] == "select a from t"
    assert messages[3]["content"] == "select b from u"
