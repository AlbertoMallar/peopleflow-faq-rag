import json
from pathlib import Path


def test_sample_queries_json_es_valido_y_consistente():
    path = Path("outputs/sample_queries.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(data, list)
    assert len(data) >= 3

    for item in data:
        assert list(item.keys()) == [
            "user_question",
            "system_answer",
            "chunks_related",
        ]
        assert len(item["chunks_related"]) == 3
