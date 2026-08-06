import sys

sys.path.insert(0, ".")
from schema_opt.rewrite import rewrite_schema

def test_rewrite_removes_patterns():
    schema = {"type": "object", "properties": {"name": {"type": "string", "pattern": "^[a-z]+$"}}}
    res = rewrite_schema(schema)
    assert "pattern" not in res["properties"]["name"]

def test_rewrite_adds_max_length():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    res = rewrite_schema(schema)
    assert res["properties"]["name"].get("maxLength") is not None
