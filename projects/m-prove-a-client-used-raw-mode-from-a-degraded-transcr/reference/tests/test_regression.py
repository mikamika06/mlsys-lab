import sys
sys.path.insert(0, ".")
from runner.mapping import map_openai_to_native

def test_mapping_preserves_temperature():
    req = {"temperature": 0.5, "max_tokens": 64, "messages": [{"role": "user", "content": "hello"}]}
    res = map_openai_to_native(req)
    assert res["temperature"] == 0.5

def test_mapping_formats_prompt():
    req = {"messages": [{"role": "user", "content": "hi"}]}
    res = map_openai_to_native(req)
    assert "user: hi" in res["prompt"]
