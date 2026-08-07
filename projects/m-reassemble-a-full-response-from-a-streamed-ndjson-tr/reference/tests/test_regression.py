import sys
sys.path.insert(0, ".")
from rawstream.reassemble import reassemble_stream
from rawstream.delta import quantify_delta
from rawstream.fim import make_fim_request
import json

def test_reassemble_basic():
    lines = [json.dumps({"response": "foo"}), json.dumps({"response": "bar", "done": True})]
    assert reassemble_stream(lines) == "foobar"

def test_delta_structure():
    res = quantify_delta("hello", system_prompt="sys")
    assert "char_delta" in res
    assert res["chat_chars"] > 0

def test_fim_format():
    req = make_fim_request("def foo():\n    ", "\n    return x")
    assert "<|fim_prefix|>" in req["prompt"]
    assert "<|fim_suffix|>" in req["prompt"]
