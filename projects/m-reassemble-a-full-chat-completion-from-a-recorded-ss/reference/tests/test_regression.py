import sys
sys.path.insert(0, ".")
from chatparse.stream import reassemble_chat_completion
from chatparse.render import render_chat_template
from chatparse.tokens import predict_token_counts
import json

def test_reassemble_tool_calls_fully():
    sample = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "t1", "type": "function", "function": {"name": "foo", "arguments": "{\"x\":"}}]}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 1}"}}]}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]"
    ]
    res = reassemble_chat_completion(sample)
    msg = res["choices"][0]["message"]
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["function"]["arguments"] == "{\"x\": 1}"

def test_render_template_basic():
    rendered = render_chat_template("user: {{messages[0].content}}", [{"role": "user", "content": "hello"}])
    assert rendered == "user: hello"

def test_token_prediction_structure():
    counts = predict_token_counts(["hello world"], ["foo bar"])
    assert len(counts) == 1
    assert counts[0]["prompt_tokens"] == 2
    assert counts[0]["completion_tokens"] == 2
    assert counts[0]["total_tokens"] == 4
