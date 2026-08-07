import json
import sys

sys.path.insert(0, ".")
from toolutils.normalize import normalize_argument, normalize_messages


def test_dict_argument_becomes_valid_json_string():
    raw_arg = {"location": "San Francisco", "units": "celsius"}
    res = normalize_argument(raw_arg)
    assert isinstance(res, str)
    parsed = json.loads(res)
    assert parsed == raw_arg


def test_messages_normalization_preserves_json_string_type():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                    }
                }
            ],
        }
    ]
    norm = normalize_messages(messages)
    args = norm[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"location": "Tokyo"}
