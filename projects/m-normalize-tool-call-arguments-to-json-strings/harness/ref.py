import json

ARG_SAMPLES = [
    {"location": "San Francisco", "units": "celsius"},
    '{"query": "llama.cpp", "top_k": 5}',
    {"nested": {"key": [1, 2, 3]}, "flag": True},
    "",
    '  {"spaced": true}  ',
    {"a": 1, "b": 2},
]

MESSAGES_SAMPLE = [
    {
        "role": "user",
        "content": "Check weather in Tokyo and Paris",
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Tokyo", "unit": "celsius"},
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris", "unit": "celsius"}',
                },
            },
        ],
    },
    {
        "role": "tool",
        "content": '{"temperature": 22}',
    },
]


def normalize_argument(arg):
    if isinstance(arg, str):
        stripped = arg.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, sort_keys=True)
        except Exception:
            return json.dumps(arg, sort_keys=True)
    try:
        return json.dumps(arg, sort_keys=True)
    except Exception:
        return "{}"


def normalize_messages(messages):
    out = []
    for msg in messages:
        msg_copy = dict(msg)
        if "tool_calls" in msg_copy and isinstance(msg_copy["tool_calls"], list):
            norm_calls = []
            for call in msg_copy["tool_calls"]:
                call_copy = dict(call)
                if "function" in call_copy and isinstance(call_copy["function"], dict):
                    fn_copy = dict(call_copy["function"])
                    if "arguments" in fn_copy:
                        fn_copy["arguments"] = normalize_argument(fn_copy["arguments"])
                    call_copy["function"] = fn_copy
                elif "arguments" in call_copy:
                    call_copy["arguments"] = normalize_argument(call_copy["arguments"])
                norm_calls.append(call_copy)
            msg_copy["tool_calls"] = norm_calls
        out.append(msg_copy)
    return out
