import json
import jinja2

STREAM_SAMPLES = [
    [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]"
    ],
    [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"loc"}}]}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ation\": \"NYC\"}"}}]}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]"
    ],
    [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Checking"}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_2", "type": "function", "function": {"name": "search", "arguments": "{\"query\""}}]}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": \"vllm\"}"}}]}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]"
    ],
    [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Sure thing."}, "finish_reason": "stop"}]}),
        "data: [DONE]"
    ],
    [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Alpha"}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": " Beta"}, "finish_reason": "stop"}]}),
        "data: [DONE]"
    ]
]

TEMPLATE_SAMPLES = [
    {
        "template": "{% for m in messages %}{{m.role}}: {{m.content}}\n{% endfor %}assistant: ",
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": True
    },
    {
        "template": "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        "messages": [{"role": "user", "content": "Hello"}],
        "add_generation_prompt": True
    },
    {
        "template": "USER: {{messages[0].content}}\nASSISTANT: ",
        "messages": [{"role": "user", "content": "Test"}],
        "add_generation_prompt": True
    },
    {
        "template": "{% for m in messages %}[{{m.role}}] {{m.content}}\n{% endfor %}",
        "messages": [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hi"}],
        "add_generation_prompt": False
    },
    {
        "template": "Q: {{messages[0].content}}\nA: ",
        "messages": [{"role": "user", "content": "Why"}],
        "add_generation_prompt": True
    }
]

def reassemble_stream(lines):
    content_parts = []
    tool_calls_map = {}
    role = "assistant"
    finish_reason = "stop"
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except Exception:
            continue
        for choice in data.get("choices", []):
            if "finish_reason" in choice and choice["finish_reason"] is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            if "role" in delta and delta["role"]:
                role = delta["role"]
            if "content" in delta and delta["content"] is not None:
                content_parts.append(delta["content"])
            if "tool_calls" in delta and delta["tool_calls"]:
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "function": {"name": "", "arguments": ""}
                        }
                    if "id" in tc and tc["id"]:
                        tool_calls_map[idx]["id"] = tc["id"]
                    if "type" in tc and tc["type"]:
                        tool_calls_map[idx]["type"] = tc["type"]
                    fn = tc.get("function", {})
                    if "name" in fn and fn["name"]:
                        tool_calls_map[idx]["function"]["name"] += fn["name"]
                    if "arguments" in fn and fn["arguments"]:
                        tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]

    message = {"role": role}
    full_content = "".join(content_parts)
    if full_content or not tool_calls_map:
        message["content"] = full_content
    else:
        message["content"] = None

    if tool_calls_map:
        sorted_indices = sorted(tool_calls_map.keys())
        message["tool_calls"] = [tool_calls_map[i] for i in sorted_indices]

    return {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }
        ]
    }

def render_template(template_str, messages, add_generation_prompt=True):
    env = jinja2.Environment()
    template = env.from_string(template_str)
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)

def predict_tokens(prompts, completions):
    res = []
    for p, c in zip(prompts, completions):
        p_tokens = len(p.split())
        c_tokens = len(c.split())
        res.append({"prompt_tokens": p_tokens, "completion_tokens": c_tokens, "total_tokens": p_tokens + c_tokens})
    return res
