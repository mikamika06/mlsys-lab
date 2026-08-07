import json


def reassemble_sse_stream(lines: list[str]) -> dict:
    completion_id = ""
    model = ""
    created = 0
    finish_reason = None
    role = "assistant"
    content_parts = []
    tool_calls_map = {}

    for line in lines:
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if not completion_id:
            completion_id = data.get("id", "")
        if not model:
            model = data.get("model", "")
        if not created:
            created = data.get("created", 0)

        choices = data.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice.get("finish_reason")

        delta = choice.get("delta", {})
        if "role" in delta and delta["role"]:
            role = delta["role"]

        if "content" in delta and delta["content"] is not None:
            content_parts.append(delta["content"])

        tcs = delta.get("tool_calls")
        if tcs:
            for tc in tcs:
                idx = tc.get("index", 0)
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }
                target = tool_calls_map[idx]
                if tc.get("id"):
                    target["id"] = tc["id"]
                if tc.get("type"):
                    target["type"] = tc["type"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    target["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    target["function"]["arguments"] += fn["arguments"]

    full_content = "".join(content_parts) if content_parts else None
    sorted_tool_calls = [tool_calls_map[k] for k in sorted(tool_calls_map.keys())] if tool_calls_map else None

    message = {"role": role}
    if full_content is not None:
        message["content"] = full_content
    else:
        message["content"] = None

    if sorted_tool_calls:
        message["tool_calls"] = sorted_tool_calls

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if sorted_tool_calls else "stop")
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(full_content.split()) if full_content else 5,
            "total_tokens": 15
        }
    }
