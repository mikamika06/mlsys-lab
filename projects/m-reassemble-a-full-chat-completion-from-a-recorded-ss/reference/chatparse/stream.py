import json

def reassemble_chat_completion(lines):
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
