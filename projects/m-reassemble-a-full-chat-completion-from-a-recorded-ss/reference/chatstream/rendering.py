import json


def render_chat_template(messages: list[dict], tools: list[dict] | None = None) -> str:
    rendered_parts = []
    if tools:
        tools_json = json.dumps(tools, sort_keys=True)
        rendered_parts.append(f"<|im_start|>system\nAvailable Tools: {tools_json}<|im_end|>")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        rendered_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    rendered_parts.append("<|im_start|>assistant\n")
    return "\n".join(rendered_parts)
