from chatstream.rendering import render_chat_template


def count_tokens(messages: list[dict], tools: list[dict] | None = None) -> dict:
    prompt_str = render_chat_template(messages, tools)
    tokens = prompt_str.split()
    prompt_tokens = len(tokens) * 2 + len(messages)
    completion_tokens = sum(len(str(m.get("content", "")).split()) * 2 for m in messages if m.get("role") == "assistant")
    if completion_tokens == 0:
        completion_tokens = 12
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
