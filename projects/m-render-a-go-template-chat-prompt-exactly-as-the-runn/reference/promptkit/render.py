def render_chat_prompt(template_str, messages, add_generation_prompt=True):
    out = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if "{{.System}}" in template_str and role == "system":
            out.append(f"System: {content}\n")
        elif role == "user":
            out.append(f"User: {content}\n")
        elif role == "assistant":
            out.append(f"Assistant: {content}\n")
    if add_generation_prompt:
        out.append("Assistant: ")
    return "".join(out)
