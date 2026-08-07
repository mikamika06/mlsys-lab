def render_chat(template: str, messages: list) -> str:
    out = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        out.append(f"<|start|>{role}\n{content}<|end|>\n")
    return "".join(out)
