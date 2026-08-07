def render(template, system, messages):
    res = system
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if "{{ if eq .Role \"user\" }}" in template:
            if role == "user":
                res += f"USER: {content}\n"
            else:
                res += f"ASSISTANT: {content}\n"
        else:
            res += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return res
