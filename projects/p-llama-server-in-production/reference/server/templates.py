def format_chat(messages, special_tokens=True):
    res = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if special_tokens:
            res += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        else:
            res += f"{role}: {content}\n"
    if special_tokens:
        res += "<|im_start|>assistant\n"
    return res
