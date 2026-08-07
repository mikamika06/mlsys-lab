def check(workdir):
    from server.templates import format_chat
    m = {"template_ok": 0.0}
    msgs = [{"role": "user", "content": "Hello"}]
    res = format_chat(msgs, special_tokens=True)
    if "<|im_start|>user" in res and "<|im_end|>" in res:
        m["template_ok"] = 1.0
    return m
