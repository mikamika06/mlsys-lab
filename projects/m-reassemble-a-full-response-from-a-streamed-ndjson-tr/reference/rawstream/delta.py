import json

def quantify_delta(prompt, system_prompt=""):
    gen_body = {"prompt": (system_prompt + "\n" + prompt) if system_prompt else prompt, "raw": True}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    chat_body = {"messages": messages}
    gen_str = json.dumps(gen_body)
    chat_str = json.dumps(chat_body)
    return {
        "generate_chars": len(gen_str),
        "chat_chars": len(chat_str),
        "char_delta": len(chat_str) - len(gen_str),
        "generate_bytes": len(gen_str.encode("utf-8")),
        "chat_bytes": len(chat_str.encode("utf-8")),
        "byte_delta": len(chat_str.encode("utf-8")) - len(gen_str.encode("utf-8"))
    }
