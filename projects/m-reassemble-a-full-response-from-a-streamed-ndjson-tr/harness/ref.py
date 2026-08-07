import json

STREAMS = [
    [
        json.dumps({"response": "Hello", "done": False}),
        json.dumps({"response": " world", "done": False}),
        json.dumps({"response": "!", "done": True, "eval_count": 3})
    ],
    [
        json.dumps({"text": "System", "finished": False}),
        json.dumps({"text": " ready.", "finished": True})
    ],
    [
        json.dumps({"token": "foo"}),
        json.dumps({"token": " bar"}),
        json.dumps({"token": "", "done": True})
    ],
    [
        json.dumps({"choices": [{"text": "one"}]}),
        json.dumps({"choices": [{"text": " two"}]}),
        json.dumps({"choices": [], "finish_reason": "stop"})
    ],
    [
        json.dumps({"response": "A"}),
        json.dumps({"response": "B"}),
        json.dumps({"response": "C", "done": True})
    ]
]

def reassemble_stream(lines):
    out = []
    for line in lines:
        if not line.strip():
            continue
        data = json.loads(line)
        if "response" in data:
            out.append(data["response"])
        elif "text" in data:
            out.append(data["text"])
        elif "token" in data:
            out.append(data["token"])
        elif "choices" in data:
            for c in data["choices"]:
                if "text" in c:
                    out.append(c["text"])
    return "".join(out)

def compute_delta(prompt, system_prompt=""):
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

def build_fim(prefix, suffix, middle=""):
    return {
        "prompt": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}",
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle
    }
