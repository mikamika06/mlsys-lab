RENDER_CASES = [
    {
        "template": "{{ .System }}{{ range .Messages }}{{ if eq .Role \"user\" }}USER: {{ .Content }}\n{{ else }}ASSISTANT: {{ .Content }}\n{{ end }}{{ end }}",
        "system": "You are a helpful assistant.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "expected": "You are a helpful assistant.USER: Hello\nASSISTANT: Hi there!\n"
    },
    {
        "template": "{{- range .Messages -}}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{- end -}}",
        "system": "",
        "messages": [
            {"role": "user", "content": "Ping"},
            {"role": "assistant", "content": "Pong"}
        ],
        "expected": "<|im_start|>user\nPing<|im_end|>\n<|im_start|>assistant\nPong<|im_end|>\n"
    }
]

TOKEN_CASES = [
    (
        {"eos_token": "<|endoftext|>", "special_tokens": ["<|im_start|>", "<|im_end|>"]},
        {"eos_token": "<|endoftext|>", "special_tokens": ["<|im_start|>", "<|im_wrong|>"]},
        "<|im_wrong|>"
    ),
    (
        {"bos_token": "<s>", "eos_token": "</s>"},
        {"bos_token": "<s>", "eos_token": "<eos>"},
        "<eos>"
    )
]

GGUF_CASES = [
    (
        {"tokenizer.chat_template": "{{ .Prompt }}", "general.architecture": "llama"},
        "{{ .Prompt }}"
    ),
    (
        {"tokenizer.chat_template": "{{- .System -}}", "general.architecture": "mistral"},
        "{{- .System -}}"
    )
]

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

def find_mismatched_token(modelfile_meta, card_meta):
    s1 = set(modelfile_meta.get("special_tokens", []))
    if "eos_token" in modelfile_meta:
        s1.add(modelfile_meta["eos_token"])
    if "bos_token" in modelfile_meta:
        s1.add(modelfile_meta["bos_token"])

    s2 = set(card_meta.get("special_tokens", []))
    if "eos_token" in card_meta:
        s2.add(card_meta["eos_token"])
    if "bos_token" in card_meta:
        s2.add(card_meta["bos_token"])

    diff = s2 - s1
    if diff:
        return sorted(list(diff))[0]
    diff_rev = s1 - s2
    if diff_rev:
        return sorted(list(diff_rev))[0]
    return None

def recover_chat_template(metadata):
    return metadata.get("tokenizer.chat_template", "")

def compare_with_ollama(recovered, show_output):
    return recovered.strip() == show_output.strip()
