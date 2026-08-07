import sys
sys.path.insert(0, ".")
from promptfmt.template import render
from promptfmt.tokens import find_mismatched_token
from promptfmt.gguf import recover_chat_template, compare_with_ollama

def test_render_basic():
    tpl = "{{ if eq .Role \"user\" }}USER: {{ .Content }}\n{{ else }}ASSISTANT: {{ .Content }}\n{{ end }}"
    res = render(tpl, "SYS", [{"role": "user", "content": "hi"}])
    assert "USER: hi" in res

def test_tokens_mismatch():
    mf = {"eos_token": "</s>"}
    card = {"eos_token": "<eos>"}
    assert find_mismatched_token(mf, card) == "<eos>"

def test_gguf_recovery():
    meta = {"tokenizer.chat_template": "{{ .Prompt }}"}
    assert recover_chat_template(meta) == "{{ .Prompt }}"
    assert compare_with_ollama("{{ .Prompt }}", "{{ .Prompt }}")
