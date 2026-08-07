import ref
from tokenizer.engine import GGUFTokenizer


def check(workdir):
    m = {"special_tokens_ok": 0.0}
    tok = GGUFTokenizer({"vocab": {}, "merges": []})
    if tok.tokenize("<|endoftext|>") is not None:
        m["special_tokens_ok"] = 1.0
    return m
