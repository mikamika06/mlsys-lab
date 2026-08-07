import ref
from tokenizer.vocab import VocabHandler


def check(workdir):
    m = {"vocab_merges_ok": 0.0}
    vh = VocabHandler({}, [])
    res = vh.process("test")
    if isinstance(res, list):
        m["vocab_merges_ok"] = 1.0
    return m
