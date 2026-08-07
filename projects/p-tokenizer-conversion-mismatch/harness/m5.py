import ref
from tokenizer.engine import GGUFTokenizer


def check(workdir):
    m = {"roundtrip_ok": 0.0}
    corpus = ref.generate_corpus(seed=123)
    tok = GGUFTokenizer({"vocab": {}, "merges": []})
    ok = True
    for text in corpus[:50]:
        tokens = tok.tokenize(text)
        back = tok.detokenize(tokens)
        if back != text:
            ok = False
            break
    m["roundtrip_ok"] = 1.0 if ok else 0.0
    return m
