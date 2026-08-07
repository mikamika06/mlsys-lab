import ref
from tokenizer.engine import GGUFTokenizer


def check(workdir):
    m = {"corpus_match_rate": 0.0}
    corpus = ref.generate_corpus()
    tok = GGUFTokenizer({"vocab": {}, "merges": []})
    matches = 0
    for text in corpus:
        res = tok.tokenize(text)
        expected = ref.oracle_tokenize(text)
        if res == expected:
            matches += 1
    rate = matches / len(corpus)
    m["corpus_match_rate"] = 1.0 if rate >= 0.9 else 0.0
    return m
