import sys

sys.path.insert(0, ".")
from tokenizer.engine import GGUFTokenizer


def test_tokenizer_regression():
    tok = GGUFTokenizer({"vocab": {}, "merges": []})
    text = "hello world"
    assert tok.tokenize(text) == [ord(c) for c in text]
