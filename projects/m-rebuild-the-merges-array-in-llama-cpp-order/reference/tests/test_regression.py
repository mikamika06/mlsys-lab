import sys
sys.path.insert(0, ".")
from llama_cpp_tok.merges import rebuild_merges
from llama_cpp_tok.vocab import classify_vocab_type, find_wrong_token_type

def test_rebuild_merges_order():
    raw = ["apple banana", "cat dog", "ant bat"]
    res = rebuild_merges(raw)
    assert len(res) == 3

def test_classify_vocab():
    art = {"has_merges": True, "has_scores": False}
    assert classify_vocab_type(art) == "BPE"

def test_find_wrong_token():
    tokens = [{"id": 0, "token_type": 1}, {"id": 1, "token_type": 1}, {"id": 2, "token_type": 99}]
    assert find_wrong_token_type(tokens) == 2
