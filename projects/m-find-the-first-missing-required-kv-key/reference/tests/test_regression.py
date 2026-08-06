import sys
sys.path.insert(0, ".")
from ggufschema.validator import find_first_missing_key
from ggufschema.parser import derive_gqa_and_head_dim

def test_find_first_missing_key_basic():
    req = ["llama.vocab_size", "llama.context_length", "llama.embedding_length"]
    meta = {"llama.vocab_size": 32000}
    assert find_first_missing_key(meta, req) == "llama.context_length"

def test_derive_gqa_and_head_dim_basic():
    meta = {"embedding_length": 4096, "attention.head_count": 32, "attention.head_count_kv": 8}
    res = derive_gqa_and_head_dim(meta)
    assert res["gqa_ratio"] == 4
    assert res["head_dim"] == 128
