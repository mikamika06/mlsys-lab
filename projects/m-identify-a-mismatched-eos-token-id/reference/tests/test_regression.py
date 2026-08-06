import sys
sys.path.insert(0, ".")

from gguf_meta.eos import find_mismatched_eos


def test_eos_mismatch_detection():
    meta_ok = {
        "general.architecture": "llama",
        "llama.tokenizer.eos_token_id": 2,
        "tokenizer.ggml.tokens": ["<pad>", "<s>", "</s>"],
        "tokenizer.ggml.token_type": [1, 1, 3]
    }
    res_ok = find_mismatched_eos(meta_ok)
    assert not res_ok["mismatch"]
    assert res_ok["arch_eos_id"] == 2
    assert res_ok["vocab_eos_id"] == 2

    meta_bad = {
        "general.architecture": "llama",
        "llama.tokenizer.eos_token_id": 2,
        "tokenizer.ggml.tokens": ["<pad>", "<s>", "<unk>", "</s>"],
        "tokenizer.ggml.token_type": [1, 1, 1, 3]
    }
    res_bad = find_mismatched_eos(meta_bad)
    assert res_bad["mismatch"]
    assert res_bad["arch_eos_id"] == 2
    assert res_bad["vocab_eos_id"] == 3
