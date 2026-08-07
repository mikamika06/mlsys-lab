from pretokenize.hash import compute_pre_tokenizer_hash
from pretokenize.disagreement import measure_disagreement
from pretokenize.prompt import check_double_bos

def test_hash_consistency():
    cfg = {"pre_tokenizer_type": "bpe", "chk_txt": "test"}
    h1 = compute_pre_tokenizer_hash(cfg)
    h2 = compute_pre_tokenizer_hash(cfg)
    assert h1 == h2

def test_disagreement_identical():
    tokens = [1, 2, 3, 4]
    assert measure_disagreement(tokens, tokens) == 0.0

def test_double_bos_detection():
    bos = "<|begin_of_text|>"
    prompt = bos + bos + "Hello world"
    assert check_double_bos(prompt, bos) is True

    clean_prompt = bos + "Hello world"
    assert check_double_bos(clean_prompt, bos) is False
