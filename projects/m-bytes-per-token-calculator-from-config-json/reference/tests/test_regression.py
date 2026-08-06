import pytest
from kvcalc.calc import bytes_per_token
from kvcalc.concurrency import max_concurrency


def test_bytes_per_token_positive():
    cfg = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
    assert bytes_per_token(cfg, 2) > 0


def test_max_concurrency_scaling():
    budget = 1024 * 1024 * 1024
    b_tok = 1024
    seq_len = 2048
    c1 = max_concurrency(budget, b_tok, seq_len, 0)
    c2 = max_concurrency(budget // 2, b_tok, seq_len, 0)
    assert c1 == 2 * c2
