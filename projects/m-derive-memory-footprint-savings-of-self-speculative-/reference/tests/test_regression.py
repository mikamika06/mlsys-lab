import pytest
from speculative.derive import derive_savings


def test_self_speculative_savings_positive():
    target_cfg = {
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "num_kv_heads": 32,
        "head_dim": 128,
        "bytes_per_param": 2,
        "bytes_per_elem": 2
    }
    draft_cfg = {
        "num_layers": 8,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "num_kv_heads": 32,
        "head_dim": 128,
        "bytes_per_param": 2,
        "bytes_per_elem": 2,
        "is_self_speculative": True
    }
    res = derive_savings(target_cfg, draft_cfg, batch_size=4, seq_len=1024)
    assert res["saved_bytes"] > 0
    assert 0.0 <= res["savings_ratio"] <= 1.0


def test_separate_draft_zero_savings():
    target_cfg = {
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "num_kv_heads": 32,
        "head_dim": 128,
        "bytes_per_param": 2,
        "bytes_per_elem": 2
    }
    draft_cfg = {
        "num_layers": 8,
        "hidden_size": 2048,
        "intermediate_size": 5600,
        "vocab_size": 32000,
        "num_kv_heads": 16,
        "head_dim": 128,
        "bytes_per_param": 2,
        "bytes_per_elem": 2,
        "is_self_speculative": False
    }
    res = derive_savings(target_cfg, draft_cfg, batch_size=4, seq_len=1024)
    assert res["saved_bytes"] == 0
