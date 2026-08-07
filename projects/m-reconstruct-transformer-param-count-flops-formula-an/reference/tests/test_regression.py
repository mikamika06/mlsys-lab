import sys

sys.path.insert(0, ".")
from studentdesign.compare import validate_architecture_match

def test_architecture_match_returns_dict():
    cfg = {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072, "vocab_size": 32000, "seq_len": 512}
    res = validate_architecture_match(cfg, 1000)
    assert isinstance(res, dict)
    assert "depth_only" in res
    assert "width_only" in res
    assert res["params"] > 0
