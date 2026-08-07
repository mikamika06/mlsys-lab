import sys

sys.path.insert(0, ".")
from seqcost.cost import execution_cost_ratio


def test_ratio_exceeds_one():
    cfg = {"hidden_size": 2048, "num_layers": 22, "num_kv_heads": 4, "head_dim": 128, "vocab_size": 32000}
    ratio = execution_cost_ratio(cfg, 2048, 4, 150.0)
    assert ratio > 1.0, f"expected sequential cost to exceed parallel estimate, got {ratio}"
