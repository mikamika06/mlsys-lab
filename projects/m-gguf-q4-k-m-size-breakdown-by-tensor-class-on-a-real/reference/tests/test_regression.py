import sys
sys.path.insert(0, ".")
from moe_quant.breakdown import compute_tensor_breakdown
from moe_quant.perplexity import estimate_perplexity_delta
from moe_quant.map import build_quant_map


def test_breakdown_categorization():
    tensors = [
        ("blk.0.attn_q.weight", 1000, "Q4_K"),
        ("blk.0.ffn_gate.weight", 500, "Q4_K"),
        ("blk.0.ffn_act.weight", 200, "FP16")
    ]
    res = compute_tensor_breakdown(tensors)
    assert "attention" in res
    assert "router" in res
    assert "expert" in res


def test_perplexity_monotonicity():
    base = 10.0
    p1 = estimate_perplexity_delta(base, True, False)
    p2 = estimate_perplexity_delta(base, True, True)
    assert p2 >= p1


def test_map_thresholding():
    traces = {0: [1.0, 2.0], 1: [10.0, 20.0]}
    res = build_quant_map(traces, 5.0)
    assert res[0] == "Q4_K_M"
    assert res[1] == "FP16"
