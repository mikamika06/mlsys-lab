import sys

sys.path.insert(0, ".")
from lquant.plan import resolve_plan
from lquant.quant import predict_size
from lquant.overrides import parse_overrides

def test_override_is_applied_to_attn_v():
    tensors = {"blk.0.attn_v.weight": [64, 64], "blk.0.attn_q.weight": [64, 64]}
    plan = resolve_plan(tensors, "Q4_0", {"attn_v": "Q8_0"})
    assert plan["blk.0.attn_v.weight"] == "Q8_0"
    assert plan["blk.0.attn_q.weight"] == "Q4_0"

def test_parse_overrides_multiple():
    args = ["attn_v=Q8_0", "ffn_gate=F16"]
    res = parse_overrides(args)
    assert res.get("attn_v") == "Q8_0"
    assert res.get("ffn_gate") == "F16"

def test_predict_size_non_zero():
    tensors = {"weight": [100, 100]}
    plan = {"weight": "Q8_0"}
    sz = predict_size(tensors, plan)
    assert sz > 0
