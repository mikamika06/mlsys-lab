import sys
sys.path.insert(0, ".")
from moeoff.resolver import resolve_ot_regexes

def test_override_priority():
    tensors = [("blk.0.ffn_gate.weight", 100)]
    overrides = [(".*", "CPU"), ("blk.0.ffn_gate.*", "GPU")]
    res = resolve_ot_regexes(tensors, overrides, "GPU")
    assert res["blk.0.ffn_gate.weight"] == "GPU"
