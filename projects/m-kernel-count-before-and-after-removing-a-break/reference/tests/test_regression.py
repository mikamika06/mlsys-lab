import sys

sys.path.insert(0, ".")
from fusion.bytes_calc import calc_bytes

def test_fusion_saves_bytes():
    ops = [
        {"op": "add", "inputs": ["a", "b"], "output": "c"},
        {"op": "mul", "inputs": ["c", "d"], "output": "e"}
    ]
    fused = calc_bytes(ops, 4, 1024, True)
    unfused = calc_bytes(ops, 4, 1024, False)
    assert fused < unfused, f"Fused ({fused}) must be less than unfused ({unfused}) bytes"

def test_fusion_exact_bytes():
    ops = [
        {"op": "add", "inputs": ["a", "b"], "output": "c"},
        {"op": "mul", "inputs": ["c", "d"], "output": "e"}
    ]
    fused = calc_bytes(ops, 4, 10, True)
    assert fused == 4 * 4 * 10
