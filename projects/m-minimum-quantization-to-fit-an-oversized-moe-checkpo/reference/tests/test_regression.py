import sys

sys.path.insert(0, ".")
from moefit.memory import calculate_memory_bytes
from moefit.quant import find_min_quant_bits

SPEC = {
    "name": "TestMoE",
    "total_params": 10e9,
    "expert_params": 2e9,
    "shared_params": 1e9,
    "layers": 16,
}

def test_memory_scales_monotonically_with_bits():
    m2 = calculate_memory_bytes(SPEC, 2)
    m4 = calculate_memory_bytes(SPEC, 4)
    m8 = calculate_memory_bytes(SPEC, 8)
    m16 = calculate_memory_bytes(SPEC, 16)
    assert m2 < m4 < m8 < m16, f"memory not monotonic: {m2}, {m4}, {m8}, {m16}"

def test_min_bits_valid_range():
    bits = find_min_quant_bits(SPEC, 5 * 1024 * 1024 * 1024)
    assert bits in [2, 3, 4, 5, 6, 8, 16], f"invalid bit width: {bits}"

def test_memory_positive():
    val = calculate_memory_bytes(SPEC, 4)
    assert val > 0, f"memory must be positive, got {val}"
