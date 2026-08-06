from sparsity.decision import build_sparsity_decision_table
from sparsity.hardware import validate_nm_tensorcore_alignment


def test_decision_engine():
    unaligned_layer = [{
        "name": "unaligned_k",
        "M": 128,
        "N": 128,
        "K": 20,
        "sparsity": 0.5,
        "dtype_bits": 16,
        "bandwidth_gbps": 900.0,
        "compute_tflops": 312.0
    }]
    res = build_sparsity_decision_table(unaligned_layer)
    assert not res[0]["is_hardware_aligned"]
    assert res[0]["recommended_format"] == "dense"
    assert res[0]["achievable_speedup"] == 1.0


def test_valid_alignment():
    val = validate_nm_tensorcore_alignment(64, 64, 64, 16)
    assert val["valid"] is True
