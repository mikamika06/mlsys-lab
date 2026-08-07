import sys

sys.path.insert(0, ".")
from kvquant import (
    check_flash_attn_requirement,
    fit_context_budget,
    measure_fused_path_penalty,
)

CONFIG = {
    "num_layers": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "weights_bytes": 4 * 1024 * 1024 * 1024,
    "fixed_overhead_bytes": 512 * 1024 * 1024,
}


def test_asymmetric_penalty_detected():
    p_sym = measure_fused_path_penalty("q4_0", "q4_0")
    p_asym = measure_fused_path_penalty("q4_0", "q8_0")
    assert p_sym == 1.0
    assert p_asym > 1.0, f"Expected asymmetric penalty > 1.0, got {p_asym}"


def test_context_aligned_to_block_size():
    c = fit_context_budget(CONFIG, 8 * 1024 * 1024 * 1024, "f16", "f16", block_size=32)
    assert c % 32 == 0, f"Context size {c} is not aligned to block size 32"


def test_flash_attn_mandatory_for_quantized():
    assert check_flash_attn_requirement("q4_0", "f16", use_flash_attn=True) is True
    assert check_flash_attn_requirement("q4_0", "f16", use_flash_attn=False) is False
    assert check_flash_attn_requirement("f16", "f16", use_flash_attn=False) is True
