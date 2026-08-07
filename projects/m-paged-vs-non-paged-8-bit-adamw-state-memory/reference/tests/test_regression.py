import sys
import numpy as np

sys.path.insert(0, ".")
from qlora_mem.adamw import compute_adamw_state_bytes
from qlora_mem.lora_merge import merge_lora_into_base, CODEBOOK_4BIT


def test_paged_adamw_memory_lower_than_non_paged():
    num_params = 20_000_000
    max_layer = 650_000
    non_paged = compute_adamw_state_bytes(num_params, block_size=256, paged=False, max_layer_params=max_layer)
    paged = compute_adamw_state_bytes(num_params, block_size=256, paged=True, max_layer_params=max_layer)
    assert paged < non_paged, f"Paged memory ({paged}) should be smaller than non-paged ({non_paged})"


def test_merge_lora_scales_dequantization():
    qweights = np.zeros((64, 64), dtype=np.uint8)
    scales = np.full((64,), 5.0, dtype=np.float32)
    lora_A = np.zeros((8, 64), dtype=np.float32)
    lora_B = np.zeros((64, 8), dtype=np.float32)
    alpha = 16.0

    merged = merge_lora_into_base(qweights, scales, lora_A, lora_B, alpha, block_size=64)
    expected_val = CODEBOOK_4BIT[0] * 5.0
    assert np.allclose(merged, expected_val), f"Expected scale 5.0 to affect dequantized values, got {merged[0, 0]}"
