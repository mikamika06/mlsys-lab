import numpy as np
from kquant.amortization import compute_superblock_footprint, calculate_amortization_advantage


def test_super_scale_amortization_advantage():
    data = np.sin(np.linspace(0, 100, 2048)).astype(np.float32)
    res = calculate_amortization_advantage(data, superblock_size=256, subblock_size=32, quant_bits=4)
    assert res["advantage_ratio"] > 1.0, f"Expected advantage ratio > 1.0, got {res['advantage_ratio']}"


def test_metadata_ratio_lower_for_superblocks():
    fp_super = compute_superblock_footprint(2048, 256, 32, 4, 6, 16)
    fp_uniform = compute_superblock_footprint(2048, 32, 32, 4, 0, 16)
    assert fp_super["metadata_ratio"] < fp_uniform["metadata_ratio"]
