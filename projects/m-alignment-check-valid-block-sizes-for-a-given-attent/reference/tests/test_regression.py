from blockalign.validator import filter_valid_block_sizes, validate_block_size


def test_tile_alignment_rejection():
    """Test that block sizes misaligned with backend block multiple are rejected."""
    backend = {"min_block_size": 16, "max_block_size": 256, "block_multiple": 16, "alignment_bytes": 16}
    model = {"num_kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "is_quantized": False}

    res_good = validate_block_size(backend, model, 32)
    assert res_good["valid"] is True, "Block size 32 should be valid for multiple 16"

    res_bad = validate_block_size(backend, model, 24)
    assert res_bad["valid"] is False, "Block size 24 should be invalid for multiple 16"
    assert res_bad["reason"] == "tile_multiple_misaligned"

    filtered = filter_valid_block_sizes(backend, model, [16, 24, 32])
    assert filtered == [16, 32], f"Expected [16, 32], got {filtered}"


def test_quant_group_alignment_rejection():
    """Test that block sizes misaligned with quantization group size are rejected."""
    backend = {"min_block_size": 16, "max_block_size": 256, "block_multiple": 16, "alignment_bytes": 16}
    model = {
        "num_kv_heads": 4,
        "head_dim": 64,
        "dtype_bytes": 1,
        "is_quantized": True,
        "group_size": 32,
        "scale_dtype_bytes": 2,
    }

    res_bad = validate_block_size(backend, model, 16)
    assert res_bad["valid"] is False, "Block size 16 should be invalid for group_size 32"
    assert res_bad["reason"] == "quant_group_misaligned"

    res_good = validate_block_size(backend, model, 32)
    assert res_good["valid"] is True, "Block size 32 should be valid for group_size 32"

    filtered = filter_valid_block_sizes(backend, model, [16, 32, 48, 64])
    assert filtered == [32, 64], f"Expected [32, 64], got {filtered}"
