from bandwidth.tracker import compute_bytes_transferred


def test_recomputation_byte_accounting():
    """Verify byte accounting catches recomputation mismatches."""
    cfg = {
        "batch_size": 2,
        "num_heads": 8,
        "seq_len": 1024,
        "head_dim": 64,
        "element_bytes": 2,
        "block_r": 128,
        "block_c": 128,
    }
    res = compute_bytes_transferred(cfg)
    tr = 1024 // 128
    expected_tiled = 2 * 8 * 1024 * 64 * 2 * (2 + 2 * tr)
    assert res["tiled_bytes"] == expected_tiled
    assert res["tiled_bytes"] < res["naive_bytes"]
