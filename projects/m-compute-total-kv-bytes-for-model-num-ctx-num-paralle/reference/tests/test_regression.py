def test_missing_kv_factor():
    from slots.memory import compute_kv_bytes
    val = compute_kv_bytes(32, 8, 128, 8192, 4, 2)
    assert val == 2 * 32 * 8 * 128 * 8192 * 4 * 2
