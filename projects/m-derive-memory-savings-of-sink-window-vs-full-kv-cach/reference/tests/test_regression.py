from streamkv.analysis import compute_kv_bytes, compute_sink_window_bytes


def test_eviction_bounds():
    full = compute_kv_bytes(10000, 32, 8, 128)
    sink = compute_sink_window_bytes(10000, 32, 8, 128, 4, 512)
    assert sink < full
    assert sink == compute_sink_window_bytes(50000, 32, 8, 128, 4, 512)
