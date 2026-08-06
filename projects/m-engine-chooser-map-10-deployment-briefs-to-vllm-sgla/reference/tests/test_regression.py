from chooser.memory import estimate_memory, compare_gguf_vs_w4a16

def test_estimate_memory_positive():
    res = estimate_memory(7, 4.0, 1024)
    assert res > 0

def test_compare_gguf_w4a16_keys():
    res = compare_gguf_vs_w4a16(7, 2048)
    assert "gguf_q4_bytes" in res
    assert "w4a16_bytes" in res
    assert "diff_bytes" in res

def test_memory_scales_with_parameters():
    small = estimate_memory(7, 4.0, 0)
    large = estimate_memory(70, 4.0, 0)
    assert large > small
