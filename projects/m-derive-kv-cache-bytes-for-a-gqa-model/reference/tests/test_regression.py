from kvderive import calc

def test_kv_bytes_baseline():
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
    val = calc.calc_kv_bytes(cfg, 1024, 2)
    expected = 2 * 32 * 8 * 128 * 1024 * 2
    assert val == expected

def test_quant_comparison():
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
    f16_bytes = calc.calc_quant_kv_bytes(cfg, 2048, "F16")
    q4_bytes = calc.calc_quant_kv_bytes(cfg, 2048, "Q4_0")
    assert q4_bytes < f16_bytes
