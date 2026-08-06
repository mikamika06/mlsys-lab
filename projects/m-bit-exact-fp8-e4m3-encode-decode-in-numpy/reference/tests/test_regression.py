import numpy as np
from fp8kv.compare import compare_formats_on_kv_dump


def test_quantization_safety():
    rng = np.random.default_rng(42)
    kv_data = rng.normal(loc=0.0, scale=1.0, size=(16, 8, 64, 128)).astype(np.float32)

    res = compare_formats_on_kv_dump(kv_data)
    assert "e4m3_mse" in res and "e5m2_mse" in res
    assert res["e4m3_mse"] < res["e5m2_mse"]
