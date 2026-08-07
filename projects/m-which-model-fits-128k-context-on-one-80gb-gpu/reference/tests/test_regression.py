import sys
sys.path.insert(0, ".")
from kvmodel.sizing import compute_kv_bytes
from kvmodel.dtypes import dtype_comparison_table
from kvmodel.feasibility import fits_on_gpu

CONFIG = {
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "weight_bytes": 30 * 1024 * 1024 * 1024
}

def test_sizing_scaling():
    b1 = compute_kv_bytes(CONFIG, 1024, 1, 2)
    b2 = compute_kv_bytes(CONFIG, 2048, 1, 2)
    assert b2 == 2 * b1

def test_dtype_ordering():
    tbl = dtype_comparison_table(CONFIG, 1024, 1)
    assert tbl["fp16"] > tbl["fp8"]
    assert tbl["fp8"] > tbl["int4"]

def test_feasibility_check():
    ok = fits_on_gpu(CONFIG, 128 * 1024, 1, 80 * 1024 * 1024 * 1024, 0.5)
    assert isinstance(ok, bool)
