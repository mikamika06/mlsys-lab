import sys
sys.path.insert(0, ".")
from mlxops.promotion import compute_promotion_table, promote_dtypes
from mlxops.reduction import measure_running_sum_error

def test_float_over_int_promotion():
    res = promote_dtypes("int32", "float16")
    assert res == "float16", f"expected float16, got {res}"

def test_float_precision_promotion():
    res = promote_dtypes("float16", "float32")
    assert res == "float32", f"expected float32, got {res}"

def test_bfloat16_float16_promotion():
    res = promote_dtypes("float16", "bfloat16")
    assert res == "float32", f"expected float32, got {res}"

def test_promotion_table():
    dtypes = ["int32", "float16", "float32"]
    table = compute_promotion_table(dtypes)
    assert len(table) == 9
    assert table[("int32", "float16")] == "float16"

def test_running_sum_error():
    data = [0.1] * 100
    res = measure_running_sum_error(data)
    assert res["max_err_fp16"] >= res["max_err_fp32"]
