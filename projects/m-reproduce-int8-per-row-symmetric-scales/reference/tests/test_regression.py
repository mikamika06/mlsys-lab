import sys
sys.path.insert(0, ".")
from quant.storage import compute_int4_group_storage_bytes


def test_int4_storage_scaling():
    b = compute_int4_group_storage_bytes(1024, 128)
    expected_max = (1024 // 2) + (1024 // 128) * 4
    assert b <= expected_max
