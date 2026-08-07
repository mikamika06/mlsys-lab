import sys

sys.path.insert(0, ".")
from kvquant.decision import should_quantize_kv


def test_should_quantize_kv_validates_threshold():
    res = should_quantize_kv(1024, 0.05, 0.01)
    assert isinstance(res, bool)
    res_large = should_quantize_kv(100000, 0.01, 0.05)
    assert res_large is False
