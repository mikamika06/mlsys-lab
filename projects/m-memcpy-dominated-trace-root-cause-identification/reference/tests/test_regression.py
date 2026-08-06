import sys

sys.path.insert(0, ".")
from trace_analyzer.diff import compare_profiles


def test_op_eliminated_entirely():
    pa = {"Memcpy": {"count": 10, "duration_us": 5000.0}}
    pb = {}
    res = compare_profiles(pa, pb)
    assert len(res) == 1
    assert res[0]["op_name"] == "Memcpy"
    assert res[0]["count_diff"] == -10
    assert res[0]["duration_diff"] == -5000.0


def test_op_added_entirely():
    pa = {}
    pb = {"FusedConv": {"count": 1, "duration_us": 100.0}}
    res = compare_profiles(pa, pb)
    assert len(res) == 1
    assert res[0]["count_diff"] == 1
