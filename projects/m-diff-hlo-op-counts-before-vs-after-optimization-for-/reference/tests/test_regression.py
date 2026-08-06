import sys
sys.path.insert(0, ".")
from hlodiff.growth import measure_growth

def test_growth_increases_with_size():
    res = measure_growth([10, 50, 100])
    assert len(res) == 3
    for i in range(len(res) - 1):
        assert res[i]["size"] < res[i+1]["size"]
        assert res[i]["bytes"] < res[i+1]["bytes"]
        assert res[i]["line_count"] < res[i+1]["line_count"]

def test_growth_non_empty_results():
    res = measure_growth([5])
    assert len(res) == 1
    assert res[0]["bytes"] > 0
    assert res[0]["line_count"] > 0
