import sys
sys.path.insert(0, ".")
from msprofiler.trace import parse_signposts
from msprofiler.metrics import compute_misattribution
from msprofiler.compare import compare_framework_timings


def test_parse_signposts_valid_xml():
    xml = '<trace><signpost name="MPSGraph" start="10.5" duration="45.2" subsystem="PyTorchMPS"/></trace>'
    res = parse_signposts(xml)
    assert len(res) == 1
    assert res[0]["name"] == "MPSGraph"
    assert res[0]["duration"] == 45.2


def test_compute_misattribution_bounds():
    val = compute_misattribution([100, 200], [50, 50])
    assert 0.0 <= val <= 1.0
    assert val > 0.5


def test_compare_framework_timings_speedup():
    res = compare_framework_timings([1.0, 1.2], [3.0, 3.4])
    assert res["speedup"] > 1.0
    assert res["mean_mlx"] < res["mean_torch"]
