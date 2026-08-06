import sys

sys.path.insert(0, ".")
from moecov.coverage import measure_coverage
from moecov.compare import compare_imatrices
from moecov.detect import detect_truncation


def test_coverage_basic():
    res = measure_coverage([[0, 1]], 2)
    assert res["coverage_ratio"] == 1.0


def test_compare_identical():
    sim = compare_imatrices({"a": [1.0]}, {"a": [1.0]})
    assert abs(sim - 1.0) < 1e-5


def test_detect_truncation_valid():
    assert detect_truncation({"layers": [{"data": [1]}]}, 1) is False


def test_detect_truncation_invalid():
    assert detect_truncation({"layers": []}, 1) is True
