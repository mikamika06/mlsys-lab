import sys
sys.path.insert(0, ".")
from quantlib.crossover import measure_scheme, find_crossover
from quantlib.metrics import compute_ratio

def test_compute_ratio_basic():
    assert compute_ratio(10.0, 5.0) == 2.0

def test_measure_scheme_positive():
    res = measure_scheme({"bits": 4, "overhead": 2, "name": "W4A16"}, {"intensity": 1.0, "base_ops": 1000})
    assert res["throughput"] > 0

def test_find_crossover_valid():
    schemes = [{"name": "A", "bits": 2, "overhead": 10}, {"name": "B", "bits": 8, "overhead": 2}]
    workloads = [{"id": 1, "intensity": 0.5, "base_ops": 1000}, {"id": 2, "intensity": 5.0, "base_ops": 1000}]
    res = find_crossover(schemes, workloads)
    assert res is not None
    assert "throughput_ratio" in res
