import sys
import numpy as np

sys.path.insert(0, ".")
from numval.gate import evaluate_gate
from numval.metrics import cosine_similarity, sqnr
from numval.amplification import analyze_amplification


def test_gate_rejects_excessive_relative_error():
    ref_data = np.ones((100,), dtype=np.float64)
    test_data = np.ones((100,), dtype=np.float64)
    test_data[0] = 50.0
    res = evaluate_gate(ref_data, test_data, min_sqnr_db=10.0, min_cos_sim=0.5, max_rel_err=0.01)
    assert not res["passed"], "Gate must reject inputs with relative error above threshold"


def test_sqnr_perfect_match():
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    val = sqnr(vec, vec)
    assert val >= 100.0, f"Perfect match should return high SQNR, got {val}"


def test_amplification_identifies_peak():
    layer_refs = [np.ones((10, 10)) for _ in range(3)]
    layer_tests = [
        np.ones((10, 10)) + 0.001,
        np.ones((10, 10)) + 0.001,
        np.ones((10, 10)) + 0.5,
    ]
    res = analyze_amplification(layer_refs, layer_tests)
    assert res["max_amplifying_layer"] == 2, f"Expected layer 2 max amplification, got {res['max_amplifying_layer']}"
