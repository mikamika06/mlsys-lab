import sys
sys.path.insert(0, ".")
from quant.budget import compute_effective_bpw
from quant.evaluator import measure_quant_error, evaluate_end_to_end
from quant.selector import check_kernel_support, recommend_format
import numpy as np

def test_effective_bpw_computation():
    bpw = compute_effective_bpw("fp4", 32, 8)
    assert bpw == 4.25

def test_quant_error_positive():
    w = np.array([0.1, -0.5, 0.3, 0.8], dtype=np.float32)
    err = measure_quant_error(w, "fp4", 2)
    assert err >= 0.0

def test_kernel_support_blackwell():
    assert check_kernel_support("fp4", "blackwell") is True
    assert check_kernel_support("fp6", "hopper") is False

def test_recommendation_logic():
    candidates = [("fp8", 8.25, 0.01), ("fp4", 4.25, 0.05)]
    rec = recommend_recommendation_safe = recommend_format(candidates, 5.0, "blackwell")
    assert rec == "fp4"

def test_end_to_end_scoring():
    w = np.array([1.0, 2.0], dtype=np.float32)
    data = np.array([1.0, 1.0], dtype=np.float32)
    score = evaluate_end_to_end(w, data, "fp8")
    assert isinstance(score, float)
