import sys
sys.path.insert(0, ".")
from pqutils.analysis import analytical_error_comparison
from pqutils.pipeline import run_joint_pipeline
from pqutils.eval import measure_reconstruction_error
import numpy as np


def test_pipeline_order_difference():
    np.random.seed(42)
    w = np.random.randn(16, 16)
    h = np.eye(16)
    res_ptq = run_joint_pipeline(w, h, 0.5, 4, "prune_then_quantize")
    res_qtp = run_joint_pipeline(w, h, 0.5, 4, "quantize_then_prune")
    assert not np.allclose(res_ptq, res_qtp), "Prune-then-quantize and quantize-then-prune should yield different outcomes"


def test_analytical_bounds():
    np.random.seed(100)
    w = np.random.randn(8, 8)
    comp = analytical_error_comparison(w, 0.3, 4)
    assert "prune_then_quantize_error" in comp
    assert "quantize_then_prune_error" in comp
    assert isinstance(comp["delta"], float)


def test_reconstruction_metrics():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[1.1, 1.9], [3.0, 4.2]]
    metrics = measure_reconstruction_error(a, b)
    assert metrics["mse"] > 0.0
    assert metrics["max"] > 0.0
