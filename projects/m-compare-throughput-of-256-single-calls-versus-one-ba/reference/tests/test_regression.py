import sys
sys.path.insert(0, ".")
import numpy as np
from embedrunner.core import compare_throughput
from embedrunner.detect import is_l2_normalized
from embedrunner.safety import analyze_model_mixing

def test_throughput_ratio_gt_one():
    res = compare_throughput(256)
    assert res["ratio"] > 1.0, f"Expected batch throughput ratio > 1.0, got {res['ratio']}"

def test_l2_detector_accuracy():
    norm_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    unnorm_vecs = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert is_l2_normalized(norm_vecs) is True
    assert is_l2_normalized(unnorm_vecs) is False

def test_model_mixing_analysis():
    res = analyze_model_mixing()
    assert isinstance(res, dict)
    assert "cross_similarity" in res
