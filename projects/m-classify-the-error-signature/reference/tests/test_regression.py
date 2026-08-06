import numpy as np
from errorclass.signature import classify_signature
from errorclass.tolerance import evaluate_tolerance
from errorclass.metrics import compute_metrics


def test_signature_shape_mismatch():
    ref = np.zeros((4, 4), dtype=np.float32)
    tgt = np.zeros((2, 2), dtype=np.float32)
    sig = classify_signature(ref, tgt, {})
    assert sig == "SHAPE_MISMATCH"


def test_tolerance_rejection():
    stats = {"max_abs_diff": 1.5, "mean_rel_diff": 0.8, "has_nan": False}
    policy = {"atol": 1e-3, "rtol": 1e-3}
    res = evaluate_tolerance(stats, policy)
    assert res["accepted"] is False
    assert res["reason"] == "REJECT_EXCEEDED_TOLERANCE"


def test_metrics_top1_agreement():
    ref = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    tgt = np.array([[1.1, 1.9, 3.1], [1.0, 2.0, 3.0]], dtype=np.float32)
    mets = compute_metrics(ref, tgt)
    assert mets["top1_agreement"] == 0.5
