import sys
sys.path.insert(0, ".")
from kvquant.cache import KVCacheTracker
from kvquant.quantize import quantize_fp8, dequantize_fp8
from kvquant.calib import calibrate_scale
import numpy as np

def test_tracker_accumulates():
    tracker = KVCacheTracker(2, 4, 64)
    b1 = tracker.record(1, 10)
    assert b1 == 1 * 10 * 2 * 2 * 4 * 64 * 2
    assert tracker.total_bytes() == b1

def test_quantization_roundtrip():
    np.random.seed(42)
    x = np.random.randn(8, 64).astype(np.float32) * 2.5
    scale = calibrate_scale([x])
    q = quantize_fp8(x, scale)
    dq = dequantize_fp8(q, scale)
    assert q.dtype == np.int8
    err = np.max(np.abs(x - dq))
    assert err < 0.1

def test_outlier_handling():
    x = np.zeros((4, 32), dtype=np.float32)
    x[0, 0] = 100.0
    scale = calibrate_scale([x])
    q = quantize_fp8(x, scale)
    dq = dequantize_fp8(q, scale)
    assert np.isfinite(dq).all()
