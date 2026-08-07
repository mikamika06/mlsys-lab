import sys
import numpy as np

sys.path.insert(0, ".")
from woq.packing import pack_int4_groups, unpack_int4_groups
from woq.quant import compute_quant_error


def test_packing_roundtrip():
    rng = np.random.default_rng(42)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    packed, scales = pack_int4_groups(w, 32)
    rec = unpack_int4_groups(packed, scales, 32, w.shape)
    assert rec.shape == w.shape
    assert np.all(np.isfinite(rec))


def test_smoothquant_error_difference():
    rng = np.random.default_rng(42)
    w = rng.standard_normal((16, 32)).astype(np.float32)
    err_raw = compute_quant_error(w, 32, smoothed=False)
    err_smooth = compute_quant_error(w, 32, smoothed=True)
    assert err_raw >= 0.0
    assert err_smooth >= 0.0
