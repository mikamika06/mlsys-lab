import sys
sys.path.insert(0, ".")
import numpy as np
from hadamard.rotation import hadamard_matrix
from hadamard.fused import rms_norm_weight_fuse

def test_fused_weight_matches():
    h = hadamard_matrix(16)
    w = np.random.randn(16, 16).astype(np.float32)
    fused = rms_norm_weight_fuse(w, h)
    expected = np.matmul(h.T, w)
    assert np.allclose(fused, expected, atol=1e-5)
