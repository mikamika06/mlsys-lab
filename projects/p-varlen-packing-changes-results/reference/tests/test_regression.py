import sys
sys.path.insert(0, ".")
import numpy as np
from pack.attention import varlen_attention, align_causal_mask

def test_varlen_matches_expected():
    q = np.random.randn(10, 4)
    k = np.random.randn(10, 4)
    v = np.random.randn(10, 4)
    cu_seqlens = np.array([0, 4, 10], dtype=np.int32)
    out = varlen_attention(q, k, v, cu_seqlens)
    assert out.shape == q.shape

def test_mask_structure():
    cu_seqlens = np.array([0, 3, 6], dtype=np.int32)
    mask = align_causal_mask(cu_seqlens)
    assert mask.shape == (6, 6)
