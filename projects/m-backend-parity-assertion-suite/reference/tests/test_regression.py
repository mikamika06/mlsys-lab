from hf_attn.suite import build_repro_case, assert_parity_on_valid
import numpy as np


def test_ignores_padding_corruption():
    q, k, v, mask = build_repro_case(2, 4, [4, 2])
    
    def ref_fn(q, k, v, mask):
        return np.zeros_like(q)
        
    def test_fn(q, k, v, mask):
        out = np.zeros_like(q)
        out[1, 2:] = 999.0
        return out
        
    diff = assert_parity_on_valid(q, k, v, mask, ref_fn, test_fn)
    assert diff == 0.0


def test_catches_valid_corruption():
    q, k, v, mask = build_repro_case(2, 4, [4, 2])
    
    def ref_fn(q, k, v, mask):
        return np.zeros_like(q)
        
    def test_fn(q, k, v, mask):
        out = np.zeros_like(q)
        out[1, 1] = 999.0
        return out
        
    diff = assert_parity_on_valid(q, k, v, mask, ref_fn, test_fn)
    assert diff > 0.0
