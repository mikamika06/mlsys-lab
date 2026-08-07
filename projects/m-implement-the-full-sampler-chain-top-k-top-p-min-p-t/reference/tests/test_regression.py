import sys
import numpy as np

sys.path.insert(0, ".")
from sampler.chain import apply_min_p

def test_min_p_scales_with_max_prob():
    logits = np.array([2.0, 2.0, 2.0, 1.9, 1.9])

    out = apply_min_p(logits.copy(), 0.5)
    assert not np.any(np.isinf(out)), "min_p shouldn't mask if relative prob > p"

    logits2 = np.array([10.0, 9.0, 0.0, -10.0])
    out2 = apply_min_p(logits2.copy(), 0.5)
    assert np.isinf(out2[1]), "min_p should mask out token with prob < max_prob * p"
