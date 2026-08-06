import numpy as np
from nf4.quant import get_nf4_quantiles


def test_quantiles_endpoints():
    q = get_nf4_quantiles()
    assert len(q) == 16
    assert np.isclose(q[0], -1.0, atol=1e-5)
    assert np.isclose(q[-1], 1.0, atol=1e-5)
    assert q[7] < 0.0 and q[8] > 0.0
