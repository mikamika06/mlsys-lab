from imatrix.derive import compute_required_tokens


def test_derivation_basic():
    var = 4.0
    se = 0.5
    assert compute_required_tokens(var, se) == 16


def test_derivation_zero_variance():
    assert compute_required_tokens(0.0, 1.0) == 0
