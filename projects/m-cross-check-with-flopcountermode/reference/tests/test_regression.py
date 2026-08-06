from attention.cross_check import rel_err


def test_rel_err_is_zero():
    """Ensure empirical and analytical exact match."""
    err = rel_err(2, 4, 128, 64)
    assert err == 0.0, f"Expected 0.0 error, got {err}"
