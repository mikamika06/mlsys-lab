def assert_backend_parity(q, k, v, mask=None, backends=("eager", "sdpa"), atol=1e-4, is_causal=False):
    """Checks tensor output parity across attention backends."""
    raise NotImplementedError
