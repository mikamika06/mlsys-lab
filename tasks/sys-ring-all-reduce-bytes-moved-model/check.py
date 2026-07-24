def grade(sol, fx) -> dict:
    """Evaluate the ring all-reduce bytes-moved model.

    Computes the expected send volume 2*(N-1)/N*B for each test case
    and compares against the student's implementation.  No expected
    values are hardcoded.
    """
    test_cases = [
        (1, 1024),       # single rank — zero communication
        (2, 1024),       # two ranks
        (4, 1000),       # non-power-of-two data size
        (8, 4096),       # standard GPU count
        (16, 65536),     # larger cluster
        (128, 1_000_000),# 128-rank training run
        (3, 777),        # odd rank count
        (2, 0),          # zero-size data
        (1, 0),          # degenerate: single rank, zero data
        (64, 1 << 24),   # 64 MiB per rank
    ]

    for n_ranks, data_bytes in test_cases:
        # Oracle: re-derive the expected value from the formula.
        expected = 2.0 * (n_ranks - 1) / n_ranks * data_bytes
        try:
            got = float(sol.ring_allreduce_bytes_moved(n_ranks, data_bytes))
        except Exception:
            return {"modeled_mem_access": 0.0}
        tolerance = 1e-6 * max(1.0, expected)
        if abs(got - expected) > tolerance:
            return {"modeled_mem_access": 0.0}

    return {"modeled_mem_access": 1.0}
