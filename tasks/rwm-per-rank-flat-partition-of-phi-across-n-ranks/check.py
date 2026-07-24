def _oracle(phi, n_ranks, param_bytes, grad_bytes, opt_bytes):
    base = phi // n_ranks
    remainder = phi % n_ranks
    total = []
    for rank in range(n_ranks):
        params = base + (1 if rank < remainder else 0)
        total.append((
            params * param_bytes,
            params * grad_bytes,
            params * opt_bytes,
        ))
    return total


def grade(sol, fx) -> dict:
    cases = [
        (10, 3, 4, 4, 8),
        (100, 8, 2, 2, 12),
        (7, 4, 16, 16, 32),
        (64, 1, 4, 8, 24),
        (103, 10, 1, 1, 4),
    ]

    ok = 1.0
    for case in cases:
        try:
            got = sol.partition_phi(*case)
            got = [tuple(x) for x in got]
        except Exception:
            ok = 0.0
            break

        expected = _oracle(*case)
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
