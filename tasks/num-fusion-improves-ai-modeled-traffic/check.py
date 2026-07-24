def _oracle(n, element_size):
    # Three unfused elementwise stages. Each stage reads two arrays and writes one.
    unfused_element_accesses = 3 * (2 + 1) * n
    # One fused pass reads four inputs and writes one output.
    fused_element_accesses = (4 + 1) * n
    return (
        unfused_element_accesses * element_size,
        fused_element_accesses * element_size,
    )


def grade(sol, fx) -> dict:
    cases = [
        (1, 1),
        (10, 4),
        (1024, 8),
        (37, 16),
        (100000, 2),
    ]
    ok = 1.0
    for n, element_size in cases:
        try:
            got = tuple(sol.model_access_count(n, element_size))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(n, element_size):
            ok = 0.0
            break
    return {"modeled_access_count": ok}
