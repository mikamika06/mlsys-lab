def _oracle(microbatches, stages):
    p = stages
    m = microbatches
    idle_slots = p - 1
    total_slots = m + p - 1
    return idle_slots / total_slots


def grade(sol, fx) -> dict:
    cases = [
        (1, 1),
        (1, 4),
        (4, 1),
        (8, 4),
        (64, 8),
        (128, 16),
        (7, 3),
        (1000, 2),
        (2, 1000),
        (256, 32),
    ]
    max_err = 0.0
    for microbatches, stages in cases:
        ref = _oracle(microbatches, stages)
        try:
            got = float(sol.gpipe_bubble_fraction(microbatches, stages))
        except Exception:
            return {"rel_err": 1.0}
        err = abs(got - ref) / (abs(ref) + 1e-12)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
