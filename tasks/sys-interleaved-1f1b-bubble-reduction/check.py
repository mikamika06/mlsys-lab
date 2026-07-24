def _oracle(stages, microbatches, virtual_stages):
    idle_slots = stages - 1
    active_slots = microbatches * virtual_stages
    total_slots = active_slots + idle_slots
    return idle_slots / total_slots


def grade(sol, fx) -> dict:
    cases = [
        (2, 1, 1),
        (4, 8, 2),
        (8, 64, 4),
        (16, 128, 2),
        (3, 7, 5),
    ]
    max_err = 0.0
    for stages, microbatches, virtual_stages in cases:
        try:
            got = float(
                sol.interleaved_1f1b_bubble_fraction(
                    stages, microbatches, virtual_stages
                )
            )
        except Exception:
            return {"rel_err": 1.0}
        ref = _oracle(stages, microbatches, virtual_stages)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
