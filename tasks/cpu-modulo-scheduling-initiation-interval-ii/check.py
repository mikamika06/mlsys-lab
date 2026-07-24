from mlsys.sim import cache as cachesim

def grade(sol, fx) -> dict:
    cases = [
        (3, 5),  # Expected II = 5
        (6, 4),  # Expected II = 6
        (7, 7),  # Expected II = 7
        (2, 10), # Expected II = 10
        (9, 3),  # Expected II = 9
    ]
    ok = 1.0
    for resource_bound, recurrence_bound in cases:
        try:
            got = sol.compute_initiation_interval(resource_bound, recurrence_bound)
        except Exception:
            ok = 0.0
            break
        if got != max(resource_bound, recurrence_bound):
            ok = 0.0
            break
    return {"exact_match": ok}
