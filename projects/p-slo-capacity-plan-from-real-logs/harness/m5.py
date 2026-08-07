import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import capacity.planner as planner

    out = {"cost_per_million_correct": 0.0}
    try:
        cost = planner.compute_cost_per_million_tokens(
            replica_count=2,
            hourly_cost_per_replica=4.0,
            rps=40,
            avg_output_tokens=100,
        )
        if abs(cost - 0.5555555) < 1e-4:
            out["cost_per_million_correct"] = 1.0
    except Exception:
        pass

    return out
