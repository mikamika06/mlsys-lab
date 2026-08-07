import sys


def check(workdir):
    sys.path.insert(0, workdir)
    import capacity.planner as planner

    out = {"replicas_correct": 0.0}
    try:
        reps1 = planner.compute_required_replicas(target_rps=40, single_replica_capacity=10, headroom_factor=1.2)
        reps2 = planner.compute_required_replicas(target_rps=40, single_replica_capacity=30, headroom_factor=1.2)
        if reps1 == 5 and reps2 == 2:
            out["replicas_correct"] = 1.0
    except Exception:
        pass

    return out
