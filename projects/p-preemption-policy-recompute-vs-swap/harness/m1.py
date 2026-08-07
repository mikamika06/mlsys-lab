import ref

def check(workdir):
    out = {"recompute_match": 0.0}
    try:
        from policy.policy import PreemptionPolicy
        p_learner = PreemptionPolicy(1000.0, 100000.0, 250.0, 2.0, 0.1)
        p_ref = ref.RefPolicy(1000.0, 100000.0, 250.0, 2.0, 0.1)

        for s in [10, 100, 1000, 4000]:
            if abs(p_learner.recompute_time(s) - p_ref.recompute_time(s)) > 1e-5:
                return out

        out["recompute_match"] = 1.0
        return out
    except Exception:
        return out
