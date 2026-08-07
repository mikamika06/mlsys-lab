import ref

def check(workdir):
    out = {"decisions_match": 0.0}
    try:
        from policy.policy import PreemptionPolicy
        p_learner = PreemptionPolicy(1000.0, 100000.0, 250.0, 2.0, 0.1)
        p_ref = ref.RefPolicy(1000.0, 100000.0, 250.0, 2.0, 0.1)
        trace = ref.generate_trace(100, 42)

        for s in trace:
            if p_learner.decide(s) != p_ref.decide(s):
                return out

        out["decisions_match"] = 1.0
        return out
    except Exception:
        return out
