import ref

def check(workdir):
    out = {"breakeven_match": 0.0}
    configs = [
        (1000.0, 100000.0, 250.0, 2.0, 0.1),
        (5000.0, 200000.0, 500.0, 5.0, 0.5),
        (2000.0, 50000.0, 120.0, 1.5, 0.2),
    ]
    try:
        from policy.policy import PreemptionPolicy
        for c in configs:
            p_learner = PreemptionPolicy(*c)
            p_ref = ref.RefPolicy(*c)
            if p_learner.breakeven_seq_len() != p_ref.breakeven_seq_len():
                return out

        out["breakeven_match"] = 1.0
        return out
    except Exception:
        return out
