def check(workdir):
    import numpy as np
    from speculative.verifier import verify_distribution
    m = {"distribution_match": 0.0}
    try:
        tp = np.array([0.5, 0.5])
        dp = np.array([0.4, 0.6])
        if verify_distribution(tp, dp):
            m["distribution_match"] = 1.0
    except Exception:
        pass
    return m
