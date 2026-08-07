def check(workdir):
    import ref
    import numpy as np
    m = {"p95_slo_ok": 0.0}
    arrivals = [(f"r{i}", 1) for i in range(100)]
    _, _, lats = ref.run_simulation(arrivals, 10, 5.0, 5.0)
    if not lats:
        return m
    p95 = float(np.percentile(lats, 95))
    if p95 > 5.0:
        return m
    m["p95_slo_ok"] = 1.0
    return m
