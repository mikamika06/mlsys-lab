def check(workdir):
    import ref
    m = {"burst_ok": 0.0}
    arrivals = [("req1", 1), ("req2", 1), ("req3", 1), ("req4", 1), ("req5", 1)]
    admitted, rejected, _ = ref.run_simulation(arrivals, 2, 1.0, 2.0)
    if admitted != 2 or rejected != 3:
        return m
    m["burst_ok"] = 1.0
    return m
