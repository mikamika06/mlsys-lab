def check(workdir):
    import ref
    m = {"policy_selection_ok": 0.0}
    try:
        p = ref.PreemptionPolicy({"hidden_size": 4096, "num_layers": 32, "tflops": 300.0, "bytes_per_token": 65536, "pcie_bandwidth_gbps": 32.0})
        res = p.decide({"context_len": 100}, {})
        if res in ["recompute", "swap"]:
            m["policy_selection_ok"] = 1.0
    except Exception:
        pass
    return m
