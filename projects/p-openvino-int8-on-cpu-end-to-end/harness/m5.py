import ref

def check(workdir):
    m = {"latency_budget_ok": 0.0}
    try:
        from cpuopt.optimizer import optimize_pipeline
        data = ref.get_sample_data()
        res = optimize_pipeline("dummy", data, target_latency_ms=80.0)
        if isinstance(res, dict) and res.get("pipeline_ok") and res.get("latency_ms", 100.0) < 80.0:
            m["latency_budget_ok"] = 1.0
    except Exception:
        pass
    return m
