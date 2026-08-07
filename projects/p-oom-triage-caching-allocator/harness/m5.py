import ref

def check(workdir):
    m = {"endurance_passed": 0.0}
    try:
        from oom_triage.config import get_allocator_config
        cfg = get_allocator_config()
        peak = ref.run_workload(cfg, steps=5000)
        if peak < 600:
            m["endurance_passed"] = 1.0
    except Exception:
        pass
    return m
