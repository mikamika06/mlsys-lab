import ref

def check(workdir):
    m = {"profile_ok": 0.0}
    try:
        from cpuopt.profiler import profile_operations
        data = ref.get_sample_data()
        prof = profile_operations(None, data[0])
        if isinstance(prof, dict) and "total" in prof and prof["total"] > 0:
            m["profile_ok"] = 1.0
    except Exception:
        pass
    return m
