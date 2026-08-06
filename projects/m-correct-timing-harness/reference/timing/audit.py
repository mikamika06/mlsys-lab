def audit_benchmark(code_str):
    is_flawed = False
    reasons = []
    if "time.time()" in code_str or "time.perf_counter()" in code_str:
        is_flawed = True
        reasons.append("uses host timer instead of cuda events")
    if "synchronize" not in code_str:
        is_flawed = True
        reasons.append("missing cuda synchronization")
    return {"is_flawed": is_flawed, "reasons": reasons}
