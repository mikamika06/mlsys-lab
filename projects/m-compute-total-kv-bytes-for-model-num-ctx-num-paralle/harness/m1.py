import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import slots.memory as m
    except ImportError:
        return {"kv_bytes_match": 0.0, "max_parallel_match": 0.0}

    import ref

    cases = [
        (32, 8, 128, 8192, 1, 2),
        (40, 8, 128, 8192, 4, 2),
        (32, 32, 128, 2048, 8, 4),
        (80, 8, 128, 131072, 2, 2)
    ]

    cases_max = [
        (32, 8, 128, 8192, 2, 24*1024**3, 16*1024**3),
        (40, 8, 128, 8192, 2, 80*1024**3, 35*1024**3),
        (32, 8, 128, 8192, 2, 16*1024**3, 16*1024**3),
        (32, 8, 128, 8192, 2, 8*1024**3, 16*1024**3)
    ]

    bytes_match = 0
    for c in cases:
        try:
            if m.compute_kv_bytes(*c) == ref.compute_kv_bytes(*c):
                bytes_match += 1
        except Exception:
            pass

    max_p_match = 0
    for c in cases_max:
        try:
            if m.max_feasible_parallel(*c) == ref.max_feasible_parallel(*c):
                max_p_match += 1
        except Exception:
            pass

    return {
        "kv_bytes_match": float(bytes_match) / len(cases),
        "max_parallel_match": float(max_p_match) / len(cases_max)
    }
