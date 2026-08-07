def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    m = {"speedup_ok": 0.0}
    try:
        from spec.eval import speculative_speedup

        s_student = speculative_speedup(4, 0.8, 0.1)
        s_oracle = ref.oracle_speedup(4, 0.8, 0.1)

        if abs(s_student - s_oracle) < 1e-5:
            m["speedup_ok"] = 1.0
    except Exception:
        pass
    return m
