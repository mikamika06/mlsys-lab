def check(workdir):
    from runner.core import split_phases

    m = {"phase_separation_ok": 0.0}
    try:
        res = split_phases([10, 10], [5, 5, 5])
        if isinstance(res, dict) and "prefill" in res and "decode" in res:
            m["phase_separation_ok"] = 1.0
    except Exception:
        pass
    return m
