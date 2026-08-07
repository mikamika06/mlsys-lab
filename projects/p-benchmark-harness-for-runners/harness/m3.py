def check(workdir):
    from runner.core import required_samples

    m = {"samples_calc_ok": 0.0}
    try:
        n = required_samples(std_dev=2.0, target_width=1.0)
        if isinstance(n, (int, float)) and n > 0:
            m["samples_calc_ok"] = 1.0
    except Exception:
        pass
    return m
