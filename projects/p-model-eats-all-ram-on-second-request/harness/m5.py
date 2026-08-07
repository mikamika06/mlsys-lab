import ref

def check(workdir):
    from runner.config import get_runtime_config

    m = {"parallel_ok": 0.0}
    cfg = get_runtime_config()
    if cfg.get("slots", 0) >= 2 and cfg.get("swap_allowed") is False:
        m["parallel_ok"] = 1.0
    return m
