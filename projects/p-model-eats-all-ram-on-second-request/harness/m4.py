import ref

def check(workdir):
    from runner.memory import optimize_config

    m = {"budget_fit": 0.0}
    cfg = optimize_config(8192, 4000, 32, 4096, 32)
    if cfg.get("slots") == 2 and cfg.get("num_ctx") >= 2048:
        m["budget_fit"] = 1.0
    return m
