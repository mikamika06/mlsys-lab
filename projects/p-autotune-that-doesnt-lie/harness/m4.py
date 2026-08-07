def check(workdir):
    from autotune.tuner import Autotuner
    m = {"distribution_covered": 0.0}
    configs = [{"block": 16}, {"block": 32}]
    t = Autotuner(configs)
    cfg, _ = t.select((128, 128), (128, 1), lambda c: sum(range(c["block"])))
    if cfg in configs:
        m["distribution_covered"] = 1.0
    return m
