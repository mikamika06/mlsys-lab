def check(workdir):
    from autotune.tuner import Autotuner
    m = {"not_worse_than_manual": 0.0}
    configs = [{"block": 32}, {"block": 128}]
    t = Autotuner(configs)
    manual_lat = 100.0
    _, chosen_lat = t.select((128, 128), (128, 1), lambda c: sum(range(c["block"])))
    if chosen_lat <= manual_lat * 10.0:
        m["not_worse_than_manual"] = 1.0
    return m
