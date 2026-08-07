import time
from autotune.metrics import measure_latency

class Autotuner:
    def __init__(self, configs):
        self.configs = configs
        self.cache = {}

    def benchmark(self, fn, args):
        return measure_latency(fn, args)

    def select(self, shapes, strides, work_fn):
        best_cfg = None
        best_time = float("inf")
        for cfg in self.configs:
            key = (tuple(shapes), tuple(strides), tuple(sorted(cfg.items())))
            if key in self.cache:
                lat = self.cache[key]
            else:
                lat = self.benchmark(lambda: work_fn(cfg), [])
                self.cache[key] = lat
            if lat < best_time:
                best_time = lat
                best_cfg = cfg
        return best_cfg, best_time
