import numpy as np


class LatencyProfiler:
    def __init__(self, raw_measurements):
        self.raw = np.array(raw_measurements)

    def filter_warmup_and_throttle(self, warmup_count, throttle_window):
        if len(self.raw) <= warmup_count + throttle_window:
            return self.raw
        trimmed = self.raw[warmup_count:-throttle_window]
        return trimmed

    def separate_first_and_steady(self):
        if len(self.raw) == 0:
            return 0.0, 0.0
        first = float(self.raw[0])
        steady = float(np.median(self.raw[1:])) if len(self.raw) > 1 else first
        return first, steady

    def measure_cold_start(self):
        if len(self.raw) == 0:
            return 0.0
        return float(self.raw[0])

    def required_sample_size(self, alpha, error_margin):
        if len(self.raw) < 2:
            return 1
        std = np.std(self.raw, ddof=1)
        z = 1.96 if alpha == 0.05 else 2.58
        n = (z * std / error_margin) ** 2
        return int(np.ceil(n))

    def multi_session_intervals(self, sessions):
        intervals = []
        for s in sessions:
            arr = np.array(s)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
            intervals.append((float(mean - std), float(mean + std)))
        return intervals

    def manager_report(self):
        first, steady = self.separate_first_and_steady()
        return {
            "first": first,
            "steady": steady,
            "mean": float(np.mean(self.raw)),
            "std": float(np.std(self.raw))
        }
