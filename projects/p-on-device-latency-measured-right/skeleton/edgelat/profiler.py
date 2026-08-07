import numpy as np


class LatencyProfiler:
    def __init__(self, raw_measurements):
        self.raw = raw_measurements

    def filter_warmup_and_throttle(self, warmup_count, throttle_window):
        raise NotImplementedError

    def separate_first_and_steady(self):
        raise NotImplementedError

    def measure_cold_start(self):
        raise NotImplementedError

    def required_sample_size(self, alpha, error_margin):
        raise NotImplementedError

    def multi_session_intervals(self, sessions):
        raise NotImplementedError

    def manager_report(self):
        raise NotImplementedError
