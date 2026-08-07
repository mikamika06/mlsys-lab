import numpy as np


def compute_throughput_ratio(block_time, striped_time):
    if striped_time <= 0:
        return 0.0
    return float(block_time / striped_time)


def calculate_relative_error(estimated, reference):
    if reference == 0:
        return float(abs(estimated - reference))
    return float(abs(estimated - reference) / abs(reference))
