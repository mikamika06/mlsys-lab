import numpy as np


def generate_task(length, depth_ratio, needle, haystack_filler="text"):
    rng = np.random.default_rng(42)
    tokens = [haystack_filler] * length
    pos = int(length * depth_ratio)
    tokens[pos] = needle
    return {"tokens": tokens, "depth_ratio": depth_ratio, "needle": needle, "pos": pos}
