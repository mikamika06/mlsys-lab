import numpy as np

def quantify_warmup_inflation(samples, warmup_count):
    cold_mean = float(np.mean(samples))
    warm_mean = float(np.mean(samples[warmup_count:]))
    return float((cold_mean - warm_mean) / warm_mean)
