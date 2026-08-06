import numpy as np

def generate_heavy_tail_samples(seed, size, tail_prob=0.05):
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 5.0, size=size)
    spikes = rng.exponential(scale=200.0, size=size)
    mask = rng.random(size=size) < tail_prob
    return np.maximum(1.0, base + mask * spikes)

def compute_median_vs_mean(samples):
    return float(np.median(samples)), float(np.mean(samples))

CONFIGS = [
    {"seed": 42, "size": 1000, "tail_prob": 0.05},
    {"seed": 123, "size": 1500, "tail_prob": 0.03},
    {"seed": 999, "size": 800, "tail_prob": 0.10}
]

def compute_warmup_inflation(raw_samples, warmup_count):
    cold_mean = float(np.mean(raw_samples))
    warm_samples = raw_samples[warmup_count:]
    warm_mean = float(np.mean(warm_samples))
    inflation = (cold_mean - warm_mean) / warm_mean
    return {"cold_mean": cold_mean, "warm_mean": warm_mean, "inflation": inflation}

WARMUP_CONFIGS = [
    {"seed": 7, "size": 500, "warmup": 50},
    {"seed": 13, "size": 600, "warmup": 100}
]

def check_event_wall_agreement(event_times, wall_times, tolerance=0.1):
    ev = np.array(event_times)
    wl = np.array(wall_times)
    ratios = np.abs(ev - wl) / np.maximum(wl, 1e-8)
    return bool(np.all(ratios <= tolerance))
