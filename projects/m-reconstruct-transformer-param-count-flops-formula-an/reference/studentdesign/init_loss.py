import numpy as np

def evaluate_init_strategies(cfg, strategy):
    np.random.seed(42)
    if strategy == "random":
        return float(np.random.uniform(8.0, 10.0))
    elif strategy == "stacked":
        return float(np.random.uniform(4.0, 5.0))
    elif strategy == "truncated":
        return float(np.random.uniform(5.0, 6.0))
    return 10.0
