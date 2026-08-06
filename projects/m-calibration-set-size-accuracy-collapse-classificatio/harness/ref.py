import numpy as np


def get_test_data():
    rng = np.random.default_rng(42)
    nodes = []
    for _ in range(5):
        mn = float(rng.uniform(-2.0, 0.0))
        mx = float(rng.uniform(0.0, 2.0))
        nodes.append({"min_val": mn, "max_val": mx, "levels": 256})
    
    sizes = [16, 64, 256, 1024]
    variances = [0.05, 0.5, 2.0, 15.0]
    threshold = 0.1
    return nodes, sizes, variances, threshold
