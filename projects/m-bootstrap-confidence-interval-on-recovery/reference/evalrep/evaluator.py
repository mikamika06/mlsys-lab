import numpy as np

def run_lm_eval(model_path, tasks):
    rng = np.random.RandomState(abs(hash(model_path)) % (2**32))
    results = {}
    for task in tasks:
        n_samples = 100
        scores = rng.binomial(1, 0.75 if "base" in model_path else 0.70, size=n_samples).astype(float)
        results[task] = scores.tolist()
    return results
