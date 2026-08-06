import numpy as np

TASKS = ["mmlu", "gsm8k", "humaneval"]

def get_reference_evals(model_path):
    rng = np.random.RandomState(abs(hash(model_path)) % (2**32))
    results = {}
    for task in TASKS:
        n_samples = 100
        scores = rng.binomial(1, 0.75 if "base" in model_path else 0.70, size=n_samples).astype(float)
        results[task] = scores.tolist()
    return results

def compute_reference_ci(base_scores, quant_scores, num_samples=1000, alpha=0.05, seed=42):
    base = np.array(base_scores, dtype=float)
    quant = np.array(quant_scores, dtype=float)
    n = len(base)
    rng = np.random.RandomState(seed)
    recoveries = []
    for _ in range(num_samples):
        idx = rng.randint(0, n, size=n)
        b_mean = np.mean(base[idx])
        q_mean = np.mean(quant[idx])
        rec = (q_mean / b_mean) * 100.0 if b_mean > 0 else 0.0
        recoveries.append(rec)
    recoveries = np.array(recoveries)
    lower = float(np.percentile(recoveries, 100 * (alpha / 2)))
    upper = float(np.percentile(recoveries, 100 * (1 - alpha / 2)))
    mean_rec = float(np.mean(recoveries))
    return {"mean": mean_rec, "lower": lower, "upper": upper}
