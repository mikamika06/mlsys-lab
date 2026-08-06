import numpy as np


def measure_dropout_stochasticity(x, w_a, w_b, lora_alpha, lora_dropout, num_samples, seed=42):
    rng = np.random.RandomState(seed)
    r = w_a.shape[0]
    scaling = lora_alpha / r
    outputs = []
    for _ in range(num_samples):
        if lora_dropout > 0.0:
            mask = (rng.uniform(0.0, 1.0, size=x.shape) >= lora_dropout).astype(np.float64)
            x_eff = (x * mask) / (1.0 - lora_dropout)
        else:
            x_eff = x.astype(np.float64)
        h = np.dot(x_eff, w_a.T)
        out = np.dot(h, w_b.T) * scaling
        outputs.append(out)
    stacked = np.stack(outputs, axis=0)
    var = np.var(stacked, axis=0)
    mean_var = float(np.mean(var))
    is_stochastic = mean_var > 1e-9
    return {
        "mean_variance": mean_var,
        "is_stochastic": is_stochastic,
        "sample_outputs": outputs
    }
