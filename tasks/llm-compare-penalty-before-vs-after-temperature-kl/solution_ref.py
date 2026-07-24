import numpy as np

def compare_penalty_temperature(logits, penalty_fn, temperature):
    logits = np.asarray(logits, dtype=np.float64)
    # before temp
    penalized_before = penalty_fn(logits)
    scaled_before = penalized_before / temperature
    exp_before = np.exp(scaled_before - np.max(scaled_before, axis=-1, keepdims=True))
    probs_before = exp_before / np.sum(exp_before, axis=-1, keepdims=True)

    # after temp
    scaled_after = logits / temperature
    penalized_after = penalty_fn(scaled_after)
    exp_after = np.exp(penalized_after - np.max(penalized_after, axis=-1, keepdims=True))
    probs_after = exp_after / np.sum(exp_after, axis=-1, keepdims=True)

    kl_rows = np.sum(probs_before * (np.log(probs_before + 1e-12) - np.log(probs_after + 1e-12)), axis=1)
    return float(np.mean(kl_rows))
