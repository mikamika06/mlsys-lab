import numpy as np

def _reference(logits: np.ndarray, targets: np.ndarray) -> float:
    # Compute softmax probabilities in a numerically stable way
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(logits - max_logits)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    log_probs = np.log(probs + 1e-12)  # avoid log(0)
    ce = -log_probs[np.arange(len(targets)), targets]
    mean_ce = np.mean(ce)
    return float(np.exp(mean_ce))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    N, V = 100, 50
    logits = rng.normal(size=(N, V))
    targets = rng.integers(low=0, high=V, size=N)

    try:
        got = sol.perplexity_from_cross_entropy(logits, targets)
    except Exception:
        return {"rel_err": 1.0}

    if not isinstance(got, (float, np.floating)):
        return {"rel_err": 1.0}

    ref = _reference(logits, targets)
    rel_err = abs(float(got) - ref) / (abs(ref) + 1e-12)
    return {"rel_err": rel_err}
