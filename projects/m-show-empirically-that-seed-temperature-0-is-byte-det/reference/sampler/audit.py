import numpy as np


def recover_parameters(logits, tokens):
    best_temp = 0.0
    min_diff = float("inf")
    for t in [0.0, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5]:
        if t == 0.0:
            pred = np.argmax(logits, axis=-1)
            diff = np.mean(pred != tokens)
            if diff < min_diff:
                min_diff = diff
                best_temp = 0.0
        else:
            scaled = logits / t
            exps = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exps / np.sum(exps, axis=-1, keepdims=True)
            log_probs = np.log(np.maximum(probs[np.arange(len(tokens)), tokens], 1e-12))
            score = -np.mean(log_probs)
            if score < min_diff:
                min_diff = score
                best_temp = t
    return {"temperature": best_temp}
