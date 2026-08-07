import numpy as np

def evaluate_upgrade_gate(corpus, weights):
    perplexities = []
    for text in corpus:
        tokens = [ord(c) for c in text]
        if not tokens:
            continue
        preds = [(t * weights) % 256 for t in tokens]
        diffs = [abs(t - p) for t, p in zip(tokens, preds)]
        perplexities.append(float(np.mean(diffs) + 1.0))
    mean_ppl = float(np.mean(perplexities)) if perplexities else float("inf")
    return mean_ppl < 50.0
