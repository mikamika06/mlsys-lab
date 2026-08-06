import numpy as np


def select_layers_to_prune(bi_scores, num_to_remove):
    indexed = sorted(enumerate(bi_scores), key=lambda x: x[1])
    to_remove = sorted([idx for idx, score in indexed[:num_to_remove]])
    return to_remove


def evaluate_perplexity(model, eval_data):
    total_loss = 0.0
    count = 0
    for sample in eval_data:
        pred = model.forward(sample)
        loss = np.mean((pred - sample) ** 2)
        total_loss += loss
        count += 1
    mean_loss = total_loss / max(1, count)
    return float(np.exp(mean_loss))
