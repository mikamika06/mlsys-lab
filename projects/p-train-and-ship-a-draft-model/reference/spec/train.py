import numpy as np

def train_draft(draft, dataset, target_probs_dict, lr=0.1, epochs=5):
    for _ in range(epochs):
        for token in dataset:
            target_p = target_probs_dict[token]
            h = draft.W1[token]
            logits = h @ draft.W2
            logits -= np.max(logits)
            exp = np.exp(logits)
            q = exp / np.sum(exp)

            d_logits = q - target_p
            d_W2 = np.outer(h, d_logits)
            d_h = d_logits @ draft.W2.T

            draft.W2 -= lr * d_W2
            draft.W1[token] -= lr * d_h
    return draft
