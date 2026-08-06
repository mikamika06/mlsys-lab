import numpy as np

def compute_token_entropy(logits):
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    return float(np.mean(ent))

def compute_vocabulary_coverage(tokens, vocab_size):
    unique_tokens = np.unique(tokens)
    return float(len(unique_tokens) / vocab_size)
