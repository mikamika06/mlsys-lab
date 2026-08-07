import numpy as np


def teacher_forced_loss(tokens, draft_logits):
    logits = draft_logits[:-1]
    targets = tokens[1:]
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    nll = -np.log(probs[np.arange(len(targets)), targets] + 1e-12)
    return float(np.mean(nll))
