import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shift = np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits - shift)
    return exps / np.sum(exps, axis=-1, keepdims=True)
