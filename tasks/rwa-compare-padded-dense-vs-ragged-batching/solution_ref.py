import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def ragged_attention_compare(sequences):
    outputs = []
    for x in sequences:
        x = np.asarray(x, dtype=np.float64)
        d = x.shape[1]
        scores = (x @ x.T) / np.sqrt(d)
        probs = _softmax(scores)
        outputs.append(probs @ x)

    lengths = [len(x) for x in sequences]
    ratio = (len(sequences) * max(lengths) ** 2) / sum(l * l for l in lengths)
    return outputs, ratio
