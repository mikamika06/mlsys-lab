import numpy as np


def compute_split_softmax(x, chunks=2):
    sub_x = np.array_split(x, chunks, axis=-1)
    maxs = [np.max(s, axis=-1, keepdims=True) for s in sub_x]
    global_max = np.maximum.reduce(maxs)

    numerators = []
    denominators = []
    for s, m in zip(sub_x, maxs):
        num = np.exp(s - global_max)
        numerators.append(num)
        denominators.append(np.sum(num, axis=-1, keepdims=True))

    total_denom = sum(denominators)
    return [num / total_denom for num in numerators]
