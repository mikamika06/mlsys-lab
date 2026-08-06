import math
import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    if len(shape) == 1:
        n = shape[0]
        res = np.zeros(n, dtype=np.float64)
        row_max = -float('inf')
        for i in range(n):
            if x[i] > row_max:
                row_max = x[i]
        row_sum = 0.0
        for i in range(n):
            val = math.exp(x[i] - row_max)
            res[i] = val
            row_sum += val
        for i in range(n):
            res[i] /= row_sum
        return res
    elif len(shape) == 2:
        rows, cols = shape
        res = np.zeros((rows, cols), dtype=np.float64)
        for i in range(rows):
            row_max = -float('inf')
            for j in range(cols):
                if x[i, j] > row_max:
                    row_max = x[i, j]
            row_sum = 0.0
            for j in range(cols):
                val = math.exp(x[i, j] - row_max)
                res[i, j] = val
                row_sum += val
            for j in range(cols):
                res[i, j] /= row_sum
        return res
    else:
        raise NotImplementedError()


def ragged_attention_compare(sequences):
    outputs = []
    for x in sequences:
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        d = x.shape[1]
        
        scores = np.zeros((n, n), dtype=np.float64)
        sqrt_d = math.sqrt(d)
        for i in range(n):
            for j in range(n):
                val = 0.0
                for k in range(d):
                    val += x[i, k] * x[j, k]
                scores[i, j] = val / sqrt_d
                
        probs = _softmax(scores)
        
        out = np.zeros((n, d), dtype=np.float64)
        for i in range(n):
            for j in range(d):
                val = 0.0
                for k in range(n):
                    val += probs[i, k] * x[k, j]
                out[i, j] = val
        outputs.append(out)

    lengths = []
    for x in sequences:
        lengths.append(len(x))
        
    max_len = 0
    for l in lengths:
        if l > max_len:
            max_len = l
            
    sum_sq = 0
    for l in lengths:
        sum_sq += l * l
        
    ratio = (len(sequences) * max_len ** 2) / sum_sq
    return outputs, ratio
