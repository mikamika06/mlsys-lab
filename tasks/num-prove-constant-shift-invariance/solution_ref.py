import math

def _stable_softmax(x):
    """Numerically stable softmax: subtract max, exponentiate, normalize, using plain lists."""
    if isinstance(x[0], list):
        rows = len(x)
        cols = len(x[0])
        out = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            max_val = x[i][0]
            for j in range(1, cols):
                if x[i][j] > max_val:
                    max_val = x[i][j]

            sum_e = 0.0
            for j in range(cols):
                val = math.exp(x[i][j] - max_val)
                out[i][j] = val
                sum_e += val

            for j in range(cols):
                out[i][j] /= sum_e
        return out
    else:
        n = len(x)
        max_val = x[0]
        for i in range(1, n):
            if x[i] > max_val:
                max_val = x[i]

        out = [0.0] * n
        sum_e = 0.0
        for i in range(n):
            val = math.exp(x[i] - max_val)
            out[i] = val
            sum_e += val

        for i in range(n):
            out[i] /= sum_e
        return out

def softmax_shift_invariant(logits, shift):
    """
    Returns the maximum absolute error between softmax(logits) and
    softmax(logits - shift), proving numerical invariance to constant shifts.
    """
    soft_original = _stable_softmax(logits)

    if isinstance(logits[0], list):
        rows = len(logits)
        cols = len(logits[0])
        shifted_logits = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                shifted_logits[i][j] = logits[i][j] - shift[i][j]
    else:
        n = len(logits)
        shifted_logits = [0.0] * n
        for i in range(n):
            shifted_logits[i] = logits[i] - shift[i]

    soft_shifted = _stable_softmax(shifted_logits)

    max_err = 0.0
    if isinstance(soft_original[0], list):
        rows = len(soft_original)
        cols = len(soft_original[0])
        for i in range(rows):
            for j in range(cols):
                val = soft_original[i][j] - soft_shifted[i][j]
                abs_val = -val if val < 0 else val
                if abs_val > max_err:
                    max_err = abs_val
    else:
        n = len(soft_original)
        for i in range(n):
            val = soft_original[i] - soft_shifted[i]
            abs_val = -val if val < 0 else val
            if abs_val > max_err:
                max_err = abs_val

    return float(max_err)
