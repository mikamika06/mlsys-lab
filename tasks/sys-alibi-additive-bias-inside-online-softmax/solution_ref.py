import numpy as np

def alibi_online_softmax(scores, slopes):
    """Online softmax with ALiBi bias integrated into the streaming loop."""
    n = scores.shape[0]
    probs = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        m = slopes[i]
        running_max = -np.inf
        running_sum = 0.0

        # First pass: compute running max and running sum
        for j in range(n):
            v_j = scores[i, j] + m * (i - j)
            if v_j > running_max:
                running_sum = running_sum * np.exp(running_max - v_j) + 1.0
                running_max = v_j
            else:
                running_sum += np.exp(v_j - running_max)

        # Second pass: compute probabilities using final M and D
        for j in range(n):
            v_j = scores[i, j] + m * (i - j)
            probs[i, j] = np.exp(v_j - running_max) / running_sum

    return probs
