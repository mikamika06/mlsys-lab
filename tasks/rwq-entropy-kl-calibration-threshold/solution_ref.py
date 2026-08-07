import math


def _kl_for_threshold(activations, edges, hist, k):
    threshold = edges[k]
    qhist = [0] * (len(edges) - 1)
    for x in activations:
        clipped = x if x < threshold else threshold
        levels = math.floor((clipped / threshold) * k)
        if levels > k - 1:
            levels = k - 1
        reconstructed = ((levels + 0.5) / k) * threshold
        bin_idx = -1
        if reconstructed == edges[-1]:
            bin_idx = len(edges) - 2
        else:
            for i in range(len(edges) - 1):
                if edges[i] <= reconstructed < edges[i + 1]:
                    bin_idx = i
                    break
        if bin_idx != -1:
            qhist[bin_idx] += 1

    sum_p = 0.0
    for i in range(len(hist)):
        sum_p += float(hist[i])

    p_norm = [0.0] * len(hist)
    for i in range(len(hist)):
        p_norm[i] = float(hist[i]) / sum_p

    sum_q = 0.0
    for i in range(len(qhist)):
        sum_q += float(qhist[i])

    q_norm = [0.0] * len(qhist)
    for i in range(len(qhist)):
        q_norm[i] = float(qhist[i]) / sum_q

    eps = 1e-12
    kl_sum = 0.0
    for i in range(len(p_norm)):
        pi = p_norm[i]
        qi = q_norm[i]
        kl_sum += pi * (math.log(pi + eps) - math.log(qi + eps))
    return float(kl_sum)


def calibrate_threshold_index(activations: list[float], num_bins: int, candidate_indices: list[int]) -> int:
    max_val = activations[0]
    for val in activations:
        if val > max_val:
            max_val = val
    max_val = float(max_val)

    edges = [0.0] * (num_bins + 1)
    for i in range(num_bins + 1):
        edges[i] = i * max_val / num_bins

    hist = [0] * num_bins
    for x in activations:
        bin_idx = -1
        if x == edges[-1]:
            bin_idx = len(edges) - 2
        else:
            for i in range(len(edges) - 1):
                if edges[i] <= x < edges[i + 1]:
                    bin_idx = i
                    break
        if bin_idx != -1:
            hist[bin_idx] += 1

    best_index = int(candidate_indices[0])
    best_kl = _kl_for_threshold(activations, edges, hist, best_index)

    for k in candidate_indices[1:]:
        k = int(k)
        kl = _kl_for_threshold(activations, edges, hist, k)
        if kl < best_kl:
            best_kl = kl
            best_index = k

    return best_index
