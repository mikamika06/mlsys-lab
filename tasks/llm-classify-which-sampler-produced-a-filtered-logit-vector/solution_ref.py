import math


def classify_sampler(orig_logits: list[list[float]], filtered_logits: list[list[float]]) -> str:
    num_rows = len(orig_logits)
    num_cols = len(orig_logits[0])

    counts = []
    kept_mask = []
    for i in range(num_rows):
        row_count = 0
        row_kept = []
        for j in range(num_cols):
            is_k = filtered_logits[i][j] != float("-inf")
            row_kept.append(is_k)
            if is_k:
                row_count += 1
        counts.append(row_count)
        kept_mask.append(row_kept)

    all_ones = True
    for c in counts:
        if c != 1:
            all_ones = False
            break

    if all_ones:
        return "greedy"

    all_equal_first = True
    first_count = counts[0]
    for c in counts:
        if c != first_count:
            all_equal_first = False
            break

    if all_equal_first:
        return "top-k"

    probs = []
    for i in range(num_rows):
        max_logit = orig_logits[i][0]
        for j in range(1, num_cols):
            if orig_logits[i][j] > max_logit:
                max_logit = orig_logits[i][j]

        row_exps = []
        sum_exp = 0.0
        for j in range(num_cols):
            e = math.exp(orig_logits[i][j] - max_logit)
            row_exps.append(e)
            sum_exp += e

        row_probs = []
        for j in range(num_cols):
            row_probs.append(row_exps[j] / sum_exp)
        probs.append(row_probs)

    max_C_prime = 0.0
    min_C = 1.0
    for i in range(num_rows):
        sorted_p = sorted(probs[i], reverse=True)
        k_i = counts[i]

        C_i = 0.0
        for idx in range(k_i):
            C_i += sorted_p[idx]

        if k_i > 1:
            C_prime = 0.0
            for idx in range(k_i - 1):
                C_prime += sorted_p[idx]
        else:
            C_prime = 0.0

        if C_prime > max_C_prime:
            max_C_prime = C_prime
        if C_i < min_C:
            min_C = C_i

    if max_C_prime + 1e-5 < min_C:
        return "top-p"

    max_R = 0.0
    min_K = 1.0
    for i in range(num_rows):
        max_p = probs[i][0]
        for j in range(1, num_cols):
            if probs[i][j] > max_p:
                max_p = probs[i][j]

        min_kept = float("inf")
        for j in range(num_cols):
            if kept_mask[i][j]:
                if probs[i][j] < min_kept:
                    min_kept = probs[i][j]
        K_i = min_kept / max_p

        has_unkept = False
        max_unkept = float("-inf")
        for j in range(num_cols):
            if not kept_mask[i][j]:
                has_unkept = True
                if probs[i][j] > max_unkept:
                    max_unkept = probs[i][j]

        if has_unkept:
            R_i = max_unkept / max_p
        else:
            R_i = 0.0

        if R_i > max_R:
            max_R = R_i
        if K_i < min_K:
            min_K = K_i

    if max_R + 1e-5 < min_K:
        return "min-p"

    return "unknown"
