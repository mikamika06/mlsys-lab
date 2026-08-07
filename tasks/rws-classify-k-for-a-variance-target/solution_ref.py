def classify_k_for_variance_target(eigenvalues: list[float], target: float) -> int:
    total = 0.0
    for i in range(len(eigenvalues)):
        total += eigenvalues[i]

    cum = 0.0
    idx = 0
    for i in range(len(eigenvalues)):
        cum += eigenvalues[i]
        ratio = cum / total
        if ratio >= target:
            idx = i
            break
    else:
        idx = len(eigenvalues)

    return int(idx + 1)
