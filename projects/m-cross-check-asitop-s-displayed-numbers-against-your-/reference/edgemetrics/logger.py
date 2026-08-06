def detect_drop(samples: list, threshold_ratio: float = 0.5) -> int:
    if not samples:
        return -1
    baseline = sum(samples[:3]) / min(3, len(samples))
    for i, s in enumerate(samples):
        if s < baseline * threshold_ratio:
            return i
    return -1
