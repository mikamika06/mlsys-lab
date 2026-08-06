def decompose_cold_start(timings):
    total = sum(timings.values())
    if total <= 0:
        return {k: 0.0 for k in timings}
    return {k: round(v / total, 4) for k, v in timings.items()}
