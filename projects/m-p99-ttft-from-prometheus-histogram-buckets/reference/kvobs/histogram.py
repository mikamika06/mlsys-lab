import math


def calculate_histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float:
    """Calculate quantile from Prometheus histogram buckets using linear interpolation."""
    if not buckets:
        return 0.0
    total_count = float(buckets[-1][1])
    if total_count <= 0.0:
        return 0.0
    rank = q * total_count
    if rank <= 0.0:
        return 0.0

    idx = -1
    for i, b in enumerate(buckets):
        if float(b[1]) >= rank:
            idx = i
            break

    if idx == -1:
        return float(buckets[-1][0])

    if idx == 0:
        lower_bound = 0.0
        lower_count = 0.0
        upper_bound = float(buckets[0][0])
        upper_count = float(buckets[0][1])
    else:
        lower_bound = float(buckets[idx - 1][0])
        lower_count = float(buckets[idx - 1][1])
        upper_bound = float(buckets[idx][0])
        upper_count = float(buckets[idx][1])

    if math.isinf(upper_bound):
        return lower_bound

    if upper_count == lower_count:
        return upper_bound

    fraction = (rank - lower_count) / (upper_count - lower_count)
    return lower_bound + fraction * (upper_bound - lower_bound)
