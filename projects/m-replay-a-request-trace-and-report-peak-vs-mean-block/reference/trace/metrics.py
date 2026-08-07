def compute_occupancy(timeline):
    if not timeline:
        return {"peak": 0, "mean": 0.0}
    peak = max(timeline)
    mean = sum(timeline) / float(len(timeline))
    return {"peak": peak, "mean": mean}
