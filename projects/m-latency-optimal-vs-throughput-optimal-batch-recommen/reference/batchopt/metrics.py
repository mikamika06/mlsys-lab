from batchopt.profile import compute_curves


def compute_metrics(profile):
    lat, tp = compute_curves(profile)
    return {"latencies": lat.tolist(), "throughput": tp.tolist()}
