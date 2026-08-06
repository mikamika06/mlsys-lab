import ref

def check(workdir):
    from streamstat.metrics import compute_percentiles
    streams = [
        (1, 10, 0.0, [0.1, 0.15, 0.2]),
        (2, 10, 0.0, [0.2, 0.3, 0.4, 0.5]),
        (3, 10, 0.0, [0.05, 0.1, 0.15])
    ]
    want = ref.compute_metrics(streams)
    try:
        got = compute_percentiles(streams)
    except Exception as e:
        return {"metrics_match": 0.0, "_note": f"raised {type(e).__name__}"}

    match = 1.0
    for k in want:
        if abs(want[k] - got.get(k, 0.0)) > 1e-5:
            match = 0.0
            break
    return {"metrics_match": match}
