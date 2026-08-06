import ref

def check(workdir):
    from turnover.model import compute_turnover_metrics
    reqs = ref.get_test_requests()
    want = ref.compute_turnover_metrics(reqs, batch_capacity=2, steps=50)
    got = compute_turnover_metrics(ref.get_test_requests(), batch_capacity=2, steps=50)

    metrics_matched = 0
    if got.get("completed_count") == want.get("completed_count"):
        metrics_matched += 1
    if abs(got.get("avg_turnover", 0) - want.get("avg_turnover", 0)) < 1e-5:
        metrics_matched += 1
    if got.get("max_active_span") == want.get("max_active_span"):
        metrics_matched += 1

    return {"metrics_matched": float(metrics_matched)}
