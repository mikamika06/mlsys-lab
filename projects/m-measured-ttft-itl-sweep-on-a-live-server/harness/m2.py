import ref

def check(workdir):
    from sweep.metrics import compute_blocking
    out = {"blocking_matched": 0.0, "latency_ratio_valid": 0.0}
    latencies = [2.0, 3.0, 15.0, 45.0]
    prompt_tokens = 131072
    want = ref.compute_blocking(latencies, prompt_tokens)
    got = compute_blocking(latencies, prompt_tokens)
    if abs(got - want) < 1e-5:
        out["blocking_matched"] = 1.0
    else:
        out["_note"] = f"got blocking {got}, want {want}"

    if got > 0:
        out["latency_ratio_valid"] = 1.0
    return out
