import ref


def check(workdir):
    from realignment.metrics import compute_metrics
    from realignment.evaluate import evaluate_uad

    out = {"metrics_match": 0.0, "overhead_valid": 0.0, "throughput_ratio": 0.0}
    cfg = ref.CONFIGS[0]
    mapping = ref.align_tokens(cfg["draft_tokens"], cfg["target_tokens"])
    want_metrics = ref.compute_metrics(cfg["draft_tokens"], cfg["target_tokens"], mapping, 2.0)
    got_metrics = compute_metrics(cfg["draft_tokens"], cfg["target_tokens"], mapping, 2.0)

    if abs(got_metrics.get("acceptance_rate", -1) - want_metrics["acceptance_rate"]) < 1e-5:
        out["metrics_match"] = 1.0

    if got_metrics.get("overhead_ms", 0.0) >= 0.0:
        out["overhead_valid"] = 1.0

    eval_got = evaluate_uad(cfg)
    eval_ref = ref.evaluate_uad(cfg)
    if eval_got.get("throughput_ratio", 1.0) >= 0.5 or "effective_throughput" in eval_got.get("metrics", {}):
        out["throughput_ratio"] = 1.0

    return out
