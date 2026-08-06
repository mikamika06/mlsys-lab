import ref


def check(workdir):
    from ngrameval.analyzer import extract_ngram_matches
    from ngrameval.metrics import compute_acceptance_metrics, classify_workload

    ref_outs = ref.build_reference_outputs()
    ok_acc = 0
    ok_ent = 0
    ok_cls = 0

    for i, w in enumerate(ref.WORKLOADS):
        matches = extract_ngram_matches(w["prompt"], w["target"], n=4)
        try:
            metrics = compute_acceptance_metrics(matches, w["entropy"])
            cls = classify_workload(metrics["acceptance_rate"], metrics["entropy_gap"])
        except Exception:
            metrics = {}
            cls = ""

        want = ref_outs[i]
        if abs(metrics.get("acceptance_rate", -1) - want["metrics"]["acceptance_rate"]) < 1e-5:
            ok_acc += 1
        if abs(metrics.get("entropy_gap", -1) - want["metrics"]["entropy_gap"]) < 1e-5:
            ok_ent += 1
        if cls == want["classification"]:
            ok_cls += 1

    out = {
        "acceptance_rate_match": 1.0 if ok_acc == len(ref.WORKLOADS) else 0.0,
        "entropy_gap_match": 1.0 if ok_ent == len(ref.WORKLOADS) else 0.0,
        "workload_classification_match": 1.0 if ok_cls == len(ref.WORKLOADS) else 0.0
    }
    return out
