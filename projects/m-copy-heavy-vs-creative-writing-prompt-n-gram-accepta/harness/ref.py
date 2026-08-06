WORKLOADS = [
    {
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
        "target": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
        "entropy": 0.5
    },
    {
        "prompt": [10, 20, 30, 40, 50],
        "target": [99, 88, 77, 66, 55],
        "entropy": 4.5
    },
    {
        "prompt": [1, 2, 1, 2, 1, 2, 1, 2],
        "target": [1, 2, 1, 2, 1, 2, 9, 9],
        "entropy": 1.2
    }
]


def build_reference_outputs():
    from ngrameval.analyzer import extract_ngram_matches
    from ngrameval.metrics import compute_acceptance_metrics, classify_workload

    results = []
    for w in WORKLOADS:
        matches = extract_ngram_matches(w["prompt"], w["target"], n=4)
        metrics = compute_acceptance_metrics(matches, w["entropy"])
        cls = classify_workload(metrics["acceptance_rate"], metrics["entropy_gap"])
        results.append({
            "matches": matches,
            "metrics": metrics,
            "classification": cls
        })
    return results
