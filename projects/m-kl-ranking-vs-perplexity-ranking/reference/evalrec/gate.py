def evaluate_acceptance_gate(
    candidate_metrics: dict, category_thresholds: dict
) -> dict:
    cat_results = {}
    failed = []

    for cat in sorted(category_thresholds.keys()):
        thresh = category_thresholds[cat]
        metrics = candidate_metrics.get(
            cat, {"kl": float("inf"), "ppl": float("inf")}
        )
        kl_pass = metrics["kl"] <= thresh["max_kl"]
        ppl_pass = metrics["ppl"] <= thresh["max_ppl"]
        passed = kl_pass and ppl_pass

        cat_results[cat] = {
            "kl_pass": bool(kl_pass),
            "ppl_pass": bool(ppl_pass),
            "passed": bool(passed),
        }

        if not passed:
            failed.append(cat)

    return {
        "accepted": len(failed) == 0,
        "category_results": cat_results,
        "failed_categories": sorted(failed),
    }
