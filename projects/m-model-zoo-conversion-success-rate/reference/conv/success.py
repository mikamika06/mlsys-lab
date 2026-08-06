def compute_success_rate(results):
    if not results:
        return 0.0
    successes = sum(1 for r in results if r.get("success", False))
    return float(successes) / float(len(results))
