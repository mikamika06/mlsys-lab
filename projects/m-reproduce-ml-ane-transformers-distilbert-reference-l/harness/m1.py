import ref


def check(workdir):
    from transformer.ane_distilbert import evaluate_latencies

    out = {"latency_ratio_match": 0.0}
    try:
        user_latencies = evaluate_latencies()
        if not isinstance(user_latencies, dict):
            out["_note"] = "evaluate_latencies must return a dictionary"
            return out
        match = ref.compute_latency_ratio(user_latencies)
        out["latency_ratio_match"] = float(match)
        if match == 0.0:
            out["_note"] = f"Got latencies {user_latencies}, expected order matching reference"
    except Exception as e:
        out["_note"] = f"Error during evaluation: {type(e).__name__}: {str(e)[:120]}"
    return out
