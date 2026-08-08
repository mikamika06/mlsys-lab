import ref


def check(workdir):
    from diagnose.metrics import parse_metrics
    from diagnose.diff import compute_diff
    from diagnose.cost import evaluate_cost

    out = {"cost_match": 0.0, "regression_detected": 0.0}
    try:
        b_parsed = parse_metrics(ref.BASELINE_CSV)
        m_parsed = parse_metrics(ref.MASKED_CSV)
        diff = compute_diff(b_parsed, m_parsed)
        res = evaluate_cost(diff)

        if isinstance(res, dict) and "score" in res:
            out["cost_match"] = 1.0
        if res.get("regression_detected") is True:
            out["regression_detected"] = 1.0
    except Exception as e:
        out["_note"] = f"Error evaluating cost: {str(e)[:120]}"
    return out
