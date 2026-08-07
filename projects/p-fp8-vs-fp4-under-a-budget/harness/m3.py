def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from quant.evaluator import evaluate_end_to_end
    import ref

    m = {"end_to_end_ok": 0.0}
    weights = ref.get_sample_weights()
    data = ref.get_validation_data()
    score = evaluate_end_to_end(weights, data, "fp8")
    if isinstance(score, float):
        m["end_to_end_ok"] = 1.0
    return m
