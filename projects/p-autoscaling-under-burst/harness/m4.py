def check(workdir):
    from scaler.predictor import predict_scaling_action
    m = {"predictive_scaling": 0.0}
    try:
        res = predict_scaling_action(10, 1, 20, 5)
        if isinstance(res, dict) and "action" in res:
            m["predictive_scaling"] = 1.0
    except Exception:
        pass
    return m
