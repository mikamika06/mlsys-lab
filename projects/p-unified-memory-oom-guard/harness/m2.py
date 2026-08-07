def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from guard.predictor import predict_peak

    m = {"predicted_ok": 0.0}
    cfg = ref.get_sample_config()
    val = predict_peak(cfg)
    if isinstance(val, (int, float)) and val > 0:
        m["predicted_ok"] = 1.0
    return m
