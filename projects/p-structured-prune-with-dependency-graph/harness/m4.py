def check(workdir):
    import ref
    m = {"metrics_ok": 0.0}
    model = {"a": ref.np.ones((10, 10))}
    out = ref.evaluate(model, ref.np.ones((1, 10)))
    if out.shape == (1, 10):
        m["metrics_ok"] = 1.0
    return m
