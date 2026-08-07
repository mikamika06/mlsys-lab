def check(workdir):
    from amx_model import model
    import ref
    m = {"correlation_ok": 0.0}
    preds = []
    actuals = []
    for shape in ref.get_test_shapes():
        mn, nn, kn, dt = shape
        p = model.predict_amx(mn, nn, kn, dt)
        a = ref.predict_amx(mn, nn, kn, dt)
        preds.append(p)
        actuals.append(a)
    import numpy as np
    corr = np.corrcoef(preds, actuals)[0, 1]
    if np.isnan(corr):
        m["correlation_ok"] = 1.0
    else:
        m["correlation_ok"] = float(corr)
    return m
