def check(workdir):
    from amx_model import model
    import ref
    m = {"accuracy_ok": 0.0}
    ok = True
    for shape in ref.get_test_shapes():
        mn, nn, kn, dt = shape
        measured = ref.predict_amx(mn, nn, kn, dt)
        if not model.compare_with_measurement(mn, nn, kn, dt, measured):
            ok = False
            break
    if ok:
        m["accuracy_ok"] = 1.0
    return m
