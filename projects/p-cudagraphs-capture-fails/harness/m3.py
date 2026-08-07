def check(workdir):
    from model.net import verify_input_shapes
    m = {"shapes_fixed": 0.0}
    try:
        if verify_input_shapes():
            m["shapes_fixed"] = 1.0
    except Exception:
        pass
    return m
