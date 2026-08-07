def check(workdir):
    from amx_model import model
    import ref
    m = {"crossover_ok": 0.0}
    shapes = [(s[0], s[1], s[2]) for s in ref.get_test_shapes()]
    try:
        res = model.find_crossover(shapes, "int8")
        if res in shapes:
            m["crossover_ok"] = 1.0
    except Exception:
        pass
    return m
