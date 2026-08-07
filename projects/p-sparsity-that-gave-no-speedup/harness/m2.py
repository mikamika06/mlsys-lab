def check(workdir):
    from sparsity.engine import Device, get_path
    m = {"dense_fb": 0.0, "hw_fb": 0.0, "sparse_ok": 0.0}
    d1 = Device("T4", False, 300.0, 8000.0)
    d2 = Device("A100", True, 1500.0, 312000.0)

    try:
        if get_path(False, d1) == "dense":
            m["dense_fb"] = 1.0
        if get_path(True, d1) == "dense":
            m["hw_fb"] = 1.0
        if get_path(True, d2) == "sparse_2_4":
            m["sparse_ok"] = 1.0
    except Exception:
        pass

    return m
