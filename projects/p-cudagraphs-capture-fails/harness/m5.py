def check(workdir):
    from model.net import test_cudagraph_capture
    m = {"capture_ok": 0.0, "output_match": 0.0}
    try:
        res, match = test_cudagraph_capture()
        if res:
            m["capture_ok"] = 1.0
        if match:
            m["output_match"] = 1.0
    except Exception:
        pass
    return m
