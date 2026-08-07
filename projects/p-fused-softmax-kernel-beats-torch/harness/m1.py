def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    m = {"interpreted_ok": 0.0}
    try:
        from softmax.kernel import fused_softmax
        import torch
        x = torch.tensor([[1.0, 2.0, 3.0]])
        out = fused_softmax(x)
        if out is not None:
            m["interpreted_ok"] = 1.0
    except Exception:
        pass
    return m
