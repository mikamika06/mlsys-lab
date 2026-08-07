def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    m = {"autotune_ok": 0.0}
    try:
        from softmax.kernel import fused_softmax
        import torch
        x = torch.randn(64, 256)
        out = fused_softmax(x)
        if out.shape == x.shape:
            m["autotune_ok"] = 1.0
    except Exception:
        pass
    return m
