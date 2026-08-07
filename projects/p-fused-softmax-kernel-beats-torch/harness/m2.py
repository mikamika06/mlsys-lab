def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    m = {"masked_ok": 0.0}
    try:
        from softmax.kernel import fused_softmax
        import torch
        x = torch.tensor([[1.0, -1e9, 3.0], [2.0, 2.0, -1e9]])
        out = fused_softmax(x)
        expected = torch.softmax(x, dim=-1)
        if torch.allclose(out, expected, atol=1e-4):
            m["masked_ok"] = 1.0
    except Exception:
        pass
    return m
