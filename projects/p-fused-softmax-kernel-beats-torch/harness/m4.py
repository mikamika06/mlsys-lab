def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    m = {"max_diff": 1.0}
    try:
        from softmax.kernel import fused_softmax
        import torch
        torch.manual_seed(123)
        x = torch.randn(32, 128)
        out = fused_softmax(x)
        expected = torch.softmax(x, dim=-1)
        diff = torch.max(torch.abs(out - expected)).item()
        m["max_diff"] = diff
    except Exception:
        pass
    return m
