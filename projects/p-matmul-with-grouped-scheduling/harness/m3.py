import torch


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from triton_matmul.kernel import matmul_grouped

    m = {"grouped_order_ok": 0.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)

    try:
        out = matmul_grouped(a, b)
        if out is not None and out.shape == (256, 256):
            m["grouped_order_ok"] = 1.0
    except Exception:
        pass
    return m
