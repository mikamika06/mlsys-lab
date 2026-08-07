import torch
import time


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from triton_matmul.kernel import matmul_grouped

    m = {"perf_ratio": 0.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        m["perf_ratio"] = 1.0
        return m

    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    lib_time = time.time() - start

    try:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = matmul_grouped(a, b)
        torch.cuda.synchronize()
        my_time = time.time() - start

        ratio = lib_time / max(my_time, 1e-6)
        m["perf_ratio"] = float(min(max(ratio, 0.0), 2.0))
    except Exception:
        pass
    return m
