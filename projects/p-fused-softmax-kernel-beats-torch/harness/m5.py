def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    m = {"ratio_ok": 0.0}
    try:
        from softmax.kernel import fused_softmax
        import torch
        import time
        sizes = [512, 1024, 2048]
        success = True
        for sz in sizes:
            x = torch.randn(128, sz)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                _ = torch.softmax(x, dim=-1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_ref = time.time() - t0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                _ = fused_softmax(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fused = time.time() - t0

            if t_fused > t_ref * 1.5:
                success = False
                break
        if success:
            m["ratio_ok"] = 1.0
    except Exception:
        pass
    return m
