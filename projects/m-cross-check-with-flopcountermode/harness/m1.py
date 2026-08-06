import ref


def check(workdir):
    out = {"tracker_works": 0.0, "matmul_flops": 0.0, "softmax_flops": 0.0}

    try:
        from attention.tracker import FlopCounterMode
        from attention.ops import matmul, softmax
    except ImportError:
        out["_note"] = "ImportError"
        return out

    try:
        with FlopCounterMode() as counter:
            FlopCounterMode.record(10)
            with FlopCounterMode() as inner:
                FlopCounterMode.record(5)
            FlopCounterMode.record(10)
        if counter.total == 20 and inner.total == 5:
            out["tracker_works"] = 1.0
        else:
            out["tracker_works"] = 0.5
    except Exception:
        pass

    try:
        with FlopCounterMode() as c:
            shape = matmul((2, 3, 4), (2, 4, 5))
        if shape == (2, 3, 5) and c.total == ref.ref_matmul_flops((2, 3, 4), (2, 4, 5)):
            out["matmul_flops"] = 1.0
    except Exception:
        pass

    try:
        with FlopCounterMode() as c:
            shape = softmax((2, 3, 5))
        if shape == (2, 3, 5) and c.total == ref.ref_softmax_flops((2, 3, 5)):
            out["softmax_flops"] = 1.0
    except Exception:
        pass

    return out
