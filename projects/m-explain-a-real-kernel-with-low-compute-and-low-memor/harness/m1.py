import ref


def check(workdir):
    from kernel_analysis.metrics import compute_metrics

    out = {"metrics_matched": 0.0, "kernels": float(len(ref.KERNELS))}
    ok = 0
    for i, k in enumerate(ref.KERNELS):
        want = ref.compute_metrics(k)
        got = compute_metrics(k)
        match = True
        for key in want:
            if abs(want[key] - got.get(key, -1.0)) > 1e-3:
                match = False
        if match:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"kernel {i}: got {got}, reference {want}"
    out["metrics_matched"] = float(ok)
    return out
