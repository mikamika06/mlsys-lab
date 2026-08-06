import ref


def check(workdir):
    from kernel_analysis.bottleneck import classify_bottleneck

    ok = 1
    for i, k in enumerate(ref.KERNELS):
        want = ref.classify_bottleneck(k)
        got = classify_bottleneck(k)
        if got != want:
            ok = 0
            return {"classification_match": 0.0, "_note": f"kernel {i}: got {got}, want {want}"}
    return {"classification_match": float(ok)}
