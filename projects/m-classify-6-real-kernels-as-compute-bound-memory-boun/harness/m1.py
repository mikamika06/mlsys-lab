import ref


def check(workdir):
    from kernelperf.classify import classify_all
    got = classify_all(ref.KERNELS)
    want = ref.reference_classify(ref.KERNELS)
    out = {"classification_match": 0.0}
    if got == want:
        out["classification_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
