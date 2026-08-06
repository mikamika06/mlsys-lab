import ref


def check(workdir):
    from fusion.analyzer import classify_kernels

    out = {"classes_matched": 0.0}
    try:
        got = classify_kernels(ref.KERNELS_DUMP)
        want = ref.classify_kernels(ref.KERNELS_DUMP)
        if got == want:
            out["classes_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = str(e)
    return out
