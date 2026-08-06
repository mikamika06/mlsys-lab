import ref


def check(workdir):
    from profiler.analysis import classify_ncu_kernel

    samples = ref.get_ncu_samples()
    ok = 0
    out = {"classifications_matched": 0.0}
    for i, (ncu_text, expected) in enumerate(samples):
        try:
            got = classify_ncu_kernel(ncu_text)
            if got == expected:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"sample {i}: got {got}, expected {expected}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"sample {i} raised {type(e).__name__}: {str(e)[:100]}"
    out["classifications_matched"] = float(ok)
    return out
