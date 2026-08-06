import ref


def check(workdir):
    from varlen.seqlens import compute_cu_seqlens
    cases = ref.generate_cases()
    ok = 0
    out = {"seqlens_matched": 0.0}
    for i, mask in enumerate(cases):
        want = ref.ref_compute_cu_seqlens(mask)
        got = compute_cu_seqlens(mask)
        if got is not None and len(got) == len(want) and (got == want).all():
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    out["seqlens_matched"] = 1.0 if ok == len(cases) else 0.0
    return out
