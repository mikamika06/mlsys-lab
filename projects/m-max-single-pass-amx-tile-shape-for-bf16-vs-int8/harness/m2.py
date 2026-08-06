import ref


def check(workdir):
    from amxtile.throughput import tmul_vs_avx512_ratio

    out = {"throughput_matched": 0.0}
    ok = 0
    for dtype in ref.DATATYPES:
        want = ref.tmul_vs_avx512_ratio(dtype)
        got = tmul_vs_avx512_ratio(dtype)
        if abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"dtype {dtype}: got {got}, reference {want}"
    if ok == len(ref.DATATYPES):
        out["throughput_matched"] = 1.0
    return out
