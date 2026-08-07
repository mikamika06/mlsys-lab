import ref


def check(workdir):
    from kvquant.alloc import allocate_bits

    cases = [(1000, 500, 10000), (2000, 1000, 20000), (500, 250, 5000)]
    matched = 0
    for w, k, b in cases:
        want = ref.compute_ref_allocation(w, k, b)
        got = allocate_bits(w, k, b)
        if want == got:
            matched += 1
    ok = 1.0 if matched == len(cases) else 0.0
    return {"allocation_matched": ok}
