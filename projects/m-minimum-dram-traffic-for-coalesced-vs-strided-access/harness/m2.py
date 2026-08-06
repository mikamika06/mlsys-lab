import ref


def check(workdir):
    from coalesce.simulate import simulate_warp_coalescing

    out = {"simulation_matched": 0.0}
    ok = 0
    for addrs in ref.WARP_TEST_CASES:
        want = ref.simulate_warp_coalescing(addrs, 4)
        got = simulate_warp_coalescing(addrs, 4)
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, want {want}"
    if ok == len(ref.WARP_TEST_CASES):
        out["simulation_matched"] = 1.0
    return out
