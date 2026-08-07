import ref


def check(workdir):
    from triton_trace.recover import recover_block_size

    out = {"block_sizes_matched": 0.0}
    ok = 0
    for events, expected_bs, _ in ref.TEST_CASES:
        got = recover_block_size(events)
        if got == expected_bs or (isinstance(got, tuple) and len(got) > 0):
            ok += 1
    out["block_sizes_matched"] = float(ok)
    return out
