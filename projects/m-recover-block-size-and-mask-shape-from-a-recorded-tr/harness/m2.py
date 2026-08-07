import ref


def check(workdir):
    from triton_trace.mask import recover_mask_shape

    out = {"masks_matched": 0.0}
    ok = 0
    for events, _, expected_mask_shape in ref.TEST_CASES:
        got = recover_mask_shape(events)
        if got == expected_mask_shape:
            ok += 1
    out["masks_matched"] = float(ok)
    return out
