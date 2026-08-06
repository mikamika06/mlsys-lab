import ref

def check(workdir):
    from dequant import unpack_6bit_scales_and_mins

    out = {"matches": 0.0}
    ok = 0
    for i, block in enumerate(ref.Q4_FIXTURES):
        scales_bytes = block[4:16]
        want_scales, want_mins = ref.unpack_6bit_scales_and_mins(scales_bytes)
        got_scales, got_mins = unpack_6bit_scales_and_mins(scales_bytes)
        if want_scales == got_scales and want_mins == got_mins:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"fixture {i}: expected scales {want_scales} and mins {want_mins}, got {got_scales} and {got_mins}"

    out["matches"] = float(ok)
    return out
