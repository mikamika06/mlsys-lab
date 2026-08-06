import ref


def check(workdir):
    from amxtile.shape import max_tile_shape

    out = {"shapes_matched": 0.0}
    ok = 0
    for dtype in ref.DATATYPES:
        want = ref.max_tile_shape(dtype)
        got = max_tile_shape(dtype)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"dtype {dtype}: got {got}, reference {want}"
    out["shapes_matched"] = float(ok)
    return out
