import ref


def check(workdir):
    from loratools.adapter import verify_adapter_shape

    out = {"shapes_matched": 0.0}
    try:
        in_f, out_f, r = 128, 64, 8
        want = ref.expected_adapter_shape(in_f, out_f, r)
        got = verify_adapter_shape(in_f, out_f, r)
        if got == want:
            out["shapes_matched"] = 1.0
        else:
            out["_note"] = f"got shapes {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error in m1: {type(e).__name__}: {str(e)[:120]}"
    return out
