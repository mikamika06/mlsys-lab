import ref


def check(workdir):
    out = {"strides_matched": 0.0, "is_cl_matched": 0.0}
    try:
        from layout.strides import compute_nhwc_strides, is_channels_last
    except ImportError:
        return out

    ok_strides = 0
    ok_is_cl = 0

    for shape in ref.SHAPES:
        want = ref.compute_nhwc_strides(shape)
        try:
            if compute_nhwc_strides(shape) == want:
                ok_strides += 1
        except Exception:
            pass

    for shape, strides, want in ref.IS_CL_CASES:
        try:
            if is_channels_last(shape, strides) == want:
                ok_is_cl += 1
        except Exception:
            pass

    out["strides_matched"] = float(ok_strides)
    out["is_cl_matched"] = float(ok_is_cl)
    return out
