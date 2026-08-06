import ref


def check(workdir):
    from profalyze.kernels import top_time_kernel

    out = {"top_kernel_matched": 0.0}
    trace = ref.TRACES[0]
    want = ref.top_time_kernel(trace)
    got = top_time_kernel(trace)
    if got == want:
        out["top_kernel_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
