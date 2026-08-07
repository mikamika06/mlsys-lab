import ref


def check(workdir):
    from roof.gpu import find_flipping_gpu

    out = {"gpu_matched": 0.0}
    want = ref.find_flipping_gpu(ref.GPUS, ref.KERNEL)
    got = find_flipping_gpu(ref.GPUS, ref.KERNEL)
    if got == want:
        out["gpu_matched"] = 1.0
    else:
        out["_note"] = f"expected {want}, got {got}"
    return out
