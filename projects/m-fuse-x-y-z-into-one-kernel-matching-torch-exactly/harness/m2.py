import ref


def check(workdir):
    from triton_ops.fuse import hbm_bytes_saved
    shapes = [
        ((128, 128), (128, 128), (128, 128), (128, 128)),
        ((64, 1, 64), (64, 64), (1, 64, 64), (64, 64, 64))
    ]
    got = hbm_bytes_saved(shapes)
    want = ref.compute_hbm_bytes(shapes)
    match = 1.0 if got == want else 0.0
    return {"bytes_saved_match": match}
