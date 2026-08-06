import ref


def check(workdir):
    from kernelplan.comparison import compare_scheduler_and_kernels

    out = {"comparison_matched": 0.0}
    got = compare_scheduler_and_kernels(ref.BLOCKS)
    want = ref.compare_blocks(ref.BLOCKS)

    if got == want:
        out["comparison_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
