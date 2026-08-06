import ref


def check(workdir):
    from hlodiff.fusion import count_fusion_kernels

    out = {"fusion_matched": 0.0}
    want = ref.count_fusion_kernels(ref.SAMPLE_FUSION_HLO)
    got = count_fusion_kernels(ref.SAMPLE_FUSION_HLO)
    if got == want:
        out["fusion_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
