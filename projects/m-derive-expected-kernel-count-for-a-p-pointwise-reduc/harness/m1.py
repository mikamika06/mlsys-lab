import ref


def check(workdir):
    from kernelplan.derivation import derive_kernel_count

    out = {"counts_matched": 0.0}
    ok = True
    for i, b in enumerate(ref.BLOCKS):
        got = derive_kernel_count(b["p_ops"], b["has_reduction"], b["q_ops"])
        want = ref.derive_count(b["p_ops"], b["has_reduction"], b["q_ops"])
        if got != want:
            ok = False
            out["_note"] = f"block {b['name']}: got {got}, want {want}"
            break
    if ok:
        out["counts_matched"] = 1.0
    return out
